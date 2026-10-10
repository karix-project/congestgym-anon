"""Tester/QA re-test probes, written independently of the developer's repro
scripts. Each probe prints PASS/FAIL/NOTE and never stops the run.

    python qa/retest/retest_probes.py
"""
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import gymnasium as gym  # noqa: E402

import carla_gym  # noqa: E402,F401
from carla_gym.baselines import POLICIES  # noqa: E402
from carla_gym.core.network import NetworkModel  # noqa: E402
from carla_gym.core.trace_traffic import TraceTrafficGenerator  # noqa: E402
from carla_gym.core.traffic import Packet, QoSClass  # noqa: E402
from carla_gym.env import CarlaGymEnv  # noqa: E402
from carla_gym.marl_env import MultiAgentCarlaGymEnv  # noqa: E402
from carla_gym.utils.stats import ci95  # noqa: E402

RESULTS = []


def probe(name):
    def deco(fn):
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001
            status, detail = "FAIL", f"crashed: {type(exc).__name__}: {exc} | {traceback.format_exc().splitlines()[-3].strip()}"
        RESULTS.append((status, name, detail))
        print(f"[{status}] {name}: {detail}", flush=True)
        return fn
    return deco


TOPOS = ["grid", "ring", "star", "mesh", "random"]


# ------------------------------------------------------------------ item 3: H1 through SB3
@probe("H1 via SB3 DummyVecEnv auto-reset: traffic differs per episode and per env")
def _():
    from argparse import Namespace
    from stable_baselines3.common.vec_env import DummyVecEnv
    from examples.train_agent import make_env
    a = Namespace(trace=None, topology="grid", nodes=9, steps=20, budget=0.8, load=104.0, load_scale=1.0)
    v = DummyVecEnv([make_env(a, 0 + r) for r in range(4)])
    v.seed(0)
    v.reset()
    seqs = {(e, ep): [] for e in range(4) for ep in range(4)}
    for t in range(80):
        _, _, _, infos = v.step(np.zeros((4, 24), np.float32))
        for e in range(4):
            seqs[(e, t // 20)].append(round(infos[e]["offered_units"], 4))
    keys = list(seqs)
    dup = [(i, j) for i, j in itertools.combinations(keys, 2) if seqs[i] == seqs[j]]
    return ("PASS" if not dup else "FAIL"), f"16 (env, episode) traffic sequences, identical pairs: {len(dup)}"


@probe("H1 same constructor seed in two processes gives identical first episode")
def _():
    code = ("import numpy as np;from carla_gym.env import CarlaGymEnv;e=CarlaGymEnv(seed=3,max_steps=30);e.reset();"
            "print(round(sum(e.step(np.full(24,0.6,np.float32))[1] for _ in range(30)),10))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout.strip() for h in ("1", "2", "3")}
    return ("PASS" if len(outs) == 1 and "" not in outs else "FAIL"), f"episode return across 3 processes: {outs}"


# ------------------------------------------------------------------ items 4, 5: simulator invariants under fuzz
def fuzz(topology, seed, steps=150, **kw):
    env = CarlaGymEnv(topology=topology, num_nodes=9, max_steps=steps, seed=seed, **kw)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    net = env.network
    offered = delivered = dropped = 0.0
    rmin, rmax = 9.0, -9.0
    problems = []
    mode = seed % 4
    for t in range(steps):
        if mode == 0:
            a = rng.uniform(0, 1, 2 * env.num_edges)
        elif mode == 1:                                  # sparse: many links starved
            a = rng.uniform(0, 1, 2 * env.num_edges) * (rng.uniform(0, 1, 2 * env.num_edges) < 0.4)
        elif mode == 2:                                  # tiny allocations, changing routes
            a = np.concatenate([rng.uniform(0, 0.15, env.num_edges), rng.uniform(0, 1, env.num_edges)])
        else:                                            # bursts of 0 then 1
            a = np.full(2 * env.num_edges, float((t // 7) % 2))
            a[env.num_edges:] = rng.uniform(0, 1, env.num_edges)
        before = {id(p): (p.hop_index, p.sent) for b in net.buffers.values() for p in b}
        obs, r, term, trunc, info = env.step(a)
        offered += info["offered_units"]; delivered += info["delivered_units"]; dropped += info["dropped_units"]
        rmin, rmax = min(rmin, r), max(rmax, r)
        if not env.observation_space.contains(obs) or not np.isfinite(r):
            problems.append(f"t{t}: bad obs/reward")
        if np.any(net.get_node_occupancy() > net.get_node_buffers() + 1e-6):
            problems.append(f"t{t}: buffer overflow")
        bw = env._apply_budget(np.clip(a[:env.num_edges], 0, 1).astype(np.float32))
        if np.any(net._served > net.get_edge_capacities() * bw + 1e-4):
            problems.append(f"t{t}: link served above allocation")
        if env.bandwidth_budget is not None and bw.sum() > env.bandwidth_budget * env.num_edges + 1e-4:
            problems.append(f"t{t}: budget exceeded")
        for node, buf in net.buffers.items():
            for i, p in enumerate(buf):
                if not (0.0 <= p.sent < p.size + 1e-9):
                    problems.append(f"t{t}: sent out of range")
                if p.sent > 0 and i != 0:
                    problems.append(f"t{t}: partially-sent packet not at head of queue")
                if p.path[p.hop_index] != node:
                    problems.append(f"t{t}: packet stored at wrong node")
                old = before.get(id(p))
                if old is not None:
                    if p.hop_index - old[0] > 1:
                        problems.append(f"t{t}: packet moved more than one hop")
                    if p.hop_index != old[0] and p.sent != 0.0:
                        problems.append(f"t{t}: sent carried over to a different link")
                    if p.hop_index == old[0] and p.sent + 1e-9 < old[1]:
                        problems.append(f"t{t}: sent decreased without moving")
        for k in ("reward_throughput", "reward_delay", "reward_loss", "reward_deadline_miss"):
            if not 0.0 <= info[k] <= 1.0:
                problems.append(f"t{t}: {k} out of [0,1]")
    if abs(offered - delivered - dropped - net.backlog_units()) > 1e-6 * max(offered, 1):
        problems.append("conservation violated")
    return problems, rmin, rmax


@probe("Simulator invariants under fuzz (5 topologies x 8 seeds x budgets/loads/bottlenecks)")
def _():
    allp, rmin, rmax, n = [], 9.0, -9.0, 0
    for topo in TOPOS:
        for seed in range(8):
            kw = [dict(bandwidth_budget=0.8), dict(bandwidth_budget=None, load_scale=2.5),
                  dict(bandwidth_budget=0.5, load_scale=0.4),
                  dict(bandwidth_budget=1.0, bottlenecks={0: 0.3, 1: 0.6})][seed % 4]
            p, lo, hi = fuzz(topo, seed, **kw)
            n += 1
            allp += [f"{topo}/s{seed}: {x}" for x in p[:2]]
            rmin, rmax = min(rmin, lo), max(rmax, hi)
    ok = not allp and -2.0 - 1e-9 <= rmin and rmax <= 1.0 + 1e-9
    return ("PASS" if ok else "FAIL"), f"{n} runs x 150 steps; reward range seen [{rmin:.3f}, {rmax:.3f}] (documented [-2, 1]); problems: {allp[:4] or 'none'}"


@probe("Dropped packet consumes link budget; delivered packet does not need destination buffer room")
def _():
    net = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=20.0, seed=0)
    net.update_routing(np.ones(net.num_edges()))
    # node 1's buffer is full; node 0 forwards a transit packet 0->1->2 into it
    net.admit([Packet(1, 3, QoSClass.VIDEO, 20.0, 1.5, 8, 0)])
    net.admit([Packet(0, 2, QoSClass.VIDEO, 10.0, 1.5, 8, 0)])
    path = net.buffers[0][0].path
    alloc = np.zeros(net.num_edges()); ei = net._edge_index[net._key(path[0], path[1])]
    alloc[ei] = 1.0
    st = net.forward(alloc, 0)
    served = net._served[ei]
    # destination with a full buffer still accepts a packet addressed to it
    net2 = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=20.0, seed=0)
    net2.update_routing(np.ones(net2.num_edges()))
    net2.admit([Packet(1, 3, QoSClass.VIDEO, 20.0, 1.5, 8, 0)])
    net2.admit([Packet(0, 1, QoSClass.VIDEO, 10.0, 1.5, 8, 0)])
    st2 = net2.forward(np.ones(net2.num_edges()), 0)
    ok = st["dropped_units"] == 10.0 and served == 10.0 and st2["delivered_units"] >= 10.0
    return ("PASS" if ok else "FAIL"), f"transit into full node: dropped {st['dropped_units']}, link served {served}; delivery to full destination: {st2['delivered_units']}"


# ------------------------------------------------------------------ items 6, 7: deadline accounting
@probe("M1 deadline boundary and denominator behaviour")
def _():
    net = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=1000.0, seed=0)
    net.update_routing(np.ones(net.num_edges()))
    net.admit([Packet(0, 1, QoSClass.VOICE, 2.0, 2.0, 3, 0)])
    z = np.zeros(net.num_edges())
    exp = [net.forward(z, s)["queue_expired"] for s in range(6)]
    late = net.forward(np.ones(net.num_edges()), 6)
    # on-time boundary: delivered with delay == deadline must not be a miss
    net.reset(); net.update_routing(np.ones(net.num_edges()))
    net.admit([Packet(0, 1, QoSClass.VOICE, 2.0, 2.0, 3, 0)])
    for s in range(2):
        net.forward(z, s)
    edge = net.forward(np.ones(net.num_edges()), 2)
    ok = sum(exp) == 1 and late["deadline_misses"] == 0 and late["delivered_count"] == 1 \
        and edge["deadline_misses"] == 0 and edge["latencies"][0][0] == 3.0
    note = ("expiry flagged at step index %d (age %d > deadline 3), one step after a miss became certain; "
            "late delivery adds 1 to the denominator again but 0 to the numerator" % (exp.index(1), exp.index(1) + 1))
    return ("PASS" if ok else "FAIL"), note


@probe("Deadline-miss term noise when few packets resolve")
def _():
    env = CarlaGymEnv(max_steps=300, bandwidth_budget=0.8, load_scale=0.1, seed=1)
    obs, _ = env.reset(seed=1)
    m, cnt = [], []
    for _ in range(300):
        obs, r, _, _, i = env.step(np.ones(24, np.float32))
        m.append(i["reward_deadline_miss"]); cnt.append(i["delivered_count"])
    m = np.array(m)
    return "NOTE", (f"load x0.1: mean packets resolved/step {np.mean(cnt):.1f}; miss term mean {m.mean():.3f}, std {m.std():.3f}, "
                    f"steps at exactly 0: {np.mean(m == 0):.0%}, at exactly 1: {np.mean(m == 1):.0%}")


# ------------------------------------------------------------------ item 9: truncation with wrappers
@probe("step() after truncation: gym.make wrappers, SB3 env checker, RecordEpisodeStatistics")
def _():
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CarlaGym-v0", max_steps=10))
    env.reset(seed=0)
    n_ep = 0
    for _ in range(35):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            n_ep += 1
            assert "episode" in info
            env.reset()
    sb3_check(CarlaGymEnv(max_steps=10, seed=0), warn=False)
    raised = False
    e = gym.make("CarlaGym-v0", max_steps=2); e.reset(seed=0); e.step(e.action_space.sample()); e.step(e.action_space.sample())
    try:
        e.step(e.action_space.sample())
    except RuntimeError:
        raised = True
    return ("PASS" if n_ep == 3 and raised else "FAIL"), f"episodes completed through wrappers: {n_ep}; extra step raises: {raised}; SB3 check_env passed"


@probe("SB3 check_env on the [-1,1]-rescaled training env")
def _():
    from stable_baselines3.common.env_checker import check_env as sb3_check
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sb3_check(gym.wrappers.RescaleAction(CarlaGymEnv(max_steps=10, seed=0), np.float32(-1), np.float32(1)), warn=True)
    return "PASS", f"passed; warnings raised: {len(w)} {[str(x.message)[:70] for x in w[:2]]}"


# ------------------------------------------------------------------ item 10: MARL ownership fuzz
@probe("MARL link ownership fuzz (random topologies, 150 seed/size combos, plus all fixed topologies)")
def _():
    bad, n = [], 0
    combos = [("random", nn, s) for nn in (4, 6, 9, 12, 16) for s in range(30)]
    combos += [(t, nn, 0) for t in ("ring", "star", "mesh") for nn in (3, 5, 9, 14)] + [("grid", nn, 0) for nn in (4, 9, 16, 25)]
    for topo, nn, s in combos:
        env = MultiAgentCarlaGymEnv(topology=topo, num_nodes=nn, seed=s, max_steps=3)
        n += 1
        owned = sorted(e for v in env.agent_edges.values() for e in v)
        edges = env.network.edge_list()
        if owned != list(range(len(edges))):
            bad.append((topo, nn, s, "coverage"))
        for a, es in env.agent_edges.items():
            node = int(a.split("_")[1])
            if not es or any(node not in edges[e] for e in es):
                bad.append((topo, nn, s, "empty or non-incident"))
        obs, _ = env.reset(seed=s)
        o, r, te, tr, inf = env.step({a: env.action_space(a).sample() for a in env.agents})
        if set(o) != set(env.possible_agents) or not all(np.isfinite(list(r.values()))):
            bad.append((topo, nn, s, "step"))
    return ("PASS" if not bad else "FAIL"), f"{n} envs; problems: {bad[:4] or 'none'}"


@probe("MARL: an agent missing from the action dict")
def _():
    env = MultiAgentCarlaGymEnv(max_steps=5, seed=0)
    env.reset(seed=0)
    acts = {a: env.action_space(a).sample() for a in env.agents[:-1]}
    o, r, *_ = env.step(acts)
    return "NOTE", f"accepted silently; the absent agent's links get bandwidth 0 and routing weight 1 (reward {list(r.values())[0]:+.3f}). Not validated, not documented."


# ------------------------------------------------------------------ item 11: bottlenecks
@probe("Bottlenecks: several degraded nodes, star hub, scale 1.0 is a no-op")
def _():
    out = []
    net = NetworkModel("grid", 9, bottlenecks={0: 0.5, 1: 0.2, 4: 0.8})
    caps = dict(zip(net.edge_list(), net.get_edge_capacities()))
    ok = abs(caps[(0, 1)] - 6.0) < 1e-6 and abs(caps[(1, 4)] - 6.0) < 1e-6 and abs(caps[(0, 3)] - 15.0) < 1e-6 and abs(caps[(3, 4)] - 24.0) < 1e-6
    out.append(f"min-scale rule on shared links: {ok}")
    hub = NetworkModel("star", 9, bottlenecks={0: 0.5})
    ok2 = np.allclose(hub.get_edge_capacities(), 15.0) and hub.get_node_buffers()[0] == 60.0
    out.append(f"star hub scales all 8 links: {ok2}")

    def ret(b):
        e = CarlaGymEnv(max_steps=60, seed=5, bottlenecks=b); e.reset(seed=5)
        return [e.step(np.full(24, 0.7, np.float32))[1] for _ in range(60)]
    ok3 = ret(None) == ret({4: 1.0})
    ok4 = np.mean(ret({4: 0.2})) < np.mean(ret({4: 0.6})) < np.mean(ret(None))
    out.append(f"scale 1.0 identical to none: {ok3}; reward monotone in scale: {ok4}")
    m = MultiAgentCarlaGymEnv(topology="star", bottlenecks={0: 0.5}, max_steps=3, seed=0)
    out.append(f"MARL accepts bottlenecks: {np.allclose(m.network.get_edge_capacities(), 15.0)}")
    return ("PASS" if ok and ok2 and ok3 and ok4 else "FAIL"), "; ".join(out)


# ------------------------------------------------------------------ item 12: trace edge cases
@probe("TraceTrafficGenerator loop=False past the end; loop=True wraps")
def _():
    gen = TraceTrafficGenerator(list(range(16)), "data/cs448b_trace.csv", target_load=110.0, loop=False, seed=0)
    env = CarlaGymEnv(topology="grid", num_nodes=16, max_steps=gen.n_steps + 40, traffic=gen, seed=0)
    env.reset(seed=0)
    tail = []
    for t in range(gen.n_steps + 40):
        _, r, _, _, i = env.step(np.ones(2 * env.num_edges, np.float32))
        if t >= gen.n_steps:
            tail.append((i["offered_units"], r, i["reward_throughput"], i["delivered_units"]))
    tail = np.array(tail)
    finite = np.isfinite(tail).all()
    drained = tail[:, 3].sum()
    return ("PASS" if finite and tail[:, 0].sum() == 0 else "FAIL"), (
        f"after the trace ends: offered 0, rewards finite ({tail[:, 1].min():+.3f}..{tail[:, 1].max():+.3f}); "
        f"{drained:.0f} units were still delivered in that tail but throughput term is 0 there (offered=0), so draining earns no throughput credit")


@probe("Trace placement invariants on both committed traces")
def _():
    out = []
    for f in ("cs448b_trace.csv", "pcap_trace.csv"):
        d = pd.read_csv(f"data/{f}")
        ok = list(d.columns) == ["step", "src", "dst", "volume"] and d.src.between(0, 15).all() and d.dst.between(0, 15).all() \
            and (d.src != d.dst).all() and (d.volume > 0).all() and not d.duplicated(["step", "src", "dst"]).any() and d.notna().all().all()
        out.append(f"{f}: {len(d)} rows, steps {d.step.min()}..{d.step.max()}, schema/range/no-self-loop/no-dup/positive: {ok}")
    return ("PASS" if all("True" in o for o in out) else "FAIL"), "; ".join(out)


# ------------------------------------------------------------------ small configurations
@probe("Edge configurations: 2 nodes, large mesh, zero load, extreme load, every baseline on each")
def _():
    bad = []
    for kw in [dict(topology="ring", num_nodes=2), dict(topology="star", num_nodes=2), dict(topology="mesh", num_nodes=12),
               dict(topology="grid", num_nodes=25), dict(load_scale=0.0), dict(load_scale=8.0), dict(topology="random", num_nodes=20, seed=3)]:
        try:
            base = dict(max_steps=25, seed=0); base.update(kw)
            for name, fn in POLICIES.items():
                env = CarlaGymEnv(**base); obs, _ = env.reset(seed=1); rng = np.random.default_rng(0)
                for _ in range(25):
                    obs, r, *_ = env.step(fn(env, obs, rng))
                    assert np.isfinite(r) and env.observation_space.contains(obs)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{kw}: {type(exc).__name__}: {str(exc)[:80]}")
    return ("PASS" if not bad else "FAIL"), f"problems: {bad or 'none'}"


@probe("Invalid constructor arguments are rejected")
def _():
    cases = {"topology": dict(topology="tree"), "num_nodes=1": dict(num_nodes=1), "capacity=0": dict(link_capacity=0),
             "buffer<0": dict(node_buffer=-1), "max_steps=0": dict(max_steps=0), "render_mode": dict(render_mode="ansi"),
             "load_scale<0": dict(load_scale=-1.0), "k<0": dict(k=-1.0), "delay_reference=0": dict(delay_reference=0.0),
             "NaN action": None}
    accepted = []
    for name, kw in cases.items():
        try:
            if kw is None:
                e = CarlaGymEnv(seed=0); e.reset(seed=0); a = np.full(24, np.nan, np.float32); _, r, *_ = e.step(a)
                accepted.append(f"{name} (reward {r})")
            else:
                e = CarlaGymEnv(**kw); e.reset(seed=0); e.step(e.action_space.sample())
                accepted.append(name)
        except (ValueError, ZeroDivisionError) as exc:
            if isinstance(exc, ZeroDivisionError):
                accepted.append(f"{name} (ZeroDivisionError at step)")
        except Exception as exc:  # noqa: BLE001
            accepted.append(f"{name} ({type(exc).__name__}: {str(exc)[:50]})")
    return ("PASS" if not accepted else "NOTE"), f"not cleanly rejected: {accepted or 'none'}"


# ------------------------------------------------------------------ reported numbers vs raw files
@probe("README / report numbers recomputed from raw result files")
def _():
    out, ok = [], True
    rows = {}
    for algo in ("ppo", "sac"):
        per = []
        for d in sorted(glob.glob(f"results/grid_b08/{algo}_seed*")):
            f = pd.read_csv(os.path.join(d, "final_eval.csv"))
            per.append(f[f.policy == algo.upper()].reward.mean())
            assert sorted(f.test_seed.unique()) == list(range(2000, 2010))
        rows[algo.upper()] = (np.mean(per), ci95(per), len(per))
    f0 = pd.read_csv("results/grid_b08/ppo_seed0/final_eval.csv")
    for pol in ("queue_proportional", "congestion_aware", "load_aware_routing", "max_bandwidth", "random", "min_bandwidth"):
        # pooled over all run directories, as scripts/summarize_training.py does since round 3
        v = pd.concat(pd.read_csv(q) for q in glob.glob("results/grid_b08/*_seed*/final_eval.csv")).query("policy == @pol").groupby("test_seed").reward.mean()
        rows[pol] = (v.mean(), ci95(v), len(v))
    readme = open("README.md", encoding="utf-8").read()
    for name, (m, c, n) in rows.items():
        cell = f"| {name} |"
        line = [l for l in readme.splitlines() if l.startswith(cell)]
        want = f"{m:+.3f} ± {c:.3f}"
        hit = bool(line) and want in line[0]
        ok &= hit
        out.append(f"{name} {want}{'' if hit else ' MISMATCH'}")
    # baselines identical in every run directory?
    base = [pd.read_csv(p).query("policy=='queue_proportional'").reward.round(9).tolist() for p in glob.glob("results/grid_b08/*_seed*/final_eval.csv")]
    same_det = all(b == base[0] for b in base)
    rnd = {round(pd.read_csv(p).query("policy=='random'").reward.mean(), 4) for p in glob.glob("results/grid_b08/*_seed*/final_eval.csv")}
    # benchmark table in README equals summary.csv
    s = pd.read_csv("results/benchmark/summary.csv")
    miss = 0
    for r in s.itertuples():
        if f"{r.reward:+.3f} ± {r.reward_ci95:.3f}" not in readme:
            miss += 1
    ok &= miss == 0 and same_det
    ep = pd.read_csv("results/benchmark/episodes.csv")
    re_ci = ep.groupby(["scenario", "policy"]).reward.agg(lambda v: ci95(v))
    ci_ok = np.allclose(re_ci.values, s.set_index(["scenario", "policy"]).loc[re_ci.index].reward_ci95.values)
    return ("PASS" if ok and ci_ok else "FAIL"), ("; ".join(out) + f"; deterministic baselines identical across 10 run dirs: {same_det}; "
            f"'random' baseline differs by training seed: {sorted(rnd)}; benchmark cells missing from README: {miss} of {len(s)}; CIs recomputed from episodes: {ci_ok}")


@probe("Benchmark determinism: one scenario recomputed on current code equals results/benchmark")
def _():
    ep = pd.read_csv("results/benchmark/episodes.csv")
    diffs = []
    for name, fn in POLICIES.items():
        for seed in (2000, 2005):
            env = CarlaGymEnv(topology="ring", num_nodes=9, load_scale=1.0, bandwidth_budget=0.5, max_steps=200, seed=seed)
            obs, _ = env.reset(seed=seed); rng = np.random.default_rng(seed); r = []
            for _ in range(200):
                obs, rew, *_ = env.step(fn(env, obs, rng)); r.append(rew)
            saved = ep[(ep.scenario == "ring/load x1/budget 0.5") & (ep.policy == name) & (ep.seed == seed)].reward.iloc[0]
            diffs.append(abs(np.mean(r[20:]) - saved))
    return ("PASS" if max(diffs) < 1e-9 else "FAIL"), f"max abs difference over 12 episodes: {max(diffs):.2e}"


@probe("Saved PPO/SAC models reload and reproduce their saved test rewards")
def _():
    from stable_baselines3 import PPO, SAC
    out, ok = [], True
    for algo, cls in (("ppo", PPO), ("sac", SAC)):
        d = f"results/grid_b08/{algo}_seed1"
        model = cls.load(os.path.join(d, "model.zip"), device="cpu")
        saved = pd.read_csv(os.path.join(d, "final_eval.csv")).query(f"policy=='{algo.upper()}'").set_index("test_seed").reward
        got = []
        for s in (2000, 2001, 2002):
            env = gym.wrappers.RescaleAction(CarlaGymEnv(bandwidth_budget=0.8, max_steps=200, seed=0), np.float32(-1), np.float32(1))
            obs, _ = env.reset(seed=s); r = []
            for _ in range(200):
                obs, rew, *_ = env.step(model.predict(obs, deterministic=True)[0]); r.append(rew)
            got.append(abs(np.mean(r[20:]) - saved.loc[s]))
        ok &= max(got) < 1e-6
        out.append(f"{algo}: max abs diff {max(got):.1e}")
    return ("PASS" if ok else "FAIL"), "; ".join(out)


@probe("Documented offered load and default sizes")
def _():
    env = CarlaGymEnv(max_steps=2000, seed=0); env.reset(seed=0)
    off = np.mean([env.step(np.ones(24, np.float32))[4]["offered_units"] for _ in range(2000)])
    ok = 100 < off < 108 and env.num_edges == 12 and env.num_nodes == 9 and env.action_space.shape == (24,) and env.observation_space.shape == (33,)
    return ("PASS" if ok else "FAIL"), f"mean offered load over 2000 steps {off:.1f} (README: about 104); E=12, N=9, action (24,), obs (33,)"


@probe("Notebooks: executed, no errors, no stale references")
def _():
    out, ok = [], True
    for p in sorted(glob.glob("notebooks/*.ipynb")):
        nb = json.load(open(p, encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        errs = sum(o["output_type"] == "error" for c in code for o in c["outputs"])
        unexec = sum(c["execution_count"] is None for c in code)
        order = [c["execution_count"] for c in code if c["execution_count"]]
        text = json.dumps(nb)
        stale = [w for w in ("pcap_trace_midterm", "random_agent_test import", "three reward", "four reference", "make_vec_env") if w in text]
        ok &= errs == 0 and unexec == 0 and order == sorted(order) and not stale
        out.append(f"{os.path.basename(p)[:2]}: errors {errs}, unexecuted {unexec}, in-order {order == sorted(order)}, stale refs {stale or 'none'}")
    return ("PASS" if ok else "FAIL"), "; ".join(out)


@probe("Repo text: stale or informal wording, leftover references")
def _():
    hits = []
    pats = ["your friends", "worth defending", "Sprint", "Phase-1", "Phase 1", "Phase 2", "pcap_trace_midterm", "PLACEHOLDER", "TODO", "FIXME", "four-fold", r"Box\(0,1,\(E,\)\)"]
    files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
             if f.endswith((".py", ".md", ".toml", ".txt")) and not f.startswith(("qa/", "docs/"))]
    for f in files:
        txt = open(f, encoding="utf-8", errors="replace").read()
        for p in pats:
            n = len(re.findall(p, txt))
            if n:
                hits.append(f"{f}: '{p}' x{n}")
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout
    junk = [l for l in tracked.split() if "__pycache__" in l or l.endswith((".DS_Store", ".pyc")) or l.startswith(".idea")]
    return ("PASS" if not hits and not junk else "NOTE"), f"tracked junk files: {len(junk)}; wording hits: {hits or 'none'}"


print("\n=== SUMMARY ===")
for s in ("FAIL", "NOTE", "PASS"):
    print(s, sum(r[0] == s for r in RESULTS))
