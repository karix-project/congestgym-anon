"""Tester/QA review of PR #5 (0.4.0), 15 Oct 2026. Run from a fresh clone of the PR branch:
    git show origin/main:results/benchmark/summary.csv > <dir>/summary_0.3.0.csv
    PYTHONPATH=. python qa/retest/qa_pr5.py <dir>
"""
import itertools, os, subprocess, sys, warnings
import numpy as np, pandas as pd

from carla_gym.env import CarlaGymEnv
from carla_gym.baselines import POLICIES
from carla_gym.core.trace_traffic import TraceTrafficGenerator

MAIN = sys.argv[1]
fails = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond: fails.append(msg)

COMBOS = [dict(queueing=q, tie_break=t, service_order=s) for q, t, s in
          itertools.product(["per_link", "shared_fifo"], ["random", "first"], ["random", "id"])]
TOPOS = [("grid", 9), ("grid", 16), ("ring", 9), ("star", 9), ("mesh", 6), ("random", 12)]

# ---------------------------------------------------------------- A invariants
def fuzz(topo, n, opts, bottleneck, seed, steps=80):
    bn = {1: 0.4} if bottleneck else None
    env = CarlaGymEnv(topology=topo, num_nodes=n, bandwidth_budget=None, load_scale=1.5,
                      bottlenecks=bn, max_steps=steps, seed=seed, **opts)
    env.reset(seed=seed)
    net, rng = env.network, np.random.default_rng(seed)
    off = dlv = drp = 0.0
    for t in range(steps):
        a = rng.uniform(0, 1, env.action_space.shape).astype(np.float32)
        a[rng.random(a.shape) < 0.15] = 0.0                      # include zero weights / allocations
        _, r, _, _, info = env.step(a)
        off += info["offered_units"]; dlv += info["delivered_units"]; drp += info["dropped_units"]
        if abs(off - dlv - drp - net.backlog_units()) > 1e-6 * max(off, 1):
            return f"conservation broken at step {t}"
        occ, cap = net.get_node_occupancy(), net.get_node_buffers()
        if np.any(occ > cap + 1e-6):
            return f"buffer over capacity at step {t}"
        if np.any(net._served > net._edge_capacity * np.clip(a[:env.num_edges], 0, 1) + 1e-6):
            return f"link served above allocation at step {t}"
        if not np.isfinite(r):
            return "non-finite reward"
        seen = set()
        for node, pkts in net.buffers.items():
            for p in pkts:
                if id(p) in seen: return "packet in two buffers"
                seen.add(id(p))
                if p.path[p.hop_index] != node: return "packet in a buffer off its path"
                if p.next_hop is None: return "delivered packet still queued"
                if not net.graph.has_edge(node, p.next_hop): return "next hop not a neighbour"
                if len(set(p.path)) != len(p.path): return "route has a loop"
    return None

bad = []
for (topo, n), opts, bn in itertools.product(TOPOS, COMBOS, [False, True]):
    for seed in (0, 1):
        err = fuzz(topo, n, opts, bn, seed)
        if err: bad.append((topo, n, opts, bn, seed, err))
check(not bad, f"A invariants: {len(TOPOS)*len(COMBOS)*2*2} fuzz episodes (6 topologies x 8 option combos x bottleneck on/off x 2 seeds)"
      + (f" -- first failure {bad[0]}" if bad else ""))

# ---------------------------------------------------------------- B determinism
def traj(opts, seed, reset_twice=False):
    env = CarlaGymEnv(topology="grid", num_nodes=9, seed=seed, max_steps=60, **opts)
    if reset_twice:                                  # an unseeded episode first, then reseed
        env.reset(); [env.step(env.action_space.sample()) for _ in range(60)]
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(99); out = [obs.tolist()]
    for _ in range(60):
        obs, r, *_ = env.step(rng.uniform(0, 1, env.action_space.shape).astype(np.float32))
        out.append((obs.tolist(), r))
    return out
ok = all(traj(o, 5) == traj(o, 5) == traj(o, 5, reset_twice=True) for o in COMBOS)
check(ok, "B same seed gives identical trajectories, also after an unseeded episode, for all 8 option combos")
check(traj({}, 5) != traj({}, 6), "B different seeds give different trajectories")

# ---------------------------------------------------------------- C traffic independent of policy
def offered(policy, seed, trace=False):
    traffic = None
    if trace:
        traffic = TraceTrafficGenerator(list(range(16)), "data/pcap_trace.csv", target_load=40, seed=seed)
    env = CarlaGymEnv(topology="grid", num_nodes=16 if trace else 9, seed=seed, traffic=traffic,
                      bandwidth_budget=0.5, max_steps=100)
    obs, _ = env.reset(seed=seed); rng = np.random.default_rng(seed); out = []
    for _ in range(100):
        obs, _, _, _, info = env.step(POLICIES[policy](env, obs, rng)); out.append(info["offered_units"])
    return out
for trace in (False, True):
    ref = offered("max_bandwidth", 2001, trace)
    same = all(offered(p, 2001, trace) == ref for p in POLICIES)
    check(same, f"C offered traffic identical across all 6 policies on one seed ({'pcap trace' if trace else 'synthetic'})")

# ---------------------------------------------------------------- D single vs multi-agent
try:
    from carla_gym.marl_env import MultiAgentCarlaGymEnv
    from pettingzoo.test import parallel_api_test
    from gymnasium.utils.env_checker import check_env
    for opts in (dict(), dict(queueing="shared_fifo", tie_break="first", service_order="id"),
                 dict(topology="random", topology_seed=4)):
        s = CarlaGymEnv(seed=3, max_steps=40, **opts); m = MultiAgentCarlaGymEnv(seed=3, max_steps=40, **opts)
        s.reset(seed=3); m.reset(seed=3)
        rng = np.random.default_rng(3); e = s.num_edges; equal = True
        for _ in range(40):
            full = rng.uniform(0, 1, 2 * e).astype(np.float32)
            _, rs, *_ = s.step(full)
            acts = {ag: np.concatenate([full[ed], full[[e + i for i in ed]]]) for ag, ed in m.agent_edges.items()}
            _, rm, *_ = m.step(acts)
            if any(abs(r - rs) > 1e-9 for r in rm.values()): equal = False; break
        check(equal, f"D single and multi-agent rewards equal for the same global action, options {opts or 'default'}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for opts in (dict(queueing="shared_fifo", tie_break="first", service_order="id"),
                     dict(topology="random", topology_seed=4)):
            check_env(CarlaGymEnv(seed=0, **opts), skip_render_check=True)
            parallel_api_test(MultiAgentCarlaGymEnv(seed=0, **opts), num_cycles=30)
    check(True, "E Gymnasium check_env and PettingZoo parallel_api_test pass with non-default options")
except Exception as e:  # noqa: BLE001
    check(False, f"D/E raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- F benchmark cells reproduce exactly
sys.path.insert(0, ".")
from scripts.run_benchmark import scenarios, run_episode
committed = pd.read_csv("results/benchmark/episodes.csv")
want = ["grid/load x1/budget 0.5", "random/load x2/budget 0.8", "pcap trace/load 40/grid-16/budget 0.5"]
mism = 0; n = 0
for sc in [s for s in scenarios(False) if s["name"] in want]:
    for pol, fn in POLICIES.items():
        for seed in range(2000, 2010):
            r = run_episode(sc, fn, seed, 200, 20)
            row = committed[(committed.scenario == sc["name"]) & (committed.policy == pol) & (committed.seed == seed)]
            n += 1
            if len(row) != 1 or abs(row.reward.iloc[0] - r["reward"]) > 1e-12 \
                    or abs(row.delivered_fraction.iloc[0] - r["delivered_fraction"]) > 1e-12:
                mism += 1
check(mism == 0 and n == 180, f"F {n} committed benchmark episodes (3 scenarios x 6 policies x 10 seeds) reproduce exactly ({mism} mismatches)")

# ---------------------------------------------------------------- G legacy mode vs real 0.3.0
old = pd.read_csv(os.path.join(MAIN, "summary_0.3.0.csv"))
legacy = dict(queueing="shared_fifo", tie_break="first", service_order="id")
print("G legacy options vs 0.3.0 (delivered fraction, mean of test seeds 2000-2009):")
for name in ["grid/load x1/budget 0.8", "grid/load x1/budget 0.5", "ring/load x1/budget 0.8"]:
    sc = [s for s in scenarios(False) if s["name"] == name][0]
    for pol in ["max_bandwidth", "congestion_aware"]:
        vals = []
        for seed in range(2000, 2010):
            env_kw = legacy
            import scripts.run_benchmark as rb
            orig = rb.CarlaGymEnv
            rb.CarlaGymEnv = lambda **kw: orig(**kw, **env_kw)
            try:
                vals.append(run_episode(sc, POLICIES[pol], seed, 200, 20)["delivered_fraction"])
            finally:
                rb.CarlaGymEnv = orig
        o = old[(old.scenario == name) & (old.policy == pol)].delivered_fraction.iloc[0]
        print(f"   {name:28s} {pol:18s} 0.3.0 {o:.3f}  legacy-0.4.0 {np.mean(vals):.3f}  diff {np.mean(vals)-o:+.3f}")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
