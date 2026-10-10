"""Tester/QA confirmation of round 4 (aggregate QoS, tie marking, removed claim)."""
import io
import json
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

from carla_gym.baselines import POLICIES  # noqa: E402
from carla_gym.core.trace_traffic import TraceTrafficGenerator  # noqa: E402
from carla_gym.env import CarlaGymEnv  # noqa: E402
from carla_gym.utils.stats import ci95  # noqa: E402

bad = 0


def report(ok, label, detail=""):
    global bad
    bad += not ok
    print(f"[{'OK' if ok else 'PROBLEM'}] {label} {detail}", flush=True)


def independent_episode(env, act_fn, seed, steps=200, warmup=20):
    """Aggregates computed WITHOUT utils/qos.py: latencies are captured straight
    from NetworkModel.forward, totals from the raw info counters."""
    net = env.unwrapped.network
    captured = []
    orig = net.forward

    def spy(alloc, step):
        st = orig(alloc, step)
        captured.append(st)
        return st
    net.forward = spy
    obs, _ = env.reset(seed=seed)
    off = dlv = drp = 0.0
    delays, misses, rewards = [], 0, []
    for t in range(steps):
        obs, r, _, _, info = env.step(act_fn(env.unwrapped, obs))
        if t >= warmup:
            st = captured[-1]
            off += info["offered_units"]; dlv += info["delivered_units"]; drp += info["dropped_units"]
            delays += [d for d, _, _ in st["latencies"]]
            misses += st["deadline_misses"] + st["queue_expired"]
            rewards.append(r)
    net.forward = orig
    n = max(len(delays), 1)
    return dict(reward=np.mean(rewards), delivered_fraction=dlv / off, dropped_fraction=drp / off,
                latency_delivered=np.sum(delays) / n, deadline_miss_per_delivered=misses / n)


COLS = ["reward", "delivered_fraction", "dropped_fraction", "latency_delivered", "deadline_miss_per_delivered"]

# ---- 1. benchmark episodes: recompute several cells independently
ep = pd.read_csv("results/benchmark/episodes.csv")
cases = [("grid/load x1/budget 0.8", dict(topology="grid", num_nodes=9, load_scale=1.0, bandwidth_budget=0.8)),
         ("star/load x2/budget 0.5", dict(topology="star", num_nodes=9, load_scale=2.0, bandwidth_budget=0.5)),
         ("random/load x0.5/budget 0.8", dict(topology="random", num_nodes=9, load_scale=0.5, bandwidth_budget=0.8)),
         ("grid/bottleneck node 4 x0.5/budget 0.8", dict(topology="grid", num_nodes=9, bandwidth_budget=0.8, bottlenecks={4: 0.5})),
         ("pcap trace/grid-16/budget 0.5", dict(topology="grid", num_nodes=16, bandwidth_budget=0.5, trace="pcap_trace.csv")),
         ("cs448b trace/grid-16/budget 0.8", dict(topology="grid", num_nodes=16, bandwidth_budget=0.8, trace="cs448b_trace.csv"))]
worst = 0.0
n = 0
for name, kw in cases:
    for pol in ("queue_proportional", "max_bandwidth", "random"):
        for seed in (2001, 2007):
            kw2 = dict(kw); trace = kw2.pop("trace", None)
            traffic = TraceTrafficGenerator(list(range(16)), f"data/{trace}", target_load=110.0, seed=seed) if trace else None
            env = CarlaGymEnv(max_steps=200, seed=seed, traffic=traffic, **kw2)
            rng = np.random.default_rng(seed)
            got = independent_episode(env, lambda e, o: POLICIES[pol](e, o, rng), seed)
            row = ep[(ep.scenario == name) & (ep.policy == pol) & (ep.seed == seed)].iloc[0]
            worst = max(worst, max(abs(got[c] - row[c]) for c in COLS))
            n += 1
report(worst < 1e-9, f"1. benchmark aggregates recomputed independently for {n} episodes", f"max abs diff {worst:.1e}")

# ---- 2. the aggregate obeys accounting identities on every benchmark episode
ident = (ep.delivered_fraction >= 0).all() and (ep.dropped_fraction >= 0).all() and (ep.dropped_fraction <= 1 + 1e-9).all() \
    and (ep.latency_delivered >= 0).all()
over = (ep.delivered_fraction + ep.dropped_fraction > 1 + 1e-9).mean()
report(ident, "2. aggregate columns within valid ranges on all 2,100 episodes",
       f"(delivered+dropped exceeds 100% in {over:.1%} of episodes: possible when a warm-up backlog drains)")

# ---- 3. rewards unchanged versus the benchmark committed before round 4
prev = subprocess.run(["git", "show", "129ab50:results/benchmark/episodes.csv"], capture_output=True, text=True).stdout
old = pd.read_csv(io.StringIO(prev))
m = old.merge(ep, on=["scenario", "policy", "seed"], suffixes=("_o", "_n"))
d = max(np.abs(m[c + "_o"] - m[c + "_n"]).max() for c in ("reward", "throughput", "loss", "latency_steps", "deadline_miss"))
report(len(m) == 2100 and d == 0.0, "3. re-run benchmark equals the previously committed one", f"{len(m)} episodes, max diff {d:.1e}")

# ---- 4. saved agents: recompute qos.csv rows independently
q = pd.read_csv("results/grid_b08/qos.csv")
worst = 0.0
for algo, cls, ts in (("ppo", PPO, 3), ("sac", SAC, 1)):
    model = cls.load(f"results/grid_b08/{algo}_seed{ts}/model.zip", device="cpu")
    for seed in (2000, 2009):
        base = CarlaGymEnv(bandwidth_budget=0.8, max_steps=200, seed=0)
        env = gym.wrappers.RescaleAction(base, np.float32(-1), np.float32(1))
        got = independent_episode(env, lambda e, o: model.predict(o, deterministic=True)[0], seed)
        row = q[(q.policy == algo.upper()) & (q.train_seed == ts) & (q.test_seed == seed)].iloc[0]
        worst = max(worst, max(abs(got[c] - row[c]) for c in COLS))
report(worst < 1e-9, "4. saved-agent aggregates recomputed independently (PPO seed 3, SAC seed 1)", f"max abs diff {worst:.1e}")
fe = pd.concat(pd.read_csv(f"results/grid_b08/{a}_seed{s}/final_eval.csv").assign(algo=a.upper(), train_seed=s) for a in ("ppo", "sac") for s in range(5))
agent = fe[fe.policy == fe.algo].rename(columns={"reward": "r_final"})[["policy", "train_seed", "test_seed", "r_final"]]
mm = q[q.kind == "RL agent"].merge(agent, on=["policy", "train_seed", "test_seed"])
report(len(mm) == 100 and np.abs(mm.reward - mm.r_final).max() < 1e-9, "4b. qos.csv agent rewards equal final_eval.csv on all 100 (seed, episode) rows")

# ---- 5. qos.md and README table cells equal my own aggregation of qos.csv
readme = open("README.md", encoding="utf-8").read()
qmd = open("results/grid_b08/qos.md", encoding="utf-8").read()
ok = True
for pol, g in q.groupby("policy"):
    unit = g.groupby("train_seed")[COLS].mean() if g.kind.iloc[0] == "RL agent" else g.set_index("test_seed")[COLS]
    cell = f"{100 * unit.delivered_fraction.mean():.1f} ± {100 * ci95(unit.delivered_fraction):.1f} | {100 * unit.dropped_fraction.mean():.1f} ± {100 * ci95(unit.dropped_fraction):.1f}"
    ok &= cell in qmd and cell in readme
report(ok, "5. delivered/dropped cells for all 8 policies match in qos.md and README")
s = pd.read_csv("results/benchmark/summary.csv")
bq = open("results/benchmark/qos.md", encoding="utf-8").read()
miss = sum(f"{100 * r.delivered_fraction:.1f} / {100 * r.dropped_fraction:.1f}" not in bq for r in s.itertuples())
report(miss == 0 and bq.strip().splitlines()[-1] in readme, "5b. benchmark delivered/dropped table matches summary.csv and is in README", f"missing cells {miss} of {len(s)}")

# ---- 6. tie marking recomputed with my own rule
md = open("results/benchmark/summary.md", encoding="utf-8").read()
rows = {l.split("|")[1].strip(): l for l in md.splitlines() if l.startswith("| ") and "±" in l}
order = [c.strip() for c in [l for l in md.splitlines() if l.startswith("| Scenario")][0].split("|")[2:-1]]
wrong, n_bold, n_tie = [], 0, 0
for sc, b in s.groupby("scenario"):
    b = b.set_index("policy")
    best = b.reward.idxmax()
    lo = b.loc[best, "reward"] - b.loc[best, "reward_ci95"]
    tied = {p for p in b.index if b.loc[p, "reward"] + b.loc[p, "reward_ci95"] >= lo}
    cells = [c.strip() for c in rows[sc].split("|")[2:-1]]
    for pol, cell in zip(order, cells):
        want_bold = pol == best and len(tied) == 1
        want_tie = pol in tied and len(tied) > 1
        if (cell.startswith("**") != want_bold) or (cell.endswith("~") != want_tie):
            wrong.append((sc, pol))
    n_bold += len(tied) == 1
    n_tie += len(tied) > 1
report(not wrong and all(l in readme for l in rows.values()), "6. bold / ~ marks match my recomputation in all 35 rows, and the README carries the same rows",
       f"clear winner in {n_bold}, ties in {n_tie}; wrong marks: {wrong[:3] or 'none'}")
report("statistical tie" in md and "statistically tied" in readme, "6b. legend and explanation present")

# ---- 7. README statements checked against the data
tr = s[s.scenario.str.contains("trace") & s.policy.isin(["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"])]
lo, hi = 100 * tr.dropped_fraction.min(), 100 * tr.dropped_fraction.max()
report("54 to 69%" in readme and 54 <= lo < 55 and 69 <= hi < 70, "7. README '54 to 69% dropped on trace scenarios'", f"data: {lo:.1f} to {hi:.1f}")
ppo = q[q.policy == "PPO"].groupby("train_seed").delivered_fraction.mean().mean() * 100
report(abs(ppo - 92.2) < 0.05, "7b. PPO delivered 92.2% (tester report, round 4)", f"data: {ppo:.2f}")
stp = pd.read_csv("results/benchmark/summary.csv").query("scenario == 'cs448b trace/grid-16/budget 0.8' and policy == 'queue_proportional'").iloc[0]
report(abs(stp.reward - 0.41) < 0.01 and abs(stp.dropped_fraction - 0.57) < 0.01, "7c. cs448b: reward +0.41 with 57% dropped", f"data: {stp.reward:+.3f}, {100 * stp.dropped_fraction:.1f}%")

# ---- 8. removed claim
note = open("docs/RESEARCHER_NOTE.md", encoding="utf-8").read()
flat = " ".join(note.split())
report("NetworkGym uses a log-ratio style reward; a comparison" not in flat and "has been removed" in flat,
       "8. unsourced NetworkGym reward claim removed from RESEARCHER_NOTE, removal noted")
others = [f for f in ("README.md", "docs/TESTER_REPORT.md") if "log-ratio" in open(f, encoding="utf-8").read()]
report(not others, "8b. claim not asserted in README or tester report", str(others))

# ---- 9. sanity script and notebook 02
out = subprocess.run([sys.executable, "examples/random_agent_test.py", "--episodes", "2", "--steps", "100", "--budget", "0.8",
                      "--out", "qa/retest/_t.csv"], capture_output=True, text=True).stdout
log = pd.read_csv("qa/retest/_t.csv"); os.remove("qa/retest/_t.csv")
w = log[log.warm]; t = w.groupby("policy")[["offered_units", "delivered_units", "dropped_units"]].sum()
mine = (t.delivered_units / t.offered_units).round(3)
printed = {l.split()[0]: float(l.split()[-2]) for l in out.splitlines() if l.split() and l.split()[0] in mine.index}
report(all(abs(printed[p] - mine[p]) < 1e-3 for p in mine.index) and out.count("[PASS]") == 6, "9. sanity script delivered_frac equals totals from its own CSV; 6/6 checks pass")
nb = json.load(open("notebooks/02_policy_comparison.ipynb", encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
report("not the fraction of traffic delivered" in "".join(nb["cells"][0]["source"]) and all(c["execution_count"] for c in code)
       and not any(o["output_type"] == "error" for c in code for o in c["outputs"]), "9b. notebook 02 carries the clarification and is still fully executed")
report("**124 passed**" in readme, "README test count says 124")

print(f"\nproblems: {bad}")
