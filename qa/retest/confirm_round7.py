"""Tester/QA confirmation of round 7 (trace load 40, stress-case labels, reward-limit text)."""
import io
import json
import re
import subprocess
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from carla_gym.baselines import POLICIES  # noqa: E402
from carla_gym.core.trace_traffic import TraceTrafficGenerator  # noqa: E402
from carla_gym.env import CarlaGymEnv  # noqa: E402
from carla_gym.utils.stats import ci95  # noqa: E402

bad = 0


def report(ok, label, detail=""):
    global bad
    bad += not ok
    print(f"[{'OK' if ok else 'PROBLEM'}] {label} {detail}", flush=True)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8").stdout


# ---- 1. scope of the change
changed = [l for l in git("diff", "--name-only", "main", "HEAD").split() if l]
pkg = [f for f in changed if f.startswith(("carla_gym/", "tests/", "examples/"))]
report(not pkg, "1. no package, test or example code changed since main", f"changed there: {pkg or 'none'}; scripts changed: {[f for f in changed if f.startswith('scripts/')]}")

# ---- 2. earlier episodes unchanged, compared with the file on main
ep = pd.read_csv("results/benchmark/episodes.csv")
old = pd.read_csv(io.StringIO(git("show", "main:results/benchmark/episodes.csv")))
old["scenario"] = old.scenario.str.replace(r"^pcap trace/grid", "pcap trace/load 110 (overload)/grid", regex=True) \
                               .str.replace(r"^cs448b trace/grid", "cs448b trace/load 110 (stress case)/grid", regex=True)
m = old.merge(ep, on=["scenario", "policy", "seed"], suffixes=("_o", "_n"))
cols = ["reward", "throughput", "loss", "latency_steps", "deadline_miss", "delivered_fraction", "dropped_fraction", "latency_delivered"]
d = max(np.abs(m[c + "_o"] - m[c + "_n"]).max() for c in cols)
new_only = ep.merge(old[["scenario", "policy", "seed"]], on=["scenario", "policy", "seed"], how="left", indicator=True).query("_merge=='left_only'")
report(len(m) == 2100 and d == 0.0, "2. all 2,100 earlier episodes identical to main (8 columns)", f"max diff {d:.1e}")
report(len(new_only) == 120 and set(new_only.scenario) == {"pcap trace/load 40/grid-16/budget 0.8", "pcap trace/load 40/grid-16/budget 0.5"},
       "2b. exactly 120 new episodes, all pcap at load 40", f"scenarios now {ep.scenario.nunique()}, episodes {len(ep)}")

# ---- 3. new episodes recomputed independently
worst, n = 0.0, 0
for budget in (0.8, 0.5):
    for pol in ("queue_proportional", "max_bandwidth", "congestion_aware"):
        for seed in (2000, 2004, 2009):
            gen = TraceTrafficGenerator(list(range(16)), "data/pcap_trace.csv", target_load=40.0, seed=seed)
            env = CarlaGymEnv(topology="grid", num_nodes=16, bandwidth_budget=budget, max_steps=200, traffic=gen, seed=seed)
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            r, off, dlv, drp = [], 0.0, 0.0, 0.0
            for t in range(200):
                obs, rew, _, _, info = env.step(POLICIES[pol](env, obs, rng))
                if t >= 20:
                    r.append(rew); off += info["offered_units"]; dlv += info["delivered_units"]; drp += info["dropped_units"]
            row = ep[(ep.scenario == f"pcap trace/load 40/grid-16/budget {budget:g}") & (ep.policy == pol) & (ep.seed == seed)].iloc[0]
            worst = max(worst, abs(np.mean(r) - row.reward), abs(dlv / off - row.delivered_fraction), abs(drp / off - row.dropped_fraction))
            n += 1
report(worst < 1e-9, f"3. {n} load-40 episodes recomputed independently (reward, delivered, dropped)", f"max diff {worst:.1e}")
gen = TraceTrafficGenerator(list(range(16)), "data/pcap_trace.csv", target_load=40.0, seed=0)
real = np.mean([sum(p.size for p in gen.generate(t)) for t in range(4 * gen.n_steps)])
report(abs(real - 40) < 2, "3b. realised mean load of the pcap trace at target 40", f"{real:.1f}")

# ---- 4. tables and tie marks
s = pd.read_csv("results/benchmark/summary.csv")
readme = open("README.md", encoding="utf-8").read()
md = open("results/benchmark/summary.md", encoding="utf-8").read()
miss = sum(f"{r.reward:+.3f} ± {r.reward_ci95:.3f}" not in readme for r in s.itertuples())
missq = sum(f"{100 * r.delivered_fraction:.1f} / {100 * r.dropped_fraction:.1f}" not in readme for r in s.itertuples())
report(miss == 0 and missq == 0 and len(s) == 222, "4. all 222 reward cells and 222 delivered/dropped cells are in the README", f"missing {miss}, {missq}")
grp = ep.groupby(["scenario", "policy"]).reward
chk = np.allclose(grp.mean().values, s.set_index(["scenario", "policy"]).loc[grp.mean().index].reward.values) and \
    np.allclose(grp.apply(ci95).values, s.set_index(["scenario", "policy"]).loc[grp.mean().index].reward_ci95.values)
report(chk, "4b. summary means and intervals recomputed from episodes")
rows = {l.split("|")[1].strip(): l for l in md.splitlines() if l.startswith("| ") and "±" in l}
order = [c.strip() for c in [l for l in md.splitlines() if l.startswith("| Scenario")][0].split("|")[2:-1]]
wrong, nb_, nt = [], 0, 0
for sc, b in s.groupby("scenario"):
    b = b.set_index("policy"); best = b.reward.idxmax(); lo = b.loc[best, "reward"] - b.loc[best, "reward_ci95"]
    tied = {p for p in b.index if b.loc[p, "reward"] + b.loc[p, "reward_ci95"] >= lo}
    for pol, cell in zip(order, [c.strip() for c in rows[sc].split("|")[2:-1]]):
        if (cell.startswith("**") != (pol == best and len(tied) == 1)) or (cell.endswith("~") != (pol in tied and len(tied) > 1)):
            wrong.append((sc, pol))
    nb_ += len(tied) == 1; nt += len(tied) > 1
report(not wrong and (nb_, nt) == (10, 27), "4c. tie marks correct in all 37 rows", f"clear {nb_}, ties {nt}")

# ---- 5. statements in the README against data
H = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
def rng_(key):
    x = 100 * s[s.scenario.str.contains(key, regex=False) & s.policy.isin(H)].dropped_fraction
    return x.min(), x.max()
a, b, c = rng_("load 40"), rng_("overload"), rng_("stress")
flat = " ".join(readme.split())
report("drop 22 to 40%" in flat and round(a[0]) == 22 and round(a[1]) == 40, "5. 'pcap at 40: 22 to 40% dropped'", f"data {a[0]:.1f} to {a[1]:.1f}")
report("54 to 69% dropped" in flat and round(b[0]) == 54 and round(b[1]) == 69, "5b. 'pcap at 110: 54 to 69% dropped'", f"data {b[0]:.1f} to {b[1]:.1f}")
report("drops 56 to 63%" in flat and round(c[0]) == 56 and round(c[1]) == 63, "5c. 'cs448b: 56 to 63% dropped'", f"data {c[0]:.1f} to {c[1]:.1f}")
log = open("qa/research/trace_load_sweep.log", encoding="utf-8").read()
cs = log[log.index("== cs448b"):]
l20 = [l for l in cs.splitlines() if l.strip().startswith("20 |")][0]; l60 = [l for l in cs.splitlines() if l.strip().startswith("60 |")][0]
v20 = re.findall(r"[-+]?\d+\.\d+", l20)[:2]; v60 = re.findall(r"[-+]?\d+\.\d+", l60)[:2]
report("rises from +0.21 to +0.40" in flat and "falls from 52% to 45%" in flat and v20 == ["+0.213", "52.3"] and v60 == ["+0.403", "45.3"],
       "5d. reward-limit sentence matches the Researcher's sweep log", f"log: load 20 {v20}, load 60 {v60}")
report("stress case" in flat and "--load 40 --budget 0.5" in flat, "5e. cs448b labelled a stress case; README command uses load 40")

# ---- 6. notebooks
for name in ("02_policy_comparison", "03_dataset_exploration", "04_ppo_training_curves"):
    nb = json.load(open(f"notebooks/{name}.ipynb", encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    ok = all(c["execution_count"] for c in code) and not any(o["output_type"] == "error" for c in code for o in c["outputs"]) \
        and not re.search(r"(target_load|load)\s*=\s*110", src) and ("40" in src)
    report(ok, f"6. {name}: executed, 0 errors, no load-110 left in source")
nb = json.load(open("notebooks/04_ppo_training_curves.ipynb", encoding="utf-8"))
txt = "\n".join("".join(o.get("text", [])) for c in nb["cells"] if c["cell_type"] == "code" for o in c["outputs"] if o["output_type"] == "stream")
last = float(re.findall(r"timesteps=\s*40000\s+held-out reward=([-+]\d+\.\d+)", txt)[0])
base = {m[0]: float(m[1]) for m in re.findall(r"^(\w+)\s+reward=([-+]\d+\.\d+)", txt, flags=re.M)}
higher = sorted(k for k, v in base.items() if last > v)
stated = re.findall(r"Higher than: (\[.*?\])", txt)[0]
report(str(higher) == stated, "6b. notebook 04's 'Higher than' list equals what its own numbers imply", f"PPO {last:+.3f}; {stated}")

# ---- 7. leftovers
hits = []
for f in git("ls-files").split():
    if f.endswith((".py", ".md", ".toml", ".txt")) and not f.startswith(("qa/", "docs/")):
        t = open(f, encoding="utf-8", errors="replace").read()
        for mm in re.finditer(r"[^\n]*--load 110[^\n]*|[^\n]*target_load=110[^\n]*", t):
            hits.append(f"{f}: {mm.group(0).strip()[:80]}")
report(not [h for h in hits if not h.startswith("tests/")], "7. no stale 'load 110' guidance in code or README (tests excluded)", str(hits[:6]))

print(f"\nproblems: {bad}")
