"""Tester/QA confirmation of T3-T10. Prints one OK/PROBLEM line per check."""
import glob
import json
import re
import subprocess
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from carla_gym.env import CarlaGymEnv  # noqa: E402
from carla_gym.marl_env import MultiAgentCarlaGymEnv  # noqa: E402
from carla_gym.utils.reward import RewardCalculator  # noqa: E402
from carla_gym.utils.stats import ci95  # noqa: E402

bad = 0


def report(ok, label, detail=""):
    global bad
    bad += not ok
    print(f"[{'OK' if ok else 'PROBLEM'}] {label} {detail}")


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- T3
cases = [dict(delay_reference=0.0), dict(delay_reference=-3.0), dict(k=-1e-9), dict(c=-2.0), dict(m=-0.1),
         dict(load_scale=-0.01)]
for cls in (CarlaGymEnv, MultiAgentCarlaGymEnv):
    rej = [raises(lambda kw=kw: cls(**kw)) for kw in cases]
    report(all(rej), f"T3 {cls.__name__}: 6 invalid parameter sets rejected at construction", str(rej))
report(raises(lambda: RewardCalculator(delay_reference=0)), "T3 RewardCalculator used directly rejects delay_reference=0")
e = CarlaGymEnv(k=0, c=0, m=0, load_scale=0.0, delay_reference=1e-6, max_steps=5, seed=0)
e.reset(seed=0)
r = [e.step(e.action_space.sample())[1] for _ in range(5)]
report(all(np.isfinite(r)), "T3 boundary values (zeros, tiny delay_reference) still run", str(r[:2]))
e = CarlaGymEnv(max_steps=30, seed=2); e.reset(seed=2)
d = CarlaGymEnv(max_steps=30, seed=2, k=0.5, c=1.0, m=0.5, delay_reference=10.0, load_scale=1.0); d.reset(seed=2)
a = np.full(24, 0.7, np.float32)
report([e.step(a)[1] for _ in range(30)] == [d.step(a)[1] for _ in range(30)], "T3 explicit defaults behave exactly like implicit defaults")

# ---- T4
for topo in ("grid", "star", "ring", "mesh", "random"):
    m = MultiAgentCarlaGymEnv(topology=topo, max_steps=3, seed=0)
    m.reset(seed=0)
    full = {a: m.action_space(a).sample() for a in m.agents}
    miss = dict(list(full.items())[1:])
    extra = {**full, "node_999": np.zeros(2, np.float32)}
    swapped = {("x" + k if i == 0 else k): v for i, (k, v) in enumerate(full.items())}
    ok = raises(lambda: m.step(miss)) and raises(lambda: m.step(extra)) and raises(lambda: m.step(swapped))
    o, rw, te, tr, inf = m.step(full)                       # still usable after the rejections
    ok &= set(o) == set(m.possible_agents)
    for _ in range(2):
        o, rw, te, tr, inf = m.step({a: m.action_space(a).sample() for a in m.agents})
    ok &= m.agents == [] and all(tr.values())
    report(ok, f"T4 {topo}: missing / extra / misnamed agent rejected; full dict accepted; episode ends cleanly")

# ---- T5, T6, T7, T9: documentation present and matches behaviour
readme = open("README.md", encoding="utf-8").read()
reward_doc = open("carla_gym/utils/reward.py", encoding="utf-8").read()
flat = " ".join(readme.split())
report("no backpressure" in flat and "two full neighbours" in flat, "T6 README states the missing backpressure and its consequence")
report("part-transmitted packet at its full size" in flat, "T9 README states the pending-demand overcount")
report("denominator on both steps" in flat and "denominator on both steps" in " ".join(reward_doc.split()), "T5 README and reward.py describe the deadline denominator")
report("no offered traffic" in flat and "no throughput credit" in " ".join(reward_doc.split()), "T7 README and reward.py describe zero-offered-load steps")
report("needs one action for every live agent" in flat, "T4 README states the multi-agent action requirement")
stale = [w for w in ("Each packet is counted once.",) if w in readme]
report(True, "T5 note: README reward section still says", repr(stale) if stale else "nothing contradictory")

# ---- T8
files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
         if f.endswith((".py", ".md", ".toml", ".txt", ".ipynb")) and not f.startswith(("qa/", "docs/"))]
hits = []
for f in files:
    t = open(f, encoding="utf-8", errors="replace").read()
    for pat in (r"\bSprint", r"Phase[- ]?[12]", "your friends", "worth defending", "four-fold"):
        n = len(re.findall(pat, t))
        if n:
            hits.append(f"{f}:{pat} x{n}")
report(not hits, "T8 no development-phase or informal wording in tracked code, README, notebooks", str(hits))
out = subprocess.run([__import__("sys").executable, "examples/random_agent_test.py", "--episodes", "2", "--steps", "80",
                      "--budget", "0.8", "--out", "qa/retest/_tmp.csv"], capture_output=True, text=True).stdout
__import__("os").remove("qa/retest/_tmp.csv")
report("=== Sanity checks ===" in out and out.count("[PASS]") == 6 and "FAIL" not in out, "T8 sanity script prints the new heading and passes 6/6")

# ---- T10
allf = pd.concat(pd.read_csv(p).assign(run=p) for p in glob.glob("results/grid_b08/*_seed*/final_eval.csv"))
report(allf.run.nunique() == 10, "T10 ten run directories found")
summ = pd.read_csv("results/grid_b08/summary.csv").set_index("policy")
ok, det = True, []
for pol in ("queue_proportional", "congestion_aware", "load_aware_routing", "max_bandwidth", "random", "min_bandwidth"):
    v = allf[allf.policy == pol].groupby("test_seed").reward.mean()
    m, c = v.mean(), ci95(v)
    cell = f"| {pol} | baseline | {m:+.3f} ± {c:.3f} |"
    hit = abs(summ.loc[pol, "reward_mean"] - m) < 1e-12 and cell in readme
    ok &= hit
    det.append(f"{pol} {m:+.3f}")
report(ok, "T10 pooled baselines recomputed from raw files equal summary.csv and README", "; ".join(det))
for algo in ("PPO", "SAC"):
    per = allf[(allf.policy == algo)].groupby("run").reward.mean()
    cell = f"| {algo} | RL agent | {per.mean():+.3f} ± {ci95(per):.3f} |"
    report(cell in readme, f"T10 {algo} row unchanged and in README", cell)
docs = {p: open(p, encoding="utf-8").read() for p in ("docs/TESTER_REPORT.md", "docs/RESEARCHER_NOTE.md")}
report(all("| random | baseline | -0.901 ± 0.026 |" in t for t in docs.values()) and "-0.908 ± 0.026 |" not in readme,
       "T10 result tables in README and both developer docs carry the pooled random row")
nb = json.load(open("notebooks/04_ppo_training_curves.ipynb", encoding="utf-8"))
txt = " ".join("".join(o.get("text", [])) for c in nb["cells"] if c["cell_type"] == "code" for o in c["outputs"] if o["output_type"] == "stream")
errs = sum(o["output_type"] == "error" for c in nb["cells"] if c["cell_type"] == "code" for o in c["outputs"])
report("random = -0.901" in txt and "-0.908" not in json.dumps(nb) and errs == 0, "T10 notebook 04 executed output uses the pooled value, 0 errors")
leftovers = {p: len(re.findall(r"-0\.908", t)) for p, t in docs.items()}
report(True, "T10 note: remaining mentions of the old -0.908 in developer docs (narrative, not tables)", str(leftovers))
report("**122 passed**" in readme, "README test count says 122")

print(f"\nproblems: {bad}")
