"""Tester/QA, 18 Oct 2026: review of PR #9 (SNDlib Abilene data and scenario).
Run in a fresh clone of the PR branch:
    PYTHONPATH=. python qa/retest/qa_pr9.py [path to the SNDlib archive, optional]
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from carla_gym.utils.stats import ci95

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


R = "results/rl/abilene_l120_b08"
q = pd.read_csv(f"{R}/qos.csv", float_precision="round_trip")

# ---------------------------------------------------------------- 1 the bug is gone from every result
check((q.latency_delivered > 1.0).all(), f"1 every Abilene episode has positive delay (min {q.latency_delivered.min():.2f} steps)")
check((q.deadline_miss_per_delivered > 0).mean() > 0.9, "1 deadline misses are counted (non-zero in >90% of episodes)")
bench = pd.read_csv("results/benchmark/episodes.csv")
ab = bench[bench.scenario.str.startswith("abilene")]
check(len(ab) == 4 * 6 * 10 and (ab[~ab.policy.isin(["min_bandwidth"])].latency_delivered > 1.0).all(),
      f"1 benchmark Abilene rows: 240 episodes, positive delays")
check(not os.path.exists("results/rl/abilene_l120_b07") and "abilene_l120_b07" not in git("ls-files"),
      "1 the discarded buggy run (abilene_l120_b07) is not in the repository")
old = git("show", "c3cd097:carla_gym/core/trace_traffic.py")
check("prof[\"deadline\"], step))" in old and "shifted = step + self.offset" in open("carla_gym/core/trace_traffic.py").read(),
      "1 the buggy line existed in c3cd097 and the fix is in place")

# regression tests must fail against the buggy generator
import shutil, tempfile
tmp = tempfile.mkdtemp()
shutil.copytree(".", os.path.join(tmp, "repo"), ignore=shutil.ignore_patterns(".git", "results", "notebooks", "qa"))
open(os.path.join(tmp, "repo", "carla_gym", "core", "trace_traffic.py"), "w").write(old)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_abilene.py", "-k", "stamps or positive"],
                   cwd=os.path.join(tmp, "repo"), capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
check("2 failed" in r.stdout, f"1 the two regression tests fail on the buggy generator ({r.stdout.strip().splitlines()[-1]})")
shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- 2 selection integrity
t_sel = int(git("log", "-1", "--format=%ct", "4accaca"))
t_res = int(git("log", "-1", "--format=%ct", "--", f"{R}/qos.csv"))
check(t_sel < t_res, "2 the corrected selection (4accaca) was committed before the Abilene results")
s = pd.read_csv("qa/research/abilene_select_summary.csv")
pl = open("qa/research/abilene_select_paired.log", encoding="utf-8").read()
check("SELECTED: load 120, budget 0.8" in pl, "2 the paired selection log selects load 120, budget 0.8")
ep = pd.read_csv("qa/research/abilene_select_episodes.csv")
cand = s[s.criteria_met.str.startswith("1234")]
ok = []
for r_ in cand.itertuples():
    g = ep[(ep.load == r_.load) & (ep.budget == r_.budget)].pivot(index="seed", columns="policy", values="reward")
    o = g.mean().sort_values(ascending=False).index
    d = (g[o[0]] - g[o[1]]).to_numpy()
    if ci95(d) <= 0.03:
        ok.append((abs(r_._6 - 76.5), r_.load, r_.budget))
check(sorted(ok)[0][1:] == (120, 0.8) and len(ok) == 2, f"2 rule recomputed from raw episodes: qualifying {sorted((l, b) for _, l, b in ok)}, pick (120, 0.8)")
check(set(ep.seed) == set(range(1000, 1030)), "2 selection used validation seeds 1000-1029 only")
check(all(os.path.exists(f"qa/research/{f}") for f in ["abilene_select_buggy_reward.log", "abilene_select_paired_buggy_reward.log"]),
      "2 the buggy selection's logs are kept for the record")
flags = open("scripts/run_training_matrix.py").read()
check('"--load", "120", "--budget", "0.8", "--trace-random-start"]' in flags and '"abilene_l120_b08": ABILENE' in flags,
      "2 the training queue uses the selected cell")

# ---------------------------------------------------------------- 3 data
check(not any(re.search(r"(pcap|cs448b)_trace\.csv$", f) for f in git("ls-files", "data").splitlines()),
      "3 no pcap or cs448b trace in the tree")
leftover = [f for f in git("ls-files").splitlines() if f.endswith((".py", ".ipynb", ".md", ".toml"))
            and not f.startswith(("qa/", "docs/", "results/")) and re.search(r"pcap_trace|cs448b_trace", open(f, encoding="utf-8", errors="replace").read())]
check(not leftover, f"3 no code, notebook or README references the removed traces {leftover}")
arch = sys.argv[1] if len(sys.argv) > 1 else ""
if os.path.exists(arch):
    out = os.path.join(tempfile.mkdtemp(), "t.csv.gz")
    subprocess.run([sys.executable, "scripts/prepare_sndlib_trace.py", "--tgz", arch, "--start", "20040301", "--days", "7", "--out", out],
                   check=True, capture_output=True, env={**os.environ, "PYTHONPATH": "."})
    check(pd.read_csv(out).equals(pd.read_csv("data/abilene_trace_20040301_7d.csv.gz")), "3 the committed trace equals the script's output from SNDlib's archive")
else:
    print("SKIP 3 trace reproduction: no SNDlib archive given")
doc = open("data/ABILENE_TRACE.md", encoding="utf-8").read()
check(all(k in doc for k in ["Idzikowski", "Koster", "Orlowski", "no licence"]), "3 data/ABILENE_TRACE.md carries SNDlib's citations and the licence caveat")

# ---------------------------------------------------------------- 4 results recomputed from final_eval.csv
def final(agent):
    rows = []
    for d in sorted(glob.glob(f"{R}/{agent.lower()}_seed*")):
        f = pd.read_csv(f"{d}/final_eval.csv")
        rows.append(f[f.policy == agent].set_index("test_seed").reward.rename(d[-1]))
    return pd.concat(rows, axis=1).T
f0 = pd.read_csv(sorted(glob.glob(f"{R}/ppo_seed*/final_eval.csv"))[0])
ca = f0[f0.policy == "congestion_aware"].set_index("test_seed").reward
for agent, want in [("PPO", ("-0.133", 3)), ("SAC", ("-0.037", 16))]:
    d = final(agent).sub(ca, axis=1)
    per = d.mean(axis=1)
    check(f"{per.mean():+.3f}" == want[0] and int((d.values > 0).sum()) == want[1],
          f"4 {agent} minus congestion_aware reward from final_eval.csv: {per.mean():+.3f} (wins {(d.values > 0).sum()}/50); doc {want[0]} ({want[1]}/50)")
log = open("qa/research/abilene_results.log", encoding="utf-8").read()
adoc = open("docs/ABILENE_RESULTS.md", encoding="utf-8").read()
for frag_log, frag_doc in [("-5.331 pts +/- 3.318", "-5.3 +/- 3.3 pts"), ("-2.269 pts +/- 2.300", "-2.3 +/- 2.3 pts"),
                           ("change 400k->500k +0.117 +/- 0.078", "+0.117 +/- 0.078"), ("on a detour 7.57% +/- 1.52", "7.6 +/- 1.5%"),
                           ("on a detour 15.15% +/- 3.42", "15.2 +/- 3.4%"), ("on a detour 2.89% +/- 1.40", "2.9 +/- 1.4%"),
                           ("+5.934 pts +/- 1.564", "+5.9 +/- 1.6 points")]:
    check(frag_log in log and frag_doc in adoc, f"4 ABILENE_RESULTS '{frag_doc}' backed by the log")

# ---------------------------------------------------------------- 5 benchmark episodes reproduce; README; notebooks
sys.path.insert(0, ".")
from carla_gym.baselines import POLICIES
from scripts.run_benchmark import run_episode, scenarios
sc = [x for x in scenarios(False) if x["name"] == "abilene trace/load 120/budget 0.8"][0]
mis = 0
for pol in ("congestion_aware", "max_bandwidth"):
    for seed in (2000, 2003):
        r_ = run_episode(sc, POLICIES[pol], seed, 200, 20)
        row = bench[(bench.scenario == sc["name"]) & (bench.policy == pol) & (bench.seed == seed)]
        mis += abs(r_["reward"] - row.reward.iloc[0]) > 1e-12
check(mis == 0, "5 four committed Abilene benchmark episodes reproduce (rel. 1e-12)")
readme = open("README.md", encoding="utf-8").read().replace("\r\n", "\n")
check("PPO delivers more than every rule-based policy in two of the four scenarios" in readme and "210 passed" in readme,
      "5 README states 2 of 4 and the test count")
qmd = open("results/benchmark/qos.md", encoding="utf-8").read()
def drops(prefix):
    v = []
    for line in qmd.splitlines():
        if line.startswith("| " + prefix):
            v += [float(x.split("/")[1]) for x in [y.strip() for y in line.strip("|").split("|")][3:]]
    return round(min(v)), round(max(v))
for prefix, want in [("abilene trace/load 80/budget 0.8", (7, 25)), ("abilene trace/load 80/budget 0.5", (14, 40)),
                     ("abilene trace/load 120/budget 0.8", (20, 39)), ("abilene trace/load 120/budget 0.5", (35, 56))]:
    check(drops(prefix) == want, f"5 README drop range for {prefix}: table {drops(prefix)}, README {want}")
import json
for nbp in ["notebooks/02_policy_comparison.ipynb", "notebooks/03_dataset_exploration.ipynb", "notebooks/04_ppo_training_curves.ipynb"]:
    nb = json.load(open(nbp, encoding="utf-8"))
    errs = sum(o.get("output_type") == "error" for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))
    unexec = sum(c["cell_type"] == "code" and c.get("execution_count") is None for c in nb["cells"])
    check(errs == 0 and unexec == 0, f"5 {nbp}: executed, no errors")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL")
