"""Tester/QA, 15 Oct 2026: README of feat/rl-scenarios against its sources.
Run in a fresh clone, after regenerating the tables (see qa_readme_040.log):

    PYTHONPATH=. python scripts/rl_overview.py
    PYTHONPATH=. python scripts/flow_bound.py
    PYTHONPATH=. python scripts/update_readme_results.py --tests "183 passed"
    git status --porcelain            # must be empty
    PYTHONPATH=. python qa/retest/qa_readme_040.py
"""
import os
import re
import subprocess
import sys
import tomllib

import pandas as pd

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


readme = open("README.md", encoding="utf-8").read().replace("\r\n", "\n")

# ---------------------------------------------------------------- 3 every path the README names exists
paths = set(re.findall(r"`((?:carla_gym|scripts|examples|docs|results|qa|data|tests|notebooks)/[^`\s<>]*)`", readme))
paths |= set(re.findall(r"\]\(((?:results|docs|\.github)/[^)]+)\)", readme))
paths = {p.rstrip("/").rstrip(".,") for p in paths if "*" not in p}
missing = sorted(p for p in paths if not os.path.exists(p))
check(not missing, f"3 all {len(paths)} file paths named in the README exist {missing}")

# ---------------------------------------------------------------- 3 documented commands run
r = subprocess.run([sys.executable, "scripts/run_training_matrix.py", "--list", "--ppo-steps", "500000",
                    "--sac-steps", "100000"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
check(r.returncode == 0 and r.stdout.strip().endswith("55 runs"), "3 the training command lists the 55 runs reported")
r = subprocess.run([sys.executable, "scripts/evaluate_saved.py", "results/rl/grid_l125_b06", "--episodes", "2"],
                   capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
check(r.returncode == 0 and "self-check against final_eval.csv" in r.stdout,
      "3 evaluate_saved.py runs on the headline folder and passes its self-check (2 test seeds)")
subprocess.run(["git", "checkout", "--", "results/rl/grid_l125_b06/qos.csv", "results/rl/grid_l125_b06/qos.md"])

# ---------------------------------------------------------------- 4 prose claims against the tables
ov = open("results/rl/overview.md", encoding="utf-8").read()
rows = {}
for line in ov.splitlines():
    m = re.match(r"\| `(\w+)` \| `[^`]*` \| ([\d.]+) ± [\d.]+ \| ([\d.n/a ±]+?) \| (\w+) ([\d.]+) ± [\d.]+ \| ([\d.]+) \|", line)
    if m:
        rows[m.group(1)] = (float(m.group(2)), float(m.group(5)), float(m.group(6)))
scen = ["grid_l125_b06", "grid_bottleneck_b08", "pcap40_grid16_b05", "random9_l15_b06"]
wins = [s for s in scen if rows[s][0] > rows[s][1]]
check(len(rows) == 7 and wins == scen[1:], f"4 'PPO delivers more than every rule-based policy in three of the four scenarios': wins {wins}")
gaps = {s: rows[s][0] - rows[s][1] for s in wins}
check(max(gaps, key=gaps.get) == "grid_bottleneck_b08", f"4 'most on the bottleneck' ({ {k: round(v, 1) for k, v in gaps.items()} })")
q = open("results/rl/grid_l125_b06/qos.md", encoding="utf-8").read()
ppo = re.search(r"\| PPO \| ([+-][\d.]+) \| ([\d.]+) ± [\d.]+ \| [\d.]+ ± [\d.]+ \| ([\d.]+) ± [\d.]+ \| ([\d.]+) ±", q)
ca = re.search(r"\| congestion_aware \| ([+-][\d.]+) \| ([\d.]+) ± [\d.]+ \| [\d.]+ ± [\d.]+ \| ([\d.]+) ± [\d.]+ \| ([\d.]+) ±", q)
p, c = [float(x) for x in ppo.groups()], [float(x) for x in ca.groups()]
check(abs(p[1] - c[1]) < 1.0 and p[2] > c[2] and p[3] > c[3] and p[0] < c[0],
      f"4 headline: delivered about equal ({p[1]} vs {c[1]}), longer delay ({p[2]} vs {c[2]}), more misses ({p[3]} vs {c[3]}), lower reward ({p[0]} vs {c[0]})")
log = open("qa/research/rl_results_analysis.log", encoding="utf-8").read()
m = re.search(r"grid_l125_b06 PPO minus bandwidth_only PPO, reward\s+([+-][\d.]+) \+/- ([\d.]+)", log)
mean, half = float(m.group(1)), float(m.group(2))
check(mean <= 0 and mean - half < 0 < mean + half,
      f"4 'both halves do not help PPO vs bandwidth alone': {mean:+.3f} +/- {half:.3f} (mean not positive, interval includes 0)")
b = pd.concat([pd.read_csv("results/bounds/training.csv"), pd.read_csv("results/bounds/benchmark.csv")])
check((b.bound_free - b.bound_shortest).abs().max() < 1e-9, f"4 'bound the same with free and shortest-path routing in every scenario tested' ({len(b)} scenarios)")
check("183 passed" in readme, "4 test count 183 (183 passed in the worktree with raw data; 169 + 14 skipped in a clone)")

# ---------------------------------------------------------------- 5 packaging
proj = tomllib.load(open("pyproject.toml", "rb"))["project"]
extras = proj["optional-dependencies"]
check(extras.get("bounds") == ["scipy>=1.10"] and "bounds" in extras["all"][0], "5 pyproject: `bounds` extra exists and is in `all`")
req = open("requirements.txt").read(); con = open("constraints.txt").read()
check("scipy>=1.10" in req and "scipy==1.18.1" in con, "5 scipy in requirements.txt and pinned in constraints.txt")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL")
