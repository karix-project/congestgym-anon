"""Tester/QA, PR #5 round 3: R2-1, Q3 and the figures quoted in README, docs
and the PR description, checked against their source files.
    PYTHONPATH=. python qa/retest/qa_pr5_round3_audit.py   (needs gh, signed in)
"""
import re
import subprocess

import pandas as pd

readme = open("README.md", encoding="utf-8").read().replace("\r\n", "\n")
body = subprocess.run([r"C:/Program Files/GitHub CLI/gh.exe", "pr", "view", "5", "--json", "body", "--jq", ".body"],
                      capture_output=True, text=True).stdout
ok = []


def c(cond, msg):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


c('service_order="id"`.\n  It reproduces' in readme, 'R2-1 README full stop after service_order="id"')
c("0.05-0.30 (0.16-0.30 for the hop-count heuristics)" in body, "Q3 PR range corrected")
c("163 passed" in readme, "README test count 163")

q = open("results/benchmark/qos.md", encoding="utf-8").read()


def drops(prefix):
    vals = []
    for line in q.splitlines():
        if line.startswith("| " + prefix):
            cells = [x.strip() for x in line.strip("|").split("|")][1:]
            vals += [float(x.split("/")[1]) for x in cells[2:]]       # skip random, min_bandwidth
    return min(vals), max(vals)


for prefix, want in [("pcap trace/load 40/", (19, 42)), ("pcap trace/load 110", (53, 69)), ("cs448b", (55, 60))]:
    lo, hi = drops(prefix)
    c((round(lo), round(hi)) == want, f"README {prefix.strip('/')} heuristics drop {want[0]}-{want[1]}% (table {lo}-{hi})")
row = [line for line in q.splitlines() if line.startswith("| grid/load x1/budget 0.8")][0]
dl = [float(x.split("/")[0]) for x in [y.strip() for y in row.strip("|").split("|")][3:]]
c(min(dl) >= 94 and max(dl) <= 98, f"README 'about 95%' for the old headline (heuristics deliver {dl})")

probe = open("qa/research/probe_after_fixes.log", encoding="utf-8").read()
cr = open("docs/CODE_REVIEW_FOR_PUBLICATION.md", encoding="utf-8").read()
pols = ("max_bandwidth", "queue_proportional", "congestion_aware")
for pol, d, sp, w in [("max_bandwidth", "0.953", "0.028", "2.8%"), ("queue_proportional", "0.950", "0.020", "4.2%"),
                      ("congestion_aware", "0.954", "0.022", "4.2%")]:
    line = [x for x in probe.splitlines() if x.startswith(pol) and " new " in x][0]
    c(d in line and sp in line and w in line and f"/ {d} | " in cr, f"code review table row {pol} matches the probe log")
c("624.1" in probe and "82.3" in probe and "57.7" in probe and "624 steps/s" in cr, "code review speed figures match the probe log")
spreads = [float(re.search(r"/(\d\.\d+)\s+\[", x).group(1)) for x in probe.splitlines()
           if x.startswith(pols) and " new " in x]
c(len(spreads) == 3 and max(spreads) <= 0.028 + 1e-9, f"PR 'at most 0.028' spread (log {spreads})")
c(all(x < 0.04 for x in spreads), "README 'gap now under 4 points'")

s30 = pd.read_csv("qa/research/headline_settle30_summary.csv")
sel = s30[s30.all_met].sort_values(["weaker_C6_lower_bound", "load"], ascending=[False, True]).iloc[0]
hd = open("docs/HEADLINE_SCENARIO.md", encoding="utf-8").read()
c(sel.scenario == "grid9/load 1.25/budget 0.6" and abs(sel.weaker_C6_lower_bound - 0.90) < 0.005 and "**0.90**" in hd,
  f"headline selection reproduces from the 30-seed summary ({sel.scenario}, {sel.weaker_C6_lower_bound})")
c(int(s30.all_met.sum()) == 3, f"doc 'three cells met all six' ({int(s30.all_met.sum())})")
print("\nRESULT:", "ALL PASS" if all(ok) else f"{ok.count(False)} FAIL")
