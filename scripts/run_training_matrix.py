"""
scripts/run_training_matrix.py

Queue the RL training runs behind the results under results/rl/: PPO and SAC
on four scenarios, five training seeds each, plus three PPO ablations. Runs N
jobs at a time, skips finished ones, and can be re-started safely.

Scenarios (0.4.0 simulator; chosen on validation seeds, see
docs/HEADLINE_SCENARIO.md):
    grid_l125_b06        headline: 3x3 grid, load x1.25, budget 0.6
    grid_bottleneck_b08  3x3 grid, centre node at half capacity, budget 0.8
    pcap40_grid16_b05    4x4 grid, pcap trace at load 40, budget 0.5
    random9_l15_b06      fixed random graph (topology_seed 0), load x1.5,
                         budget 0.6; bandwidth-dominated, not a replication
                         of the headline
Ablations (PPO only): bandwidth-only and routing-only on the headline, and
routing-only on random9_l15_b06.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/run_training_matrix.py --workers 10
    python scripts/run_training_matrix.py --list          # show the queue only

Step budgets: PPO 1,000,000 and SAC 200,000 per run. They are not equal on
purpose: SAC is about ten times slower per step and levels off much earlier.
Each algorithm is run until its held-out curve is flat; compare final values
and sample efficiency, not rewards at equal step counts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HEADLINE = ["--load-scale", "1.25", "--budget", "0.6"]
RANDOM9 = ["--topology", "random", "--topology-seed", "0", "--load-scale", "1.5", "--budget", "0.6"]
SCENARIOS = {
    # name: extra train_agent.py arguments
    "grid_l125_b06": HEADLINE,
    "grid_bottleneck_b08": ["--bottleneck", "4:0.5"],
    "pcap40_grid16_b05": ["--trace", "data/pcap_trace.csv", "--nodes", "16", "--load", "40", "--budget", "0.5"],
    "random9_l15_b06": RANDOM9,
}
ABLATIONS = {
    "grid_l125_b06_bandwidth_only": HEADLINE + ["--control", "bandwidth"],
    "grid_l125_b06_routing_only": HEADLINE + ["--control", "routing"],
    "random9_l15_b06_routing_only": RANDOM9 + ["--control", "routing"],
}
BUDGET = {"ppo": ("1000000", "50000"), "sac": ("200000", "20000")}   # (timesteps, eval-every)
SEEDS = range(5)


def jobs(seeds=SEEDS, budget=BUDGET, algos_all=("ppo", "sac")):
    out = []
    for seed in seeds:                       # seed-major order: every scenario gets early results
        for name, extra in list(SCENARIOS.items()) + list(ABLATIONS.items()):
            algos = ["ppo"] if name in ABLATIONS else list(algos_all)
            for algo in algos:
                steps, every = budget[algo]
                d = os.path.join("results", "rl", name, f"{algo}_seed{seed}")
                out.append((d, ["--algo", algo, "--seed", str(seed), "--timesteps", steps,
                                "--eval-every", every, "--out", d] + extra))
    return out


def run(job):
    d, args = job
    if os.path.exists(os.path.join(ROOT, d, "final_eval.csv")):
        return d, "skipped (already finished)"
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    env = {**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    t0 = time.time()
    with open(os.path.join(ROOT, d, "train.log"), "w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, os.path.join("examples", "train_agent.py")] + args,
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return d, f"{'done' if code == 0 else f'FAILED (exit {code})'} in {(time.time() - t0) / 60:.1f} min"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--list", action="store_true")
    p.add_argument("--seeds", type=int, default=5, help="training seeds per configuration")
    p.add_argument("--ppo-steps", type=int, default=int(BUDGET["ppo"][0]))
    p.add_argument("--sac-steps", type=int, default=int(BUDGET["sac"][0]))
    p.add_argument("--no-sac", action="store_true", help="train PPO only")
    a = p.parse_args()
    budget = {"ppo": (str(a.ppo_steps), str(max(a.ppo_steps // 20, 1000))),
              "sac": (str(a.sac_steps), str(max(a.sac_steps // 10, 1000)))}
    queue = jobs(range(a.seeds), budget, ("ppo",) if a.no_sac else ("ppo", "sac"))
    if a.list:
        for d, args in queue:
            print(d, " ".join(args))
        print(f"{len(queue)} runs")
        return
    print(f"{len(queue)} runs, {a.workers} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, (d, status) in enumerate(pool.map(run, queue), 1):
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(queue)} {d}: {status}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
