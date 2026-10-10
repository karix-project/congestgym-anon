"""
scripts/evaluate_saved.py

Re-evaluate the saved agents of a results directory, and every baseline, on
the test seeds, and report episode-aggregate QoS next to the reward: the
fraction of offered traffic delivered and dropped, the mean delay of delivered
packets, and deadline misses per delivered packet.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/evaluate_saved.py results/grid_b08

Expects <dir>/<algo>_seed<N>/model.zip from examples/train_agent.py run with
its default scenario options (pass the same --topology/--nodes/--budget/
--load-scale here if they differed). Trains nothing. Writes qos.csv and qos.md.

The per-step reward terms `throughput` and `loss` are ratios within one step;
their episode mean is not the fraction of traffic delivered. This table is.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3

from carla_gym.baselines import POLICIES
from carla_gym.env import CarlaGymEnv
from carla_gym.utils.qos import EpisodeQoS
from carla_gym.utils.stats import ci95

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}
COLS = ["reward", "delivered_fraction", "dropped_fraction", "latency_delivered",
        "deadline_miss_per_delivered"]


def episode(a, act_fn, seed, rescale):
    base = CarlaGymEnv(topology=a.topology, num_nodes=a.nodes, max_steps=a.steps,
                       bandwidth_budget=a.budget, load_scale=a.load_scale, seed=0)
    env = gym.wrappers.RescaleAction(base, np.float32(-1.0), np.float32(1.0)) if rescale else base
    obs, _ = env.reset(seed=seed)
    rewards, qos = [], EpisodeQoS()
    for t in range(a.steps):
        obs, r, _, _, info = env.step(act_fn(base, obs))
        if t >= a.warmup:
            rewards.append(r)
            qos.add(info)
    return {"reward": float(np.mean(rewards)), **qos.summary()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root")
    p.add_argument("--topology", default="grid")
    p.add_argument("--nodes", type=int, default=9)
    p.add_argument("--budget", type=float, default=0.8)
    p.add_argument("--load-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed0", type=int, default=2000)
    p.add_argument("--episodes", type=int, default=10)
    a = p.parse_args()
    seeds = range(a.seed0, a.seed0 + a.episodes)

    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "*_seed*"))):
        m = re.fullmatch(r"(\w+?)_seed(\d+)", os.path.basename(d))
        path = os.path.join(d, "model.zip")
        if not m or m.group(1) not in ALGOS or not os.path.exists(path):
            continue
        model = ALGOS[m.group(1)].load(path, device="cpu")
        fn = lambda e, o, model=model: model.predict(o, deterministic=True)[0]
        for s in seeds:
            rows.append({"policy": m.group(1).upper(), "kind": "RL agent",
                         "train_seed": int(m.group(2)), "test_seed": s,
                         **episode(a, fn, s, rescale=True)})
    for name, fn in POLICIES.items():
        for s in seeds:
            rng = np.random.default_rng(s)
            rows.append({"policy": name, "kind": "baseline", "train_seed": -1, "test_seed": s,
                         **episode(a, lambda e, o, fn=fn, rng=rng: fn(e, o, rng), s, rescale=False)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.root, "qos.csv"), index=False)

    out = []
    for (policy, kind), g in df.groupby(["policy", "kind"]):
        # agents: one value per training seed; baselines: one per test episode
        unit = g.groupby("train_seed")[COLS].mean() if kind == "RL agent" else g.set_index("test_seed")[COLS]
        row = {"policy": policy, "kind": kind, "n": len(unit)}
        for c in COLS:
            row[c], row[c + "_ci95"] = unit[c].mean(), ci95(unit[c])
        out.append(row)
    table = pd.DataFrame(out).sort_values("reward", ascending=False)

    def cell(r, c, scale=1.0, fmt="{:.1f}"):
        return f"{fmt.format(scale * r[c])} ± {fmt.format(scale * r[c + '_ci95'])}"

    lines = ["| Policy | Reward | Delivered % | Dropped % | Delay of delivered packets (steps) | Deadline misses per delivered packet |",
             "|---|---:|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['policy']} | {r['reward']:+.3f} | {cell(r, 'delivered_fraction', 100)} | "
                     f"{cell(r, 'dropped_fraction', 100)} | {cell(r, 'latency_delivered', fmt='{:.2f}')} | "
                     f"{cell(r, 'deadline_miss_per_delivered', fmt='{:.3f}')} |")
    note = (f"Episode aggregates over steps {a.warmup}..{a.steps - 1} of test seeds "
            f"{a.seed0}..{a.seed0 + a.episodes - 1}, mean ± 95% CI (agents: over training seeds; "
            f"baselines: over test episodes). Delivered % and Dropped % are of traffic offered "
            f"in the same window; the remainder is backlog growth. The `random` row draws its "
            f"own actions here, so its reward differs slightly from the reward table.\n\n")
    with open(os.path.join(a.root, "qos.md"), "w", encoding="utf-8") as f:
        f.write(note + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
