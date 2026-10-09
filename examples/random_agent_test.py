"""
examples/random_agent_test.py

Sprint 4 deliverable. Runs a random-action agent and logs every step with
Pandas to verify that actions genuinely move queue lengths, latency and
reward.

It also runs three reference policies alongside random. A random agent on
its own only proves the env *runs*; comparing it against a deliberately
bad policy (min bandwidth) and a sensible one (max / queue-proportional)
is what demonstrates there is a real control problem to learn -- which is
the claim your Phase-1 evaluation actually rests on.

Usage:
    python examples/random_agent_test.py --episodes 3 --steps 150
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from carla_gym.env import CarlaGymEnv


# ---------------------------------------------------------------------- #
# Reference policies
#
# `rng` defaults to None (and is unused by every policy except the random
# one) so these can be called either as fn(env, obs, rng) -- this file's
# own run_policy() -- or as fn(env, obs) -- examples/train_ppo.py's
# rollout(), which evaluates PPO against these same baselines and doesn't
# thread an rng through. Keep that compatibility if you add a new policy.
# ---------------------------------------------------------------------- #
def policy_random(env, obs, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(0.0, 1.0, size=2 * env.num_edges)


def policy_max(env, obs, rng=None):
    return np.ones(2 * env.num_edges)


def policy_min(env, obs, rng=None):
    return np.full(2 * env.num_edges, 0.05)


def policy_queue_proportional(env, obs, rng=None):
    """Allocate bandwidth in proportion to each link's pending demand.
    Keep routing weights uniform for the baseline."""
    pending = obs[env.num_edges:2 * env.num_edges]
    if pending.max() <= 1e-9:
        bw = np.full(env.num_edges, 0.5)
    else:
        bw = np.clip(pending / pending.max(), 0.05, 1.0)
    
    routing = np.ones(env.num_edges)
    return np.concatenate([bw, routing])


POLICIES = {
    "random": policy_random,
    "min_bandwidth": policy_min,
    "max_bandwidth": policy_max,
    "queue_proportional": policy_queue_proportional,
}


def run_policy(name, fn, episodes, steps, topology, seed, budget=None, warmup=20):
    rows = []
    for ep in range(episodes):
        env = CarlaGymEnv(topology=topology, max_steps=steps,
                          bandwidth_budget=budget, seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        rng = np.random.default_rng(seed + ep)
        for t in range(steps):
            action = fn(env, obs, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append({
                "policy": name, "episode": ep, "step": t,
                "action_mean": float(np.mean(action)),
                "reward": reward,
                "throughput": info["reward_throughput"],
                "delay_norm": info["reward_delay"],
                "loss_ratio": info["reward_loss"],
                "latency_steps": info["mean_latency_steps"],
                "backlog_units": info["backlog_units"],
                "deadline_misses": info["deadline_misses"],
                "mean_link_util": info["mean_link_utilization"],
                "mean_node_occupancy": float(np.mean(obs[-env.num_nodes:])),
                "warm": t >= warmup,
            })
            if terminated or truncated:
                break
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser(description="CarlaGym Sprint 4 validation")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--topology", type=str, default="grid",
                   choices=["grid", "random", "star", "mesh", "ring"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="random_agent_log.csv")
    p.add_argument("--budget", type=float, default=None,
                   help="Bandwidth budget in (0,1]. Try 0.8 for the constrained, "
                        "non-degenerate allocation problem. Omit for unconstrained.")
    args = p.parse_args()

    rows = []
    for name, fn in POLICIES.items():
        rows += run_policy(name, fn, args.episodes, args.steps, args.topology,
                           args.seed, budget=args.budget)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    mode = ("unconstrained" if args.budget is None
            else f"bandwidth_budget={args.budget}")
    print(f"Logged {len(df)} steps across {len(POLICIES)} policies "
          f"({mode}) -> {args.out}\n")

    warm = df[df["warm"]]
    summary = warm.groupby("policy").agg(
        reward=("reward", "mean"),
        throughput=("throughput", "mean"),
        loss=("loss_ratio", "mean"),
        latency=("latency_steps", "mean"),
        backlog=("backlog_units", "mean"),
        node_occ=("mean_node_occupancy", "mean"),
    ).sort_values("reward", ascending=False)
    print("=== Policy comparison (post-warmup means) ===")
    print(summary.round(3).to_string())

    rnd = warm[warm["policy"] == "random"]
    print("\n=== Random-agent sensitivity: action_mean vs outcomes ===")
    corr = rnd[["action_mean", "reward", "backlog_units",
                "loss_ratio", "latency_steps"]].corr()["action_mean"]
    print(corr.round(3).to_string())

    print("\n=== Sprint 4 checks ===")
    checks = [
        ("node occupancy is a live observable (not constant zero)",
         warm["mean_node_occupancy"].std() > 1e-6),
        ("actions change backlog",
         abs(corr["backlog_units"]) > 0.05),
        ("actions change reward",
         abs(corr["reward"]) > 0.05),
        ("policy ranking is non-degenerate (best > worst)",
         summary["reward"].iloc[0] - summary["reward"].iloc[-1] > 0.05),
        ("loss term is informative (not pinned at 0 or 1)",
         0.001 < warm["loss_ratio"].mean() < 0.99),
        ("latency term is informative",
         warm["latency_steps"].std() > 1e-6),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


if __name__ == "__main__":
    main()
