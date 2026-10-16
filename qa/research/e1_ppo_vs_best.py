"""Researcher, 15 Oct 2026: E1 follow-up. PPO against the strongest rule-based
policy available after E1, chosen per scenario and per metric on the test
seeds from: the four untuned heuristics (results/rl/*/qos.csv) and tuned
congestion_aware / tuned queue_proportional (qa/research/e1_test_episodes.csv).
Choosing the comparator on the test seeds favours the rule-based side.

    PYTHONPATH=. python qa/research/e1_ppo_vs_best.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.utils.stats import ci95                     # noqa: E402
from scripts.run_training_matrix import SCENARIOS          # noqa: E402

HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
tuned = pd.read_csv(os.path.join(os.path.dirname(__file__), "e1_test_episodes.csv"))
print("PPO minus the strongest rule-based policy (untuned or tuned), per metric; CI over 5 training seeds")
for s in SCENARIOS:
    q = pd.read_csv(os.path.join("results", "rl", s, "qos.csv"))
    cands = {p: q[q.policy == p].set_index("test_seed")[["reward", "delivered_fraction"]] for p in HEUR}
    for p in ["congestion_aware", "queue_proportional"]:
        t = tuned[(tuned.scenario == s) & (tuned.policy == p)].set_index("seed")
        cands["tuned " + p] = t.rename(columns={"delivered": "delivered_fraction"})[["reward", "delivered_fraction"]]
    ppo = q[q.policy == "PPO"]
    for col, scale, unit in [("reward", 1, ""), ("delivered_fraction", 100, " pts")]:
        best = max(cands, key=lambda k: cands[k][col].mean())
        piv = ppo.pivot(index="train_seed", columns="test_seed", values=col)
        diff = piv.sub(cands[best][col], axis=1)
        per_seed = diff.mean(axis=1).to_numpy()
        print(f"   {s:22s} {col:18s} vs {best:30s} ({scale*cands[best][col].mean():.3f}{unit.strip()}) "
              f"PPO {scale*per_seed.mean():+.3f}{unit} +/- {scale*ci95(per_seed):.3f}  (wins {(diff.values > 0).sum()}/{diff.size})")
