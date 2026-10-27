"""Researcher, 15 Oct 2026: experiment E1 of docs/NEXT_EXPERIMENTS.md.
Tune the rule-based policies' constants on validation seeds, then report
them once on the test seeds.

Rule (fixed in the plan before running): per scenario, the setting with the
highest mean VALIDATION reward (seeds 1000-1009) becomes the tuned policy.
  congestion_aware: floor {0, 0.05, 0.1, 0.2} x base {0, 0.1, 0.3} (slope 1 - base)
                    x signal {max, pending}   -> 24 settings
  queue_proportional: floor {0, 0.05, 0.1, 0.2} -> 4 settings
The default setting (floor 0.05, base 0.1, signal max) is in the grid.
Test seeds (2000-2009) are used only after the choice, once.

Also: PPO (existing models, results/rl/*/qos.csv) minus the tuned
congestion_aware and the tuned queue_proportional, paired on test seeds,
CI over the 5 training seeds.

    PYTHONPATH=. python qa/research/e1_tune_heuristics.py
"""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from congestgym.baselines import make_policy, policy_rng              # noqa: E402
from congestgym.utils.stats import ci95                                # noqa: E402
from examples.train_agent import parse_args                           # noqa: E402
from scripts.evaluate_saved import episode                            # noqa: E402
from scripts.run_training_matrix import SCENARIOS                     # noqa: E402

VAL, TEST = range(1000, 1010), range(2000, 2010)
CA = [dict(floor=f, base=b, slope=round(1 - b, 10), signal=s)
      for f, b, s in itertools.product([0.0, 0.05, 0.1, 0.2], [0.0, 0.1, 0.3], ["max", "pending"])]
QP = [dict(floor=f) for f in [0.0, 0.05, 0.1, 0.2]]
DEFAULT_CA = dict(floor=0.05, base=0.1, slope=0.9, signal="max")


def key(p):
    return " ".join(f"{k}={v}" for k, v in p.items())


def job(args):
    scen, pol, params, seed = args
    targs = parse_args(SCENARIOS[scen])
    fn = make_policy(pol, **params)
    rng = policy_rng(seed)
    r = episode(targs, lambda e, o: fn(e, o, rng), seed, False, 20)
    return {"scenario": scen, "policy": pol, "params": key(params), "seed": seed,
            "reward": r["reward"], "delivered": r["delivered_fraction"]}


def main():
    here = os.path.dirname(__file__)
    val_jobs = [(s, "congestion_aware", p, seed) for s in SCENARIOS for p in CA for seed in VAL]
    val_jobs += [(s, "queue_proportional", p, seed) for s in SCENARIOS for p in QP for seed in VAL]
    with Pool(10) as pool:
        val = pd.DataFrame(pool.map(job, val_jobs, chunksize=4))
    val.to_csv(os.path.join(here, "e1_validation_episodes.csv"), index=False)

    chosen = {}
    print("=" * 100)
    print("Selection on validation seeds 1000-1009 (mean reward; delivered %)")
    print("=" * 100)
    for (s, pol), g in val.groupby(["scenario", "policy"], sort=False):
        m = g.groupby("params")[["reward", "delivered"]].mean().sort_values("reward", ascending=False)
        best = m.index[0]
        default = key(DEFAULT_CA) if pol == "congestion_aware" else key(dict(floor=0.05))
        chosen[(s, pol)] = best
        print(f"{s:22s} {pol:20s} chosen [{best}] reward {m.loc[best, 'reward']:+.3f} "
              f"({100*m.loc[best, 'delivered']:.1f}%)   default [{default}] {m.loc[default, 'reward']:+.3f} "
              f"({100*m.loc[default, 'delivered']:.1f}%)   rank of default {list(m.index).index(default) + 1}/{len(m)}")

    test_jobs = []
    for (s, pol), best in chosen.items():
        params = dict(kv.split("=") for kv in best.split())
        params = {k: (v if k == "signal" else float(v)) for k, v in params.items()}
        test_jobs += [(s, pol, params, seed) for seed in TEST]
    with Pool(10) as pool:
        test = pd.DataFrame(pool.map(job, test_jobs, chunksize=2))
    test.to_csv(os.path.join(here, "e1_test_episodes.csv"), index=False)

    print("\n" + "=" * 100)
    print("Test seeds 2000-2009, once, after the choice: tuned vs untuned (untuned from results/rl/*/qos.csv)")
    print("=" * 100)
    for s in SCENARIOS:
        q = pd.read_csv(os.path.join("results", "rl", s, "qos.csv"))
        for pol in ["congestion_aware", "queue_proportional"]:
            t = test[(test.scenario == s) & (test.policy == pol)].set_index("seed")
            u = q[q.policy == pol].set_index("test_seed")
            d_r = (t.reward - u.reward).to_numpy(); d_d = 100 * (t.delivered - u.delivered_fraction).to_numpy()
            print(f"{s:22s} {pol:20s} tuned reward {t.reward.mean():+.3f} vs {u.reward.mean():+.3f} "
                  f"(diff {d_r.mean():+.3f} +/- {ci95(d_r):.3f})   delivered {100*t.delivered.mean():.1f}% vs "
                  f"{100*u.delivered_fraction.mean():.1f}% (diff {d_d.mean():+.2f} +/- {ci95(d_d):.2f} pts)")

    print("\n" + "=" * 100)
    print("PPO minus the better tuned rule-based policy, paired on test seeds, CI over 5 training seeds")
    print("=" * 100)
    for s in SCENARIOS:
        q = pd.read_csv(os.path.join("results", "rl", s, "qos.csv"))
        t = test[test.scenario == s]
        best = t.groupby("policy").reward.mean().idxmax()
        b = t[t.policy == best].set_index("seed")
        ppo = q[q.policy == "PPO"]
        for col, scale, unit in [("reward", 1, ""), ("delivered", 100, " pts")]:
            src = "reward" if col == "reward" else "delivered_fraction"
            piv = ppo.pivot(index="train_seed", columns="test_seed", values=src)
            diff = piv.sub(b[col], axis=1)
            per_seed = diff.mean(axis=1).to_numpy()
            print(f"{s:22s} vs tuned {best:20s} {col:9s} {scale*per_seed.mean():+.3f}{unit} +/- "
                  f"{scale*ci95(per_seed):.3f}   (wins {(diff.values > 0).sum()}/{diff.size})")


if __name__ == "__main__":
    main()
