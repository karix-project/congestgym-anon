"""Paired (same validation seed) effects of each action half on delivered
traffic, with 95% CIs, for every scenario of a sweep that met all five
criteria of headline_sweep.py.

    PYTHONPATH=. python qa/research/headline_effects.py                 # headline sweep
    PYTHONPATH=. python qa/research/headline_effects.py second_topology
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.utils.stats import ci95  # noqa: E402

PAIRS = {"bandwidth (queue_prop - max_bw)": ("queue_proportional", "max_bandwidth"),
         "bandwidth (congestion - load_aware)": ("congestion_aware", "load_aware_routing"),
         "routing (load_aware - max_bw)": ("load_aware_routing", "max_bandwidth"),
         "routing (congestion - queue_prop)": ("congestion_aware", "queue_proportional")}


def main(prefix="headline"):
    here = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(here, f"{prefix}_sweep_episodes.csv"))
    summary = pd.read_csv(os.path.join(here, f"{prefix}_sweep_summary.csv"))
    finalists = summary[summary.n_met == 5].scenario.tolist()
    print("delivered-fraction effects in points, paired over validation seeds 1000-1009, mean +/- 95% CI")
    for sc in finalists:
        g = df[df.scenario == sc].pivot(index="seed", columns="policy", values="delivered") * 100
        print(sc)
        for label, (a, b) in PAIRS.items():
            d = (g[a] - g[b]).to_numpy()
            print(f"   {label:38s} {d.mean():+6.2f} +/- {ci95(d):.2f}")
    if not finalists:
        print("(no scenario met all five criteria)")


if __name__ == "__main__":
    main(*sys.argv[1:])
