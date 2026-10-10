"""
scripts/summarize_training.py

Aggregate the per-seed outputs of examples/train_agent.py into one table and
one learning-curve figure.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/summarize_training.py results/grid_b08

Expects <dir>/<algo>_seed<N>/{eval_curve.csv, final_eval.csv}. Writes:
    summary.csv / summary.md   test-seed reward, mean and 95% CI
    learning_curves.png        held-out reward vs. timesteps, mean and 95% CI band

How the intervals are formed:
  * RL agents: one number per TRAINING seed (that seed's mean over the test
    episodes); the CI is over training seeds, so it reflects run-to-run
    variation of training itself.
  * Baselines: deterministic given the traffic, so the CI is over the test
    episodes (they are identical in every run directory).
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from carla_gym.utils.stats import ci95  # noqa: E402


def load(root: str):
    finals, curves = [], []
    for d in sorted(glob.glob(os.path.join(root, "*_seed*"))):
        m = re.fullmatch(r"(\w+?)_seed(\d+)", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "final_eval.csv")):
            continue
        algo, seed = m.group(1).upper(), int(m.group(2))
        finals.append(pd.read_csv(os.path.join(d, "final_eval.csv")).assign(algo=algo, train_seed=seed))
        curves.append(pd.read_csv(os.path.join(d, "eval_curve.csv")).assign(algo=algo, train_seed=seed))
    if not finals:
        raise SystemExit(f"No finished runs found under {root}")
    return pd.concat(finals), pd.concat(curves)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root")
    a = p.parse_args()
    finals, curves = load(a.root)
    algos = sorted(finals["algo"].unique())

    rows = []
    for algo in algos:
        per_seed = (finals[(finals.algo == algo) & (finals.policy == algo)]
                    .groupby("train_seed")["reward"].mean())
        rows.append({"policy": algo, "kind": "RL agent", "reward_mean": per_seed.mean(),
                     "reward_ci95": ci95(per_seed), "n": len(per_seed),
                     "ci_over": "training seeds",
                     "per_seed": " ".join(f"{v:+.3f}" for v in per_seed)})
    one_run = finals[(finals.algo == algos[0])
                     & (finals.train_seed == finals[finals.algo == algos[0]].train_seed.min())]
    for name, g in one_run[~one_run.policy.isin(algos)].groupby("policy"):
        rows.append({"policy": name, "kind": "baseline", "reward_mean": g["reward"].mean(),
                     "reward_ci95": ci95(g["reward"]), "n": len(g),
                     "ci_over": "test episodes", "per_seed": ""})
    summary = pd.DataFrame(rows).sort_values("reward_mean", ascending=False)
    summary.to_csv(os.path.join(a.root, "summary.csv"), index=False)

    lines = ["| Policy | Kind | Reward (mean ± 95% CI) | n | CI over |", "|---|---|---:|---:|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.policy} | {r.kind} | {r.reward_mean:+.3f} ± {r.reward_ci95:.3f} "
                     f"| {r.n} | {r.ci_over} |")
    with open(os.path.join(a.root, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(9, 5))
    for algo, color in zip(algos, ["crimson", "darkorange", "purple"]):
        g = curves[curves.algo == algo].groupby("timesteps")["reward_mean"]
        mean, band = g.mean(), g.apply(ci95)
        ax.plot(mean.index, mean.values, marker="o", ms=3, color=color,
                label=f"{algo} (mean of {curves[curves.algo == algo].train_seed.nunique()} seeds)")
        ax.fill_between(mean.index, mean - band, mean + band, color=color, alpha=0.2)
    styles = {"queue_proportional": "seagreen", "congestion_aware": "teal",
              "load_aware_routing": "olive", "max_bandwidth": "steelblue", "random": "gray"}
    for r in summary[summary.kind == "baseline"].itertuples():
        if r.policy in styles:
            ax.axhline(r.reward_mean, ls="--", color=styles[r.policy], alpha=0.8, label=r.policy)
    ax.set_xscale("log")
    ax.set_xlabel("training timesteps (log scale)")
    ax.set_ylabel("held-out reward (validation seeds; baselines on test seeds)")
    ax.set_title(f"Learning curves, shaded band = 95% CI over training seeds ({os.path.basename(os.path.normpath(a.root))})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.root, "learning_curves.png"), dpi=150)
    print(f"\nWrote summary and learning_curves.png to {a.root}")


if __name__ == "__main__":
    main()
