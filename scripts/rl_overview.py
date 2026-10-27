"""
scripts/rl_overview.py

One table over every folder of the training matrix: delivered traffic of PPO
and SAC against the best rule-based policy, with the fluid flow bound.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python scripts/rl_overview.py

Reads results/rl/<scenario>/qos.csv (scripts/evaluate_saved.py) and
results/bounds/training.csv (scripts/flow_bound.py). Writes
results/rl/overview.md. Delivered % on test seeds 2000-2009, steps 20..199,
mean +/- 95% CI (agents over training seeds, baselines over test episodes).
The best rule-based policy is picked per row on the test seeds, which favours
the baselines. Ablation rows compare a one-half agent with baselines that
control both halves.
"""
from __future__ import annotations

import os

import pandas as pd

from congestgym.utils.stats import ci95
from scripts.run_training_matrix import ABLATIONS, ABLATIONS_E3, SCENARIOS

R = os.path.join("results", "rl")
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]


def agent(q, name):
    g = q[q.policy == name]
    if g.empty:
        return "n/a"
    per_seed = g.groupby("train_seed").delivered_fraction.mean()
    return f"{100 * per_seed.mean():.1f} ± {100 * ci95(per_seed.to_numpy()):.1f}"


def main():
    bounds = pd.read_csv(os.path.join("results", "bounds", "training.csv")).set_index("scenario")
    lines = ["| Training folder | Flags | PPO | SAC | Best rule-based policy | Flow bound |",
             "|---|---|---:|---:|---|---:|"]
    for name, flags in list(SCENARIOS.items()) + list(ABLATIONS.items()) + list(ABLATIONS_E3.items()):
        if not os.path.exists(os.path.join(R, name, "qos.csv")):
            continue                                  # E3 ablations appear once trained
        q = pd.read_csv(os.path.join(R, name, "qos.csv"))
        h = q[q.policy.isin(HEUR)].groupby("policy").delivered_fraction
        best = h.mean().idxmax()
        vals = q[q.policy == best].delivered_fraction.to_numpy()
        base = name if name in bounds.index else next(s for s in SCENARIOS if name.startswith(s))
        lines.append(f"| `{name}` | `{' '.join(flags)}` | {agent(q, 'PPO')} | {agent(q, 'SAC')} | "
                     f"{best} {100 * vals.mean():.1f} ± {100 * ci95(vals):.1f} | "
                     f"{100 * bounds.loc[base, 'bound_free']:.1f} |")
    note = ("Delivered % of offered traffic, test seeds 2000-2009, steps 20..199, mean ± 95% CI "
            "(agents: over 5 training seeds; rule-based: over test episodes). The best rule-based "
            "policy is picked per row on the test seeds. Ablation rows: PPO controls one half of the "
            "action, the rule-based policies both. Flow bound: congestgym/bounds.py, an upper limit "
            "that is loose (see docs/RL_RESULTS_0.4.0.md, section 2.6).\n\n")
    with open(os.path.join(R, "overview.md"), "w", encoding="utf-8") as f:
        f.write(note + "\n".join(lines) + "\n")
    print(note + "\n".join(lines))


if __name__ == "__main__":
    main()
