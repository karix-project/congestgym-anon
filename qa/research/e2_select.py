"""Researcher, 16 Oct 2026: experiment E2 of docs/NEXT_EXPERIMENTS.md, selection step.

Rule (fixed in the plan before any run): the configuration with the highest
mean held-out reward at 500k steps on VALIDATION seeds (eval_curve.csv,
seeds 1000-1004), averaged over its 3 training seeds, is chosen.

This script reads eval_curve.csv only. It does not open final_eval.csv,
which train_agent.py writes from the TEST seeds; those must stay unread
until the choice is recorded (results/rl_hp/SELECTED.txt).

    PYTHONPATH=. python qa/research/e2_select.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from congestgym.utils.stats import ci95                    # noqa: E402
from scripts.run_training_matrix import HP_SWEEP          # noqa: E402

ROOT = os.path.join("results", "rl_hp")
rows = []
for name in HP_SWEEP:
    finals, tails = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, name, "ppo_seed[0-2]", "eval_curve.csv"))):
        c = pd.read_csv(f).set_index("timesteps").reward_mean
        finals.append(c.loc[500000])
        tails.append(c.loc[500000] - c.loc[400000])
    rows.append({"config": name, "seeds": len(finals), "val_reward_500k": sum(finals) / len(finals),
                 "ci95": ci95(finals), "change_400k_500k": sum(tails) / len(tails),
                 "per_seed": " ".join(f"{v:+.3f}" for v in finals)})
t = pd.DataFrame(rows).sort_values("val_reward_500k", ascending=False).reset_index(drop=True)
assert (t.seeds == 3).all(), "every configuration needs its 3 seeds"
pd.set_option("display.width", 200)
print("Validation reward at 500k (seeds 1000-1004), mean over 3 training seeds, 95% CI over seeds")
print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
winner = t.config.iloc[0]
default = "g0.95_lr3e-4_ent0"
print(f"\nSELECTED: {winner}")
print(f"default configuration ({default}) ranks {int(t.index[t.config == default][0]) + 1} of {len(t)}; "
      f"winner minus default {t.val_reward_500k.iloc[0] - t.loc[t.config == default, 'val_reward_500k'].iloc[0]:+.3f}")
with open(os.path.join(ROOT, "SELECTED.txt"), "w", encoding="utf-8") as f:
    f.write(f"{winner}\n\nChosen by qa/research/e2_select.py from eval_curve.csv (validation seeds 1000-1004) "
            f"only, before any final_eval.csv (test seeds) of the sweep was read.\n\n" + t.to_string(index=False) + "\n")
