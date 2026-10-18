"""Tester/QA, 18 Oct 2026: is SAC better than PPO on Abilene? (finding Q9-2)
Per-training-seed means over test seeds 2000-2009; the runs of the two
algorithms are independent, so the comparison is unpaired across seeds
(Welch-style interval on the difference of means)."""
import numpy as np, pandas as pd
from scipy import stats
q = pd.read_csv("results/rl/abilene_l120_b08/qos.csv")
for col, sc in [("reward", 1), ("delivered_fraction", 100)]:
    p = q[q.policy == "PPO"].groupby("train_seed")[col].mean().to_numpy() * sc
    s = q[q.policy == "SAC"].groupby("train_seed")[col].mean().to_numpy() * sc
    d = s.mean() - p.mean()
    se = np.sqrt(s.var(ddof=1) / len(s) + p.var(ddof=1) / len(p))
    dof = se**4 / ((s.var(ddof=1) / len(s))**2 / (len(s) - 1) + (p.var(ddof=1) / len(p))**2 / (len(p) - 1))
    h = stats.t.ppf(0.975, dof) * se
    print(f"SAC minus PPO, {col:18s} {d:+.3f} +/- {h:.3f}  (Welch, dof {dof:.1f})")
