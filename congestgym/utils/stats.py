"""congestgym/utils/stats.py -- small statistics helpers for reporting results."""

from __future__ import annotations

import numpy as np

# two-sided 95% Student-t critical values, index = degrees of freedom
_T95 = [float("nan"), 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
        2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def ci95(values) -> float:
    """Half-width of the 95% Student-t confidence interval of the mean.
    NaN for fewer than two values."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float("nan")
    df = len(v) - 1
    t = _T95[df] if df < len(_T95) else 1.96
    return float(t * v.std(ddof=1) / np.sqrt(len(v)))
