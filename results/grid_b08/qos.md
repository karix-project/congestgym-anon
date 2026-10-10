Episode aggregates over steps 20..199 of test seeds 2000..2009, mean ± 95% CI (agents: over training seeds; baselines: over test episodes). Delivered % and Dropped % are of traffic offered in the same window; the remainder is backlog growth.

| Policy | Reward | Delivered % | Dropped % | Delay of delivered packets (steps) | Deadline misses per delivered packet |
|---|---:|---:|---:|---:|---:|
| PPO | +0.494 | 92.2 ± 0.5 | 7.7 ± 0.5 | 3.89 ± 0.07 | 0.192 ± 0.008 |
| queue_proportional | +0.404 | 87.0 ± 1.4 | 13.0 ± 1.4 | 3.88 ± 0.06 | 0.194 ± 0.005 |
| congestion_aware | +0.386 | 86.6 ± 1.0 | 13.4 ± 1.0 | 4.00 ± 0.07 | 0.211 ± 0.009 |
| SAC | +0.383 | 87.6 ± 0.9 | 12.3 ± 0.9 | 4.30 ± 0.22 | 0.237 ± 0.021 |
| load_aware_routing | +0.378 | 86.2 ± 1.2 | 13.9 ± 1.3 | 4.00 ± 0.12 | 0.212 ± 0.014 |
| max_bandwidth | +0.317 | 82.6 ± 1.5 | 17.3 ± 1.5 | 4.02 ± 0.09 | 0.208 ± 0.011 |
| random | -0.898 | 39.4 ± 1.5 | 60.3 ± 1.7 | 12.01 ± 0.32 | 0.867 ± 0.026 |
| min_bandwidth | -1.803 | 3.6 ± 0.2 | 96.4 ± 0.2 | 81.59 ± 1.37 | 4.194 ± 0.240 |
