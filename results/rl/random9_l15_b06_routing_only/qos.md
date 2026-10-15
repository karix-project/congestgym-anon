Scenario `random9_l15_b06_routing_only` (--topology random --topology-seed 0 --load-scale 1.5 --budget 0.6 --control routing). Episode aggregates over steps 20..199 of test seeds 2000..2009, mean ± 95% CI (agents: over training seeds; baselines: over test episodes). Delivered % and Dropped % are of traffic offered in the same window; the remainder is backlog growth. The `random` row draws its own actions here, so its reward differs slightly from the reward table. Agent and deterministic-baseline rewards match final_eval.csv (300 compared, max difference 2.2e-16).

| Policy | Reward | Delivered % | Dropped % | Delay of delivered packets (steps) | Deadline misses per delivered packet |
|---|---:|---:|---:|---:|---:|
| queue_proportional | +0.329 | 79.7 ± 0.7 | 20.2 ± 0.7 | 3.47 ± 0.03 | 0.158 ± 0.006 |
| congestion_aware | +0.221 | 76.1 ± 0.7 | 23.9 ± 0.6 | 3.92 ± 0.04 | 0.208 ± 0.007 |
| PPO | -0.060 | 66.3 ± 0.2 | 33.4 ± 0.2 | 4.59 ± 0.11 | 0.298 ± 0.011 |
| load_aware_routing | -0.130 | 64.8 ± 1.0 | 35.1 ± 1.1 | 5.22 ± 0.15 | 0.342 ± 0.014 |
| max_bandwidth | -0.175 | 63.3 ± 0.8 | 36.4 ± 0.8 | 5.47 ± 0.11 | 0.374 ± 0.012 |
| random | -0.656 | 45.3 ± 1.0 | 54.6 ± 1.1 | 7.43 ± 0.12 | 0.577 ± 0.014 |
| min_bandwidth | -1.679 | 6.4 ± 0.2 | 93.5 ± 0.2 | 40.48 ± 1.18 | 1.824 ± 0.062 |
