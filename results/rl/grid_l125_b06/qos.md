Scenario `grid_l125_b06` (--load-scale 1.25 --budget 0.6). Episode aggregates over steps 20..199 of test seeds 2000..2009, mean ± 95% CI (agents: over training seeds; baselines: over test episodes). Delivered % and Dropped % are of traffic offered in the same window; the remainder is backlog growth. The `random` row draws its own actions here, so its reward differs slightly from the reward table. Agent and deterministic-baseline rewards match final_eval.csv (600 compared, max difference 2.2e-16).

| Policy | Reward | Delivered % | Dropped % | Delay of delivered packets (steps) | Deadline misses per delivered packet |
|---|---:|---:|---:|---:|---:|
| congestion_aware | +0.099 | 74.1 ± 1.4 | 25.8 ± 1.2 | 5.16 ± 0.09 | 0.330 ± 0.010 |
| queue_proportional | +0.093 | 73.2 ± 1.5 | 26.9 ± 1.4 | 5.03 ± 0.09 | 0.319 ± 0.007 |
| load_aware_routing | +0.045 | 72.5 ± 1.5 | 27.5 ± 1.2 | 5.42 ± 0.06 | 0.349 ± 0.005 |
| PPO | +0.037 | 73.5 ± 0.6 | 26.4 ± 0.5 | 5.82 ± 0.11 | 0.370 ± 0.006 |
| max_bandwidth | -0.034 | 68.5 ± 1.5 | 31.4 ± 1.3 | 5.38 ± 0.09 | 0.358 ± 0.010 |
| SAC | -0.059 | 68.6 ± 1.3 | 31.3 ± 1.2 | 5.82 ± 0.40 | 0.381 ± 0.023 |
| random | -0.581 | 49.4 ± 1.0 | 50.4 ± 0.8 | 7.96 ± 0.22 | 0.602 ± 0.010 |
| min_bandwidth | -1.729 | 5.5 ± 0.3 | 94.5 ± 0.3 | 52.02 ± 2.35 | 2.545 ± 0.050 |
