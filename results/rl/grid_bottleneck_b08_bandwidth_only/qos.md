Scenario `grid_bottleneck_b08_bandwidth_only` (--bottleneck 4:0.5 --control bandwidth). Episode aggregates over steps 20..199 of test seeds 2000..2009, mean ± 95% CI (agents: over training seeds; baselines: over test episodes). Delivered % and Dropped % are of traffic offered in the same window; the remainder is backlog growth. The `random` row draws its own actions here, so its reward differs slightly from the reward table. Agent and deterministic-baseline rewards match final_eval.csv (300 compared, max difference 2.2e-16).

| Policy | Reward | Delivered % | Dropped % | Delay of delivered packets (steps) | Deadline misses per delivered packet |
|---|---:|---:|---:|---:|---:|
| load_aware_routing | +0.397 | 85.0 ± 0.8 | 15.0 ± 0.8 | 3.44 ± 0.08 | 0.143 ± 0.009 |
| congestion_aware | +0.368 | 84.0 ± 0.7 | 15.9 ± 0.8 | 3.62 ± 0.06 | 0.167 ± 0.007 |
| queue_proportional | +0.153 | 73.4 ± 1.1 | 26.4 ± 1.1 | 3.82 ± 0.07 | 0.214 ± 0.008 |
| PPO | +0.147 | 72.8 ± 1.1 | 27.0 ± 1.0 | 3.73 ± 0.12 | 0.209 ± 0.018 |
| max_bandwidth | -0.104 | 64.6 ± 1.4 | 35.0 ± 1.4 | 4.36 ± 0.14 | 0.301 ± 0.017 |
| random | -0.869 | 39.2 ± 0.8 | 60.8 ± 0.9 | 8.24 ± 0.22 | 0.614 ± 0.020 |
| min_bandwidth | -1.729 | 5.3 ± 0.3 | 94.7 ± 0.3 | 46.05 ± 2.04 | 2.686 ± 0.069 |
