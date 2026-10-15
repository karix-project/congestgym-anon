Fluid flow upper bound on the delivered fraction (%), from carla_gym/bounds.py: the expected traffic matrix, link capacities and the bandwidth budget, with the allocation chosen optimally. No policy can exceed it in the long run; a finite window can exceed it slightly through backlog carried in. Policy figures: delivered % on test seeds 2000-2009, steps 20..199. The bound is loose (docs/RL_RESULTS_0.4.0.md, section 2.6): the room below it is an upper limit on what any policy could gain, not a measure of it. The pcap row's bound uses the trace's mean matrix and ignores bursts.

| Scenario | Bound, free routing | Bound, shortest paths | Best heuristic | PPO | SAC |
|---|---:|---:|---|---:|---:|
| grid_l125_b06 | 90.3 | 90.3 | congestion_aware 74.1 | 73.5 | 68.6 |
| grid_bottleneck_b08 | 100.0 | 100.0 | load_aware_routing 85.0 | 94.6 | 90.3 |
| pcap40_grid16_b05 | 100.0 | 100.0 | congestion_aware 75.2 | 79.1 | 75.4 |
| random9_l15_b06 | 99.4 | 99.4 | queue_proportional 79.7 | 81.3 | 76.1 |
