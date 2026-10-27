# Changelog

## 0.5.0

* Repository cleanup: removed the 0.3.0 training runs (`results/grid_b08/`),
  the withdrawn pcap scenario (`results/rl/pcap40_grid16_b05/`), the raw-dataset
  adapters for the withdrawn traces (`congestgym/core/dataset_adapters.py`,
  `scripts/prepare_trace.py`), `examples/train_ppo.py` and
  `examples/random_agent_test.py`, run logs, early QA rounds (`qa/baseline`,
  `qa/after`, `qa/repro`, `qa/retest`) and process documents (handoffs,
  readiness reviews, QA reports). Everything removed is still in the git
  history, e.g. at tag v0.4.0.
* `examples/demo.py` now defaults to the degraded-node grid with the PPO agent
  trained on it against the best rule there (`--bottleneck none` for the plain
  grid).
* Added `AGENTS.md` (guide for contributors and coding agents) and this file.
* Renamed the project from CarlaGym to **CongestGym** to avoid confusion with
  the CARLA driving simulator: package `carla_gym` -> `congestgym`, classes
  `CarlaGymEnv` -> `CongestGymEnv` and `MultiAgentCarlaGymEnv` ->
  `MultiAgentCongestGymEnv`, Gymnasium id `CarlaGym-v0` -> `CongestGym-v0`,
  PettingZoo name `congestgym_marl_v0`, distribution name `congestgym`.
  Saved models and committed results are unaffected.

## 0.4.0

Dynamics changed, so results from 0.3.0 are not comparable. The reasons and
measurements are in `docs/CODE_REVIEW_FOR_PUBLICATION.md` at tag v0.4.0 (F1 to F3).

* Equal-cost paths are chosen at random, and nodes are served in a random
  order each step. In 0.3.0 the four corners of the 3x3 grid delivered 72%
  to 96% of their traffic under hop-count routing; the gap is now under 4
  points.
* One FIFO per outgoing link replaces one FIFO per node. In 0.3.0, 16% to 20%
  of allocated link bandwidth went unused while traffic waited for it.
* The `random` topology comes from `topology_seed`, not the episode seed.
  In 0.3.0 its edge count, and so the action space, changed with the seed.
* New options on both envs and `NetworkModel`: `topology_seed`,
  `queueing`, `tie_break`, `service_order`. The 0.3.0-style behaviour is
  `queueing="shared_fifo", tie_break="first", service_order="id"`.
  It reproduces 0.3.0 exactly where shortest paths are unique (e.g. the
  ring). Where paths tie, its lowest-id rule differs from the one networkx
  used, and delivered traffic can differ by several points: 5.8 on the 3x3
  grid under equal split (`qa/retest/qa_pr5.log` at tag v0.4.0).
* Routing distances use vectorised Floyd-Warshall instead of networkx.
* Tie-breaks and service order, and the `random` baseline in the benchmark
  scripts (`congestgym.baselines.policy_rng`), draw from their own streams
  derived from the episode seed, independent of the traffic generator.

## 0.3.0

Dynamics and reward changed in this release, so results from 0.2.0 are not
comparable.

* `reset()` without a seed no longer replays the same traffic.
* Fractional link service replaces all-or-nothing packet service.
* Node buffers are enforced while traffic is moving.
* Delay and deadline terms now include packets still waiting in queues.
* Added: routing-aware baselines, node bottlenecks, SAC/TD3 training,
  scenario-matrix benchmark, input validation, `rgb_array` rendering.
* Multi-agent env: every agent controls at least one link.
