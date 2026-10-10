"""M1: reward blind spots.

(a) Constant low weights fill every buffer, yet deadline misses stay at 0
    because delay/deadline are only measured on delivery.
(b) The per-step delivered/offered ratio exceeds 1 while a backlog drains,
    and the reward clips it, so draining looks like steady state.
"""
import numpy as np

from carla_gym.env import CarlaGymEnv

STEPS = 120

for w in (0.30, 0.45, 1.0):
    env = CarlaGymEnv(max_steps=STEPS, bandwidth_budget=None, seed=0)
    env.reset(seed=0)
    a = np.full(2 * env.num_edges, w, np.float32)
    rewards, misses = [], 0
    for _ in range(STEPS):
        _, r, _, _, info = env.step(a)
        rewards.append(r)
        misses += info["deadline_misses"]
    occ = env.network.get_node_occupancy()
    print(f"(a) constant weight {w:.2f}: mean reward {np.mean(rewards):+.3f}, "
          f"final backlog {info['backlog_units']:.0f} / {env.network.node_buffer * env.num_nodes:.0f}, "
          f"fullest node {occ.max():.0f}/{env.network.node_buffer:.0f}, deadline misses {misses}")

# (b) starve for 40 steps, then open every link: the backlog drains
env = CarlaGymEnv(max_steps=STEPS, bandwidth_budget=None, seed=0)
env.reset(seed=0)
over = 0
for t in range(STEPS):
    w = 0.5 if t < 40 else 1.0
    _, _, _, _, info = env.step(np.full(2 * env.num_edges, w, np.float32))
    raw = info["delivered_units"] / info["offered_units"]
    over += raw > 1.0
print(f"(b) raw delivered/offered > 1 on {over} of {STEPS} steps (reward clips it to 1)")
