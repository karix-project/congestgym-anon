"""N1 (new finding): node buffers exceed `node_buffer` while traffic is moving.

forward() checked downstream space against packets already re-queued this
step only, so a node that had not been processed yet looked empty.
"""
import numpy as np

from carla_gym.env import CarlaGymEnv

for w in (0.3, 0.6, 0.8, 1.0):
    for load in (1.0, 2.0):
        env = CarlaGymEnv(max_steps=150, bandwidth_budget=None, load_scale=load, seed=0)
        env.reset(seed=0)
        a = np.full(2 * env.num_edges, w, np.float32)
        worst = 0.0
        for _ in range(150):
            env.step(a)
            worst = max(worst, float(env.network.get_node_occupancy().max()))
        flag = "OVERFLOW" if worst > env.network.node_buffer + 1e-6 else "ok"
        print(f"weight {w:.1f} load x{load:.0f}: fullest node {worst:6.1f} / {env.network.node_buffer:.0f}  {flag}")
