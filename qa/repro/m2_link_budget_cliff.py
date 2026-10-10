"""M2: a packet larger than a link's per-step budget can never move.

A single 12-unit video packet on a 30-unit link at weight 0.3 (9 units/step)
is delivered 0.0 units in 50 steps while the bug is present.
"""
import numpy as np

from carla_gym.core.network import NetworkModel
from carla_gym.core.traffic import Packet, QoSClass
from carla_gym.env import CarlaGymEnv

net = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=1000.0, seed=0)
net.update_routing(np.ones(net.num_edges()))
net.admit([Packet(0, 1, QoSClass.VIDEO, 12.0, 1.5, 8, 0)])
delivered = sum(net.forward(np.full(net.num_edges(), 0.3), s)["delivered_units"] for s in range(50))
print("single 12-unit packet, weight 0.3, 50 steps -> delivered units:", delivered)

print("constant-weight sweep (unconstrained, 150 steps, warmup 20, seed 0):")
for w in (0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1.0):
    env = CarlaGymEnv(max_steps=150, bandwidth_budget=None, seed=0)
    env.reset(seed=0)
    a = np.full(2 * env.num_edges, w, np.float32)
    r = [env.step(a)[1] for _ in range(150)]
    print(f"  weight {w:.2f} -> mean reward {np.mean(r[20:]):+.3f}")
