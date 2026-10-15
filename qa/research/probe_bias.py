"""Separate the two candidate causes of the per-node bias (max_bandwidth, grid 3x3).
tie   : 'fixed' = simulator as shipped; 'random' = weights jittered by U(0, 1e-3)
        each step, so equal-cost paths are chosen at random (ECMP-like, per step).
order : 'id' = nodes served in id order (as shipped); 'shuffle' = new random order each step.
"""
import inspect, textwrap
import numpy as np
import carla_gym.core.network as nm
from carla_gym.env import CarlaGymEnv
from carla_gym.utils.qos import EpisodeQoS

src = textwrap.dedent(inspect.getsource(nm.NetworkModel.forward))
old = "for node in self.graph.nodes():\n        blocked = False"
assert old in src
src = src.replace(old, "for node in self._order():\n        blocked = False")
ns = {}
exec(src, nm.__dict__, ns)
forward_patched = ns["forward"]

STEPS, WARM, SEEDS = 200, 20, range(2000, 2010)
for tie in ["fixed", "random"]:
    for order in ["id", "shuffle"]:
        frac_off = np.zeros(9); frac_del = np.zeros(9); tot = EpisodeQoS()
        for s in SEEDS:
            env = CarlaGymEnv(topology="grid", num_nodes=9, bandwidth_budget=0.8, seed=s)
            net = env.network
            rng = np.random.default_rng(s + 77)
            nodes = list(net.graph.nodes())
            net._order = (lambda: list(rng.permutation(nodes))) if order == "shuffle" else (lambda: nodes)
            net.forward = forward_patched.__get__(net)
            if tie == "random":
                o = net.update_routing
                net.update_routing = lambda w, _o=o: _o(np.asarray(w) + rng.uniform(0, 1e-3, len(w)))
            born = []
            g = env.traffic.generate
            env.traffic.generate = lambda t, _g=g: (lambda p: (born.extend((t, q) for q in p), p)[1])(_g(t))
            env.reset(seed=s)
            a = np.ones(env.action_space.shape, dtype=np.float32)
            for t in range(STEPS):
                _, _, _, _, info = env.step(a)
                if t >= WARM: tot.add(info)
            for t, p in born:
                if WARM <= t < STEPS - 30:
                    frac_off[p.src] += p.size
                    frac_del[p.src] += p.size if (p.path and p.arrived) else 0
        f = frac_del / frac_off
        print(f"tie={tie:6s} order={order:7s} delivered {tot.summary()['delivered_fraction']:.3f}  "
              f"per-source min {f.min():.3f} max {f.max():.3f} spread {f.max()-f.min():.3f}  "
              f"node0 {f[0]:.3f} node8 {f[8]:.3f}")
