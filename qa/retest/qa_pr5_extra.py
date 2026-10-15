"""Tester/QA, PR #5: edge cases (H) and RNG stream independence (finding Q1).
    PYTHONPATH=. python qa/retest/qa_pr5_extra.py
"""
import numpy as np
from carla_gym.env import CarlaGymEnv
print("H edge cases")
cases = [dict(topology="grid", num_nodes=4), dict(topology="ring", num_nodes=3), dict(topology="star", num_nodes=2),
         dict(topology="mesh", num_nodes=2), dict(topology="random", num_nodes=3), dict(topology="grid", num_nodes=64),
         dict(bandwidth_budget=None), dict(load_scale=0.0), dict(topology_seed=None, topology="random")]
for kw in cases:
    for act in ("zeros", "ones", "random"):
        try:
            env = CarlaGymEnv(seed=0, max_steps=30, **kw); env.reset(seed=0)
            a = {"zeros": np.zeros, "ones": np.ones}.get(act, lambda s: np.random.default_rng(0).uniform(0, 1, s))(env.action_space.shape)
            for _ in range(30):
                obs, r, *_ = env.step(np.asarray(a, dtype=np.float32))
            assert env.observation_space.contains(obs) and np.isfinite(r)
        except Exception as e:
            print("   FAIL", kw, act, type(e).__name__, e); break
    else:
        continue
print("   done (no FAIL lines above = all passed)")

print("\nRNG streams after env.reset(seed=7):")
env = CarlaGymEnv(seed=7); env.reset(seed=7)
a = env.network.rng.bit_generator.state["state"]; b = env.traffic.rng.bit_generator.state["state"]
print("   network and traffic generators in identical state:", a == b)
n_raw = env.network.rng.integers(0, 2**62, 5); t_raw = env.traffic.rng.integers(0, 2**62, 5)
print("   first 5 raw draws identical:", bool(np.all(n_raw == t_raw)))
