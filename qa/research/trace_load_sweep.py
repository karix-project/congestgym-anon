"""Researcher check: at what target load do the trace scenarios stop being
overloaded, and where do the heuristics separate? Reads committed traces only."""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from carla_gym.baselines import policy_max, policy_queue_proportional, policy_congestion_aware
from carla_gym.core.trace_traffic import TraceTrafficGenerator
from carla_gym.env import CarlaGymEnv
from carla_gym.utils.qos import EpisodeQoS
from carla_gym.utils.stats import ci95

POL = {"max": policy_max, "qp": policy_queue_proportional, "cong": policy_congestion_aware}
for trace in ("pcap", "cs448b"):
    print(f"== {trace} trace, grid-16, budget 0.5, 200 steps, warm-up 20, 5 seeds")
    print("  load | " + " | ".join(f"{p}: reward  deliv% drop%" for p in POL) + " | qp - max")
    for load in (20, 40, 60, 80, 110):
        cells, rew = [], {}
        for name, fn in POL.items():
            r, d, x = [], [], []
            for seed in range(2000, 2005):
                gen = TraceTrafficGenerator(list(range(16)), f"data/{trace}_trace.csv", target_load=float(load), seed=seed)
                env = CarlaGymEnv(topology="grid", num_nodes=16, bandwidth_budget=0.5, max_steps=200, traffic=gen, seed=seed)
                obs, _ = env.reset(seed=seed); q = EpisodeQoS(); rr = []
                for t in range(200):
                    obs, rw, _, _, info = env.step(fn(env, obs))
                    if t >= 20: q.add(info); rr.append(rw)
                s = q.summary(); r.append(np.mean(rr)); d.append(100 * s["delivered_fraction"]); x.append(100 * s["dropped_fraction"])
            rew[name] = np.array(r)
            cells.append(f"{np.mean(r):+.3f}  {np.mean(d):5.1f} {np.mean(x):5.1f}")
        diff = rew["qp"] - rew["max"]
        print(f"  {load:4d} | " + " | ".join(cells) + f" | {diff.mean():+.3f} +/- {ci95(diff):.3f}")
