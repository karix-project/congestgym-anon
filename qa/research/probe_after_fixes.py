"""Developer check after F1-F3 (15 Oct 2026). Same measures as probe_model.py,
run on 0.4.0 with the legacy-style options and with the new defaults.

"legacy" = queueing shared_fifo, tie_break first, service_order id. Its
tie-break (lowest id) is not the exact rule networkx used in 0.3.0, so its
numbers are close to, not identical with, probe_model.log.

    PYTHONPATH=. python qa/research/probe_after_fixes.py
"""
import time
import numpy as np
from carla_gym.env import CarlaGymEnv
from carla_gym.baselines import POLICIES
from carla_gym.utils.qos import EpisodeQoS

STEPS, WARM, SEEDS = 200, 20, range(2000, 2010)
LEGACY = dict(queueing="shared_fifo", tie_break="first", service_order="id")
NEW = {}


def run(policy_name, opts):
    pol = POLICIES[policy_name]
    off, dlv = np.zeros(9), np.zeros(9)
    waste = budget_total = 0.0
    qos = EpisodeQoS()
    for s in SEEDS:
        env = CarlaGymEnv(topology="grid", num_nodes=9, bandwidth_budget=0.8, seed=s, **opts)
        born = []
        gen = env.traffic.generate
        def record(t, _g=gen):
            p = _g(t); born.extend(p); return p
        env.traffic.generate = record
        obs, _ = env.reset(seed=s)
        net, cap = env.network, env.network._edge_capacity
        rng = np.random.default_rng(s)
        for t in range(STEPS):
            a = np.asarray(pol(env, obs, rng), dtype=np.float32)
            bw = env._apply_budget(np.clip(a[:env.num_edges], 0, 1))
            obs, _, _, _, info = env.step(a)
            if t >= WARM:
                qos.add(info)
                served, pend = net._served, env._pending.astype(float)
                waste += np.minimum(np.maximum(cap * bw - served, 0), np.maximum(pend - served, 0)).sum()
                budget_total += (cap * bw).sum()
        for p in born:
            if WARM <= p.birth_step < STEPS - 30:
                off[p.src] += p.size
                dlv[p.src] += p.size if (p.path and p.arrived) else 0.0
    f = dlv / off
    return qos.summary()["delivered_fraction"], f, waste / budget_total


print("grid 3x3, budget 0.8, load x1, test seeds 2000-2009")
print(f"{'policy':20s} {'options':7s} delivered  per-source min/max/spread   corners 0/2/6/8          HOL-wasted budget")
for name in ["max_bandwidth", "queue_proportional", "congestion_aware"]:
    for label, opts in (("legacy", LEGACY), ("new", NEW)):
        d, f, w = run(name, opts)
        c = np.round(f[[0, 2, 6, 8]], 3)
        print(f"{name:20s} {label:7s} {d:.3f}      {f.min():.3f}/{f.max():.3f}/{f.max()-f.min():.3f}"
              f"          {c}   {100*w:.1f}%", flush=True)

print("\nspeed: grid, load scaled with N/9, equal split, 300 steps, new defaults")
for n in [9, 16, 25, 36, 49, 64]:
    env = CarlaGymEnv(topology="grid", num_nodes=n, load_scale=n / 9, bandwidth_budget=0.8,
                      max_steps=10_000, seed=0)
    env.reset(seed=0)
    a = np.ones(env.action_space.shape, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(300):
        env.step(a)
    print(f"N={n:3d}  {300/(time.perf_counter()-t0):7.1f} steps/s", flush=True)

print("\nrandom topology, 9 nodes, edges for env seeds 0-7:",
      [CarlaGymEnv(topology="random", num_nodes=9, seed=s).num_edges for s in range(8)])
