"""Researcher probes, 15 Oct 2026. Read-only use of the package.

1. Node-id bias: on the 3x3 grid with uniform traffic, nodes in symmetric
   positions should see the same delivered fraction for the traffic they source.
2. Head-of-line waste: link budget left unused in a step while packets that
   need that link sat blocked behind a packet for a different link.
3. Speed vs size, and the share of step time spent in all-pairs Dijkstra.
4. Random topology: does the edge count (action-space size) change with seed?
"""
import time, sys
import numpy as np
from congestgym.env import CongestGymEnv
from congestgym.baselines import POLICIES

STEPS, WARM, SEEDS = 200, 20, range(2000, 2010)


def run(policy_name, seeds=SEEDS, **kw):
    pol = POLICIES[policy_name]
    by_src_off = np.zeros(9); by_src_del = np.zeros(9)
    waste = budget_total = 0.0
    for s in seeds:
        env = CongestGymEnv(topology="grid", num_nodes=9, bandwidth_budget=0.8, seed=s, **kw)
        obs, _ = env.reset(seed=s)
        net = env.network
        cap = net._edge_capacity
        orig_gen = env.traffic.generate
        born = []
        def gen(step, _g=orig_gen):
            p = _g(step); born.extend((step, q) for q in p); return p
        env.traffic.generate = gen
        rng = np.random.default_rng(s)
        for t in range(STEPS):
            a = np.asarray(pol(env, obs, rng), dtype=np.float32)
            bw = env._apply_budget(np.clip(a[:env.num_edges], 0, 1))
            obs, r, _, _, info = env.step(a)
            if t >= WARM:
                served = net._served
                pend = env._pending.astype(float)
                left = np.maximum(cap * bw - served, 0)
                waiting = np.maximum(pend - served, 0)
                waste += np.minimum(left, waiting).sum()
                budget_total += (cap * bw).sum()
        for step, p in born:
            if step < WARM or step >= STEPS - 30:   # skip warmup and packets with no time to finish
                continue
            by_src_off[p.src] += p.size
            if p.path and p.arrived:
                by_src_del[p.src] += p.size
    return by_src_del / by_src_off, waste / budget_total


print("== 1 and 2: per-source delivered fraction, HOL waste (grid 3x3, budget 0.8, load 1, 10 test seeds)")
print("   node layout:  0 1 2 / 3 4 5 / 6 7 8 ; corners 0,2,6,8 ; edges 1,3,5,7 ; centre 4")
for name in ["max_bandwidth", "queue_proportional", "congestion_aware"]:
    frac, w = run(name)
    c = frac[[0, 2, 6, 8]]; e = frac[[1, 3, 5, 7]]
    print(f"{name:20s} corners {np.round(c,3)}  edges {np.round(e,3)}  centre {frac[4]:.3f}"
          f"  | node0-node8 {frac[0]-frac[8]:+.3f} | HOL-wasted budget {100*w:.1f}%")
sys.stdout.flush()

print("\n== 3: speed vs size (grid, max_bandwidth, load scaled with N/9, 300 steps)")
import congestgym.core.network as nm
for n in [9, 16, 25, 36, 49, 64]:
    env = CongestGymEnv(topology="grid", num_nodes=n, load_scale=n / 9, bandwidth_budget=0.8,
                      max_steps=10_000, seed=0)
    obs, _ = env.reset(seed=0)
    a = np.ones(env.action_space.shape, dtype=np.float32)
    t_route = 0.0
    orig = env.network.update_routing
    def timed(w, _o=orig):
        global t_route
        t0 = time.perf_counter(); _o(w); t_route += time.perf_counter() - t0
    env.network.update_routing = timed
    t0 = time.perf_counter()
    for _ in range(300):
        env.step(a)
    dt = time.perf_counter() - t0
    print(f"N={n:3d} E={env.num_edges:3d}  {300/dt:7.1f} steps/s   routing share {100*t_route/dt:4.1f}%")
    sys.stdout.flush()

print("\n== 4: random topology edge count by seed (9 nodes)")
print([CongestGymEnv(topology="random", num_nodes=9, seed=s).num_edges for s in range(8)])
