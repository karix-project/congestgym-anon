"""Researcher, 15 Oct 2026: experiment E4 of docs/NEXT_EXPERIMENTS.md.
Do policies take detours on the bottleneck and on the random graph?

As qa/research/headline_paths.py did for the headline: for every packet born
at step >= 20 and delivered, hops travelled and stretch = hops / hop-count
shortest-path length on the graph. Test seeds 2000-2009. PPO: the saved
models (interval over training seeds); rule-based: interval over test seeds.

    PYTHONPATH=. python qa/research/e4_detours.py
"""
import glob
import os
import sys

import networkx as nx
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.baselines import POLICIES, policy_rng           # noqa: E402
from carla_gym.utils.stats import ci95                         # noqa: E402
from examples.train_agent import make_env, parse_args          # noqa: E402
from scripts.run_training_matrix import known_configurations   # noqa: E402

SEEDS, WARMUP = range(2000, 2010), 20
HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]


def episode(args, act_fn, seed, rescale):
    env = make_env(args, 0, rescale=rescale)()
    base = env.unwrapped
    hop = dict(nx.all_pairs_shortest_path_length(base.network.graph))
    born = []
    gen = base.traffic.generate
    def record(t, _g=gen):
        p = _g(t); born.extend(p); return p
    base.traffic.generate = record
    obs, _ = env.reset(seed=seed)
    for _ in range(args.steps):
        obs, *_ = env.step(act_fn(base, obs))
    h, st = [], []
    for p in born:
        if p.birth_step >= WARMUP and p.path and p.arrived:
            n = len(p.path) - 1
            h.append(n); st.append(n / hop[p.src][p.dst])
    st = np.array(st)
    return np.mean(h), st.mean(), np.mean(st > 1.0 + 1e-9)


def line(label, rows):
    rows = np.array(rows)
    print(f"   {label:34s} hops {rows[:, 0].mean():.3f} +/- {ci95(rows[:, 0]):.3f}   stretch {rows[:, 1].mean():.4f} "
          f"+/- {ci95(rows[:, 1]):.4f}   on a detour {100*rows[:, 2].mean():.2f}% +/- {100*ci95(rows[:, 2]):.2f}", flush=True)


for scen, models in [("grid_bottleneck_b08", ["grid_bottleneck_b08"]),
                     ("random9_l15_b06", ["random9_l15_b06", "random9_l15_b06_routing_only"])]:
    args = parse_args(known_configurations()[scen])
    print(f"\n{scen} ({' '.join(known_configurations()[scen])})")
    for name in HEUR:
        fn = POLICIES[name]
        rows = []
        for s in SEEDS:
            rng = policy_rng(s)
            rows.append(episode(args, lambda e, o, fn=fn, rng=rng: fn(e, o, rng), s, False))
        line(name + " (over test seeds)", rows)
    for folder in models:
        margs = parse_args(known_configurations()[folder])
        per_seed = []
        for path in sorted(glob.glob(f"results/rl/{folder}/ppo_seed*/model.zip")):
            model = PPO.load(path, device="cpu")
            fn = lambda e, o, model=model: model.predict(o, deterministic=True)[0]
            per_seed.append(np.mean([episode(margs, fn, s, True) for s in SEEDS], axis=0))
        line(f"PPO {folder.replace(scen, '').strip('_') or 'full control'} (5 seeds)", per_seed)
