"""Tester/QA, 15 Oct 2026: detour share of the TUNED congestion_aware on the
random graph (finding QE-1). Same measure as qa/research/e4_detours.py.

    PYTHONPATH=. python qa/retest/qa_e1_tuned_detours.py
"""
import networkx as nx
import numpy as np

from carla_gym.baselines import make_policy, policy_rng
from carla_gym.utils.stats import ci95
from examples.train_agent import make_env, parse_args
from scripts.run_training_matrix import SCENARIOS

args = parse_args(SCENARIOS["random9_l15_b06"])
for label, params in [("untuned congestion_aware", {}),
                      ("tuned congestion_aware (floor 0.1, base 0.3, slope 0.7)", dict(floor=0.1, base=0.3, slope=0.7, signal="max"))]:
    fn = make_policy("congestion_aware", **params)
    shares = []
    for seed in range(2000, 2010):
        env = make_env(args, 0, rescale=False)()
        base = env.unwrapped
        hop = dict(nx.all_pairs_shortest_path_length(base.network.graph))
        born = []
        gen = base.traffic.generate
        base.traffic.generate = lambda t, _g=gen: (lambda p: (born.extend(p), p)[1])(_g(t))
        obs, _ = env.reset(seed=seed)
        rng = policy_rng(seed)
        for _ in range(args.steps):
            obs, *_ = env.step(fn(base, obs, rng))
        pk = [p for p in born if p.birth_step >= 20 and p.path and p.arrived]
        shares.append(np.mean([(len(p.path) - 1) > hop[p.src][p.dst] for p in pk]))
    print(f"random9_l15_b06  {label:58s} on a detour {100*np.mean(shares):.2f}% +/- {100*ci95(shares):.2f}")
