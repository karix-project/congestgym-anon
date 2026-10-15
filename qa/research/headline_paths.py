"""Researcher, 15 Oct 2026: QA finding QR-2. Does PPO route headline traffic
over longer paths than congestion_aware?

Headline scenario (grid 3x3, load x1.25, budget 0.6), test seeds 2000-2009,
steps 20..199 as in the result tables. For every packet delivered in that
window: hops travelled, and stretch = hops / shortest possible hops (grid
Manhattan distance). PPO: the 5 saved models; interval over training seeds.

    PYTHONPATH=. python qa/research/headline_paths.py
"""
import glob
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.baselines import POLICIES, policy_rng            # noqa: E402
from carla_gym.utils.stats import ci95                          # noqa: E402
from examples.train_agent import make_env, parse_args           # noqa: E402
from scripts.run_training_matrix import SCENARIOS               # noqa: E402

ARGS = parse_args(SCENARIOS["grid_l125_b06"])
SEEDS = range(2000, 2010)
WARMUP, SIDE = 20, 3


def manhattan(a, b):
    return abs(a // SIDE - b // SIDE) + abs(a % SIDE - b % SIDE)


def episode(act_fn, seed, rescale):
    env = make_env(ARGS, 0, rescale=rescale)()
    base = env.unwrapped
    born = []
    gen = base.traffic.generate
    def record(t, _g=gen):
        p = _g(t); born.extend(p); return p
    base.traffic.generate = record
    obs, _ = env.reset(seed=seed)
    for _ in range(ARGS.steps):
        obs, *_ = env.step(act_fn(base, obs))
    hops, stretch = [], []
    for p in born:
        if p.birth_step >= WARMUP and p.path and p.arrived:
            h = len(p.path) - 1
            hops.append(h); stretch.append(h / manhattan(p.src, p.dst))
    return np.mean(hops), np.mean(stretch), np.mean(np.array(stretch) > 1.0)


def summarise(label, rows):
    rows = np.array(rows)
    print(f"{label:24s} hops {rows[:, 0].mean():.3f} +/- {ci95(rows[:, 0]):.3f}   "
          f"stretch {rows[:, 1].mean():.3f} +/- {ci95(rows[:, 1]):.3f}   "
          f"share of packets on a longer-than-shortest path {100*rows[:, 2].mean():.1f}% +/- {100*ci95(rows[:, 2]):.1f}")


print(f"headline flags: {' '.join(SCENARIOS['grid_l125_b06'])}; test seeds 2000-2009; packets born at step >= {WARMUP}, delivered")
for name in ["congestion_aware", "queue_proportional", "load_aware_routing", "max_bandwidth"]:
    fn = POLICIES[name]
    rows = []
    for s in SEEDS:
        rng = policy_rng(s)
        rows.append(episode(lambda e, o, fn=fn, rng=rng: fn(e, o, rng), s, False))
    summarise(name + " (over test seeds)", rows)

per_seed = []
for path in sorted(glob.glob("results/rl/grid_l125_b06/ppo_seed*/model.zip")):
    model = PPO.load(path, device="cpu")
    fn = lambda e, o, model=model: model.predict(o, deterministic=True)[0]
    per_seed.append(np.mean([episode(fn, s, True) for s in SEEDS], axis=0))
summarise("PPO (over 5 train seeds)", per_seed)
