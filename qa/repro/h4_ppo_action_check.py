"""H4: inspect a trained PPO model's deterministic actions.

Run after:  python examples/train_ppo.py --timesteps 20000 --budget 0.8
Usage:      python qa/repro/h4_ppo_action_check.py [path/to/ppo_carla.zip]
"""
import sys

import numpy as np
from stable_baselines3 import PPO

from carla_gym.env import CarlaGymEnv

path = sys.argv[1] if len(sys.argv) > 1 else "ppo_carla.zip"
model = PPO.load(path)

env = CarlaGymEnv(bandwidth_budget=0.8, max_steps=200, seed=1000)
obs, _ = env.reset(seed=1000)
raw, rewards = [], []
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    raw.append(action)
    obs, r, *_ = env.step(action)
    rewards.append(r)
raw = np.array(raw)
clipped = np.clip(raw, 0.0, 1.0)
E = env.num_edges
print(f"raw deterministic action: mean {raw.mean():+.3f}, min {raw.min():+.3f}, max {raw.max():+.3f}")
print(f"after clip to [0,1]: bandwidth mean {clipped[:, :E].mean():.3f}, "
      f"routing mean {clipped[:, E:].mean():.3f}, "
      f"share of bandwidth entries below 0.4: {(clipped[:, :E] < 0.4).mean():.1%}")
print(f"policy log_std mean: {model.policy.log_std.mean().item():+.3f} "
      f"(std {np.exp(model.policy.log_std.mean().item()):.3f})")
print(f"mean reward (steps 20+): {np.mean(rewards[20:]):+.3f}")
