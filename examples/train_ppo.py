"""
examples/train_ppo.py  --  train PPO on CarlaGym and compare to the heuristics.

    pip install stable-baselines3 torch
    set PYTHONPATH=.            (PowerShell:  $env:PYTHONPATH=".")
    python examples/train_ppo.py --timesteps 300000 --budget 0.8
    python examples/train_ppo.py --trace my_trace.csv --budget 0.8   # dataset-driven

Train on the CONSTRAINED setting (--budget 0.8). Unconstrained, the optimum is
"output 1.0 everywhere" and there is nothing to learn.
"""
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from carla_gym.env import CarlaGymEnv
from carla_gym.core.trace_traffic import TraceTrafficGenerator
from examples.random_agent_test import policy_max, policy_queue_proportional, policy_random


def make_env(args, seed):
    def _f():
        traffic = None
        if args.trace:
            traffic = TraceTrafficGenerator(list(range(args.nodes)), args.trace,
                                            target_load=args.load, seed=seed)
        env = CarlaGymEnv(topology=args.topology, num_nodes=args.nodes,
                          max_steps=args.steps, bandwidth_budget=args.budget,
                          traffic=traffic, seed=seed)
        return Monitor(env)
    return _f


def rollout(env, act_fn, seeds, steps, warmup=20):
    # `env` here is Monitor-wrapped (see make_env); Monitor does not forward
    # custom attributes like num_edges/num_nodes that the reference policies
    # need, so hand those policies the unwrapped env while still stepping
    # the wrapped one (to keep Monitor's episode logging intact).
    base_env = env.unwrapped
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s); r = []
        for t in range(steps):
            obs, rew, term, trunc, info = env.step(act_fn(base_env, obs))
            r.append(rew)
            if term or trunc: break
        out.append(np.mean(r[warmup:]))
    return float(np.mean(out)), float(np.std(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--topology", default="grid")
    p.add_argument("--nodes", type=int, default=9)
    p.add_argument("--budget", type=float, default=0.8)
    p.add_argument("--trace", default=None, help="CSV: step,src,dst,volume")
    p.add_argument("--load", type=float, default=104.0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="ppo_carla")
    a = p.parse_args()

    # Distinct seed per env rank, so parallel envs do not share one traffic stream.
    venv = DummyVecEnv([make_env(a, a.seed + rank) for rank in range(a.n_envs)])
    venv.seed(a.seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True)   # reward scale ~[-2,1]
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, gamma=0.95,
                learning_rate=3e-4, ent_coef=0.0, seed=a.seed, verbose=1,
                tensorboard_log=None)
    model.learn(total_timesteps=a.timesteps)
    model.save(a.save)

    # ---- held-out evaluation: seeds the agent never trained on ----
    test_seeds = range(1000, 1010)
    env = make_env(a, 0)()
    ppo_fn = lambda e, o: model.predict(o, deterministic=True)[0]
    rows = []
    for name, fn in [("PPO", ppo_fn), ("queue_proportional", policy_queue_proportional),
                     ("max_bandwidth", policy_max),
                     ("random", lambda e, o: policy_random(e, o, np.random.default_rng()))]:
        m, sd = rollout(env, fn, test_seeds, a.steps)
        rows.append({"policy": name, "reward_mean": round(m, 3), "reward_std": round(sd, 3)})
    print("\n=== Held-out evaluation ===")
    print(pd.DataFrame(rows).sort_values("reward_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
