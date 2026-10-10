"""H1: un-seeded reset() replays identical traffic every episode.

Prints True while the bug is present, False once it is fixed.
"""
import numpy as np

from carla_gym.env import CarlaGymEnv
from carla_gym.marl_env import MultiAgentCarlaGymEnv


def eps(env, n=15):
    env.reset()
    a = np.full(2 * env.num_edges, 0.5, np.float32)
    return [round(env.step(a)[4]["offered_units"], 4) for _ in range(n)]


e = CarlaGymEnv(seed=0, max_steps=15)
e.reset(seed=0)
first, second = eps(e), eps(e)
print("single-agent: consecutive un-seeded episodes identical:", first == second)


def marl_eps(env, n=15):
    obs, _ = env.reset()
    out = []
    for _ in range(n):
        acts = {a: np.full(env.action_space(a).shape, 0.5, np.float32) for a in env.agents}
        obs, *_ = env.step(acts)
        out.append(np.round(obs[env.possible_agents[0]], 5).tolist())
    return out


m = MultiAgentCarlaGymEnv(seed=0, max_steps=15)
m.reset(seed=0)
print("multi-agent:  consecutive un-seeded episodes identical:", marl_eps(m) == marl_eps(m))
