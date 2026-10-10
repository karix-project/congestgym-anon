"""M5: missing input validation."""
import gymnasium as gym
import numpy as np

import carla_gym  # noqa: F401 - registers CarlaGym-v0
from carla_gym.env import CarlaGymEnv


def attempt(label, fn):
    try:
        print(f"{label}: {fn()}")
    except Exception as exc:  # noqa: BLE001 - the exception type is the finding
        print(f"{label}: raised {type(exc).__name__}: {exc}")


def bad_budget(b):
    env = CarlaGymEnv(bandwidth_budget=b, seed=0)
    applied = env._apply_budget(np.ones(env.num_edges, dtype=np.float32))
    return f"accepted, applied bandwidth sum = {applied.sum():.2f}"


for b in (-0.5, 0.0, 1.5):
    attempt(f"bandwidth_budget={b}", lambda b=b: bad_budget(b))


def wrong_length():
    env = CarlaGymEnv(seed=0)
    env.reset(seed=0)
    env.step(np.ones(5, dtype=np.float32))
    return "accepted silently"


attempt("wrong-length action", wrong_length)

for n in (7, 9, 10):
    attempt(f"grid num_nodes={n}", lambda n=n: f"built {CarlaGymEnv(num_nodes=n, seed=0).num_nodes} nodes")


def make_max_steps(requested):
    env = gym.make("CarlaGym-v0", max_steps=requested)
    env.reset(seed=0)
    for t in range(1, 1000):
        *_, truncated, _ = env.step(env.action_space.sample())
        if truncated:
            return f"gym.make(max_steps={requested}) truncated at step {t}"
    return "never truncated"


for requested in (50, 500):
    attempt("register vs max_steps", lambda r=requested: make_max_steps(r))


def step_after_truncation():
    env = CarlaGymEnv(max_steps=3, seed=0)
    env.reset(seed=0)
    for _ in range(3):
        env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    return "step() after truncation accepted silently"


attempt("step after truncation", step_after_truncation)
