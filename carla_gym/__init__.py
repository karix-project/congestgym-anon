"""carla_gym: modular network-simulation RL environment (mobile-env-inspired)."""

from gymnasium.envs.registration import register

from carla_gym.env import CarlaGymEnv

__all__ = ["CarlaGymEnv"]
__version__ = "0.2.0"

register(id="CarlaGym-v0", entry_point="carla_gym.env:CarlaGymEnv",
         max_episode_steps=200)
