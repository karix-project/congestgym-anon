"""carla_gym: modular network-simulation RL environment (mobile-env-inspired)."""

from gymnasium.envs.registration import register

from carla_gym.env import CarlaGymEnv
from carla_gym.marl_env import MultiAgentCarlaGymEnv

__all__ = ["CarlaGymEnv", "MultiAgentCarlaGymEnv"]
__version__ = "0.4.0"

# No max_episode_steps here: the env truncates itself at its own `max_steps`
# (default 200), so gym.make("CarlaGym-v0", max_steps=...) is honoured.
register(id="CarlaGym-v0", entry_point="carla_gym.env:CarlaGymEnv")
