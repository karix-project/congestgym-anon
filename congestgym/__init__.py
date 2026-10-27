"""congestgym: modular network-simulation RL environment (mobile-env-inspired)."""

from gymnasium.envs.registration import register

from congestgym.env import CongestGymEnv
from congestgym.marl_env import MultiAgentCongestGymEnv

__all__ = ["CongestGymEnv", "MultiAgentCongestGymEnv"]
__version__ = "0.4.0"

# No max_episode_steps here: the env truncates itself at its own `max_steps`
# (default 200), so gym.make("CongestGym-v0", max_steps=...) is honoured.
register(id="CongestGym-v0", entry_point="congestgym.env:CongestGymEnv")
