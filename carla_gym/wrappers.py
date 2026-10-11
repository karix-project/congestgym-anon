"""
carla_gym/wrappers.py

Action wrappers for ablation studies: let an agent control only one half of
CarlaGymEnv's action (bandwidth or routing) while the other half is held at a
fixed, uninformed default.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ControlSubsetWrapper(gym.ActionWrapper):
    """Expose only part of the action to the agent.

    control="bandwidth": the agent sets the E bandwidth weights; routing
        weights are fixed at 1.0 (hop-count shortest paths).
    control="routing":   the agent sets the E routing weights; bandwidth
        weights are fixed at 1.0 (equal split, scaled down by the budget).
    control="both":      no change.
    """

    CONTROLS = ("both", "bandwidth", "routing")

    def __init__(self, env: gym.Env, control: str = "both"):
        super().__init__(env)
        if control not in self.CONTROLS:
            raise ValueError(f"control must be one of {self.CONTROLS}, got {control!r}")
        self.control = control
        self._edges = env.unwrapped.num_edges
        if control != "both":
            self.action_space = spaces.Box(0.0, 1.0, shape=(self._edges,), dtype=np.float32)

    def action(self, action):
        if self.control == "both":
            return action
        action = np.asarray(action, dtype=np.float32)
        fixed = np.ones(self._edges, dtype=np.float32)
        if self.control == "bandwidth":
            return np.concatenate([action, fixed])
        return np.concatenate([fixed, action])
