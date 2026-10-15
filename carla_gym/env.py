"""
carla_gym/env.py

The Core. Thin by design: it owns the spaces and the step()/reset()
sequencing, and delegates everything else. If queueing math or topology
code starts appearing here, it belongs in core/network.py instead.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from carla_gym.core.network import NetworkModel
from carla_gym.core.step import advance
from carla_gym.core.traffic import TrafficGenerator
from carla_gym.utils.reward import RewardCalculator


class CarlaGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, topology: str = "grid", num_nodes: int = 9,
                 link_capacity: float = 30.0, node_buffer: float = 120.0,
                 load_scale: float = 1.0, max_steps: int = 200,
                 k: float = 0.5, c: float = 1.0, m: float = 0.5,
                 delay_reference: float = 10.0,
                 bandwidth_budget: float | None = 0.8,
                 traffic=None, bottlenecks: dict[int, float] | None = None,
                 topology_seed: int | None = 0, queueing: str = "per_link",
                 tie_break: str = "random", service_order: str = "random",
                 render_mode: str | None = None, seed: int | None = None):
        """
        k, c, m : reward coefficients for delay, packet loss and deadline
            misses (see utils/reward.py).
        bottlenecks : optional {node id: scale in (0, 1]} of degraded nodes;
            see NetworkModel. Observations stay normalised to each link's and
            node's own (possibly reduced) capacity.
        topology_seed, queueing, tie_break, service_order : simulator
            options; see core/network.py. `seed` seeds traffic, tie-breaks
            and service order; the graph comes from `topology_seed` only.
        bandwidth_budget : if set (0, 1], the agent may only spend that
            fraction of the maximum total allocation (every link at 1.0) per
            step; actions summing above it are rescaled down. The budget is
            counted in allocation weights, not capacity units, so on links
            with reduced capacity (bottlenecks) a weight costs the same. Without this constraint the optimal policy
            is degenerate ("allocate 1.0 everywhere") and there is nothing
            interesting to learn.
        """
        super().__init__()
        if bandwidth_budget is not None and not 0.0 < bandwidth_budget <= 1.0:
            raise ValueError(f"bandwidth_budget must be None or in (0, 1], got {bandwidth_budget}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        modes = self.metadata["render_modes"]
        if render_mode is not None and render_mode not in modes:
            raise ValueError(f"render_mode must be one of {modes}, got {render_mode!r}")
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.bandwidth_budget = bandwidth_budget
        self._seed = seed

        self.network = NetworkModel(topology=topology, num_nodes=num_nodes,
                                    link_capacity=link_capacity,
                                    node_buffer=node_buffer, seed=seed,
                                    bottlenecks=bottlenecks, topology_seed=topology_seed,
                                    queueing=queueing, tie_break=tie_break,
                                    service_order=service_order)
        # `traffic`: optional pre-built generator (e.g. TraceTrafficGenerator).
        self.traffic = traffic or TrafficGenerator(nodes=list(self.network.graph.nodes()),
                                        load_scale=load_scale, seed=seed)
        self.reward_calc = RewardCalculator(k=k, c=c, m=m, delay_reference=delay_reference)

        self.num_edges = self.network.num_edges()
        self.num_nodes = self.network.num_nodes

        # Action: per-link bandwidth weight (E), followed by per-link routing weight (E)
        self.action_space = spaces.Box(0.0, 1.0, shape=(2 * self.num_edges,), dtype=np.float32)
        # Observation: [link utilization (E) | link pending demand (E) | node occupancy (N)]
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(2 * self.num_edges + self.num_nodes,), dtype=np.float32)

        self._step_count = 0
        self._pending = np.zeros(self.num_edges, dtype=np.float32)
        self._needs_reset = False
        self._visualizer = None

    # ---------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.network.reset(seed=seed)
        # Re-seed traffic only on an explicit seed. The constructor seed already
        # seeded the generator once; un-seeded resets must continue that stream
        # so successive episodes see fresh traffic.
        self.traffic.reset(seed=seed)
        self._step_count = 0
        self._pending = np.zeros(self.num_edges, dtype=np.float32)
        self._needs_reset = False
        return self._observe(), {"step": 0}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Episode has ended (truncated at max_steps); "
                               "call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape} "
                             f"(E bandwidth weights then E routing weights), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or infinite values")
        action = np.clip(action, 0.0, 1.0)

        # Split action into bandwidth allocation (first E) and routing weights (last E)
        bw_action = self._apply_budget(action[:self.num_edges])
        routing_action = action[self.num_edges:]

        reward, info, self._pending = advance(
            self.network, self.traffic, self.reward_calc,
            bw_action, routing_action, self._step_count)

        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        self._needs_reset = truncated
        info["step"] = self._step_count
        return self._observe(), float(reward), False, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        from carla_gym.utils.visualization import NetworkVisualizer
        if self._visualizer is None:
            self._visualizer = NetworkVisualizer(self.network)
        if self.render_mode == "rgb_array":
            return self._visualizer.render(step=self._step_count, return_rgb=True)
        self._visualizer.render(step=self._step_count, show=True)
        return None

    def close(self):
        self._visualizer = None

    # ---------------------------------------------------------------- #
    def _apply_budget(self, action: np.ndarray) -> np.ndarray:
        """Rescale the action down if it exceeds the spending budget. Scaling
        (rather than clipping) preserves the agent's *relative* priorities,
        which is the part of the decision we want it to learn."""
        if self.bandwidth_budget is None:
            return action
        allowed = self.bandwidth_budget * self.num_edges
        total = float(action.sum())
        if total > allowed and total > 1e-9:
            action = action * (allowed / total)
        return action.astype(np.float32)

    def _observe(self) -> np.ndarray:
        util = np.clip(self.network.get_edge_utilization(), 0.0, 1.0)
        pending = np.clip(self._pending / self.network.get_edge_capacities(), 0.0, 1.0)
        occ = np.clip(self.network.get_node_occupancy() / self.network.get_node_buffers(),
                      0.0, 1.0)
        return np.concatenate([util, pending, occ]).astype(np.float32)
