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
from carla_gym.core.traffic import TrafficGenerator
from carla_gym.utils.reward import RewardCalculator


class CarlaGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, topology: str = "grid", num_nodes: int = 9,
                 link_capacity: float = 30.0, node_buffer: float = 120.0,
                 load_scale: float = 1.0, max_steps: int = 200,
                 k: float = 0.5, c: float = 1.0, delay_reference: float = 10.0,
                 bandwidth_budget: float | None = 0.8,
                 traffic=None,
                 render_mode: str | None = None, seed: int | None = None):
        """
        bandwidth_budget : if set (0, 1], the agent may only spend that
            fraction of total link capacity per step; actions summing above
            it are rescaled down. Without this constraint the optimal policy
            is degenerate ("allocate 1.0 everywhere") and there is nothing
            interesting to learn.
        """
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.bandwidth_budget = bandwidth_budget
        self._seed = seed

        self.network = NetworkModel(topology=topology, num_nodes=num_nodes,
                                    link_capacity=link_capacity,
                                    node_buffer=node_buffer, seed=seed)
        # `traffic`: optional pre-built generator (e.g. TraceTrafficGenerator).
        self.traffic = traffic or TrafficGenerator(nodes=list(self.network.graph.nodes()),
                                        load_scale=load_scale, seed=seed)
        self.reward_calc = RewardCalculator(k=k, c=c, delay_reference=delay_reference)

        self.num_edges = self.network.num_edges()
        self.num_nodes = self.network.num_nodes

        # Action: per-link bandwidth weight (E), followed by per-link routing weight (E)
        self.action_space = spaces.Box(0.0, 1.0, shape=(2 * self.num_edges,), dtype=np.float32)
        # Observation: [link utilization (E) | link pending demand (E) | node occupancy (N)]
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(2 * self.num_edges + self.num_nodes,), dtype=np.float32)

        self._step_count = 0
        self._pending = np.zeros(self.num_edges, dtype=np.float32)
        self._visualizer = None

    # ---------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.network.reset()
        self.traffic.reset(seed=seed if seed is not None else self._seed)
        self._step_count = 0
        self._pending = np.zeros(self.num_edges, dtype=np.float32)
        return self._observe(), {"step": 0}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        
        # Split action into bandwidth allocation (first E) and routing weights (last E)
        bw_action = self._apply_budget(action[:self.num_edges])
        routing_action = action[self.num_edges:]
        
        # Apply routing weights so admit() routes dynamically
        self.network.update_routing(routing_action)

        packets = self.traffic.generate(self._step_count)
        offered, admit_dropped = self.network.admit(packets)
        stats = self.network.forward(bw_action, self._step_count)

        dropped = admit_dropped + stats["dropped_units"]
        delivered = stats["delivered_units"]

        lat = stats["latencies"]
        if lat:
            num = sum(d * s * sz for d, s, sz in lat)
            den = sum(sz for _, _, sz in lat)
            weighted_delay = num / den
            raw_delay = sum(d for d, _, _ in lat) / len(lat)
        else:
            weighted_delay = raw_delay = 0.0

        t_n, d_n, l_n, m_n = self.reward_calc.normalize(delivered, offered, weighted_delay, dropped, stats["deadline_misses"], stats["delivered_count"])
        reward = self.reward_calc.compute(t_n, d_n, l_n, m_n)

        self._pending = stats["pending_units"].astype(np.float32)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        info = {
            "step": self._step_count,
            "reward_throughput": t_n,
            "reward_delay": d_n,
            "reward_loss": l_n,
            "reward_deadline_miss": m_n,
            "delivered_units": delivered,
            "offered_units": offered,
            "dropped_units": dropped,
            "mean_latency_steps": raw_delay,
            "weighted_latency_steps": weighted_delay,
            "delivered_count": stats["delivered_count"],
            "deadline_misses": stats["deadline_misses"],
            "backlog_units": self.network.backlog_units(),
            "mean_link_utilization": float(stats["utilization"].mean()),
        }
        return self._observe(), float(reward), False, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        from carla_gym.utils.visualization import NetworkVisualizer
        if self._visualizer is None:
            self._visualizer = NetworkVisualizer(self.network)
        return self._visualizer.render(step=self._step_count,
                                       show=self.render_mode == "human")

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
        pending = np.clip(self._pending / max(self.network.link_capacity, 1e-9), 0.0, 1.0)
        occ = np.clip(self.network.get_node_occupancy() / max(self.network.node_buffer, 1e-9),
                      0.0, 1.0)
        return np.concatenate([util, pending, occ]).astype(np.float32)
