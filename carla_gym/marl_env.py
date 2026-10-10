"""
carla_gym/marl_env.py

Multi-agent version of CarlaGymEnv using PettingZoo ParallelEnv API.
Each node is an independent agent controlling the bandwidth and routing weights
for its outgoing links.
"""
from __future__ import annotations

import functools
import numpy as np
from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
except ImportError:
    # Fallback/mock if pettingzoo is not installed
    class ParallelEnv:
        pass

from carla_gym.core.network import NetworkModel
from carla_gym.core.step import advance
from carla_gym.core.traffic import TrafficGenerator
from carla_gym.utils.reward import RewardCalculator


class MultiAgentCarlaGymEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "carla_marl_v0"}

    def __init__(self, topology: str = "grid", num_nodes: int = 9,
                 link_capacity: float = 30.0, node_buffer: float = 120.0,
                 load_scale: float = 1.0, max_steps: int = 200,
                 k: float = 0.5, c: float = 1.0, m: float = 0.5, delay_reference: float = 10.0,
                 bandwidth_budget: float | None = 0.8,
                 traffic=None, seed: int | None = None):
        super().__init__()
        self.max_steps = max_steps
        self.bandwidth_budget = bandwidth_budget
        self._seed = seed

        self.network = NetworkModel(topology=topology, num_nodes=num_nodes,
                                    link_capacity=link_capacity,
                                    node_buffer=node_buffer, seed=seed)
        self.traffic = traffic or TrafficGenerator(nodes=list(self.network.graph.nodes()),
                                                   load_scale=load_scale, seed=seed)
        self.reward_calc = RewardCalculator(k=k, c=c, m=m, delay_reference=delay_reference)

        self.possible_agents = [f"node_{i}" for i in self.network.graph.nodes()]
        self.agents = self.possible_agents[:]

        # Precompute which edges belong to which agent.
        # Since undirected links share one budget, we assign control of the link
        # to the node with the numerically lower ID.
        self.agent_edges = {}
        for agent in self.possible_agents:
            node_id = int(agent.split("_")[1])
            edges = []
            for i, (u, v) in enumerate(self.network._edges):
                if u == node_id: 
                    edges.append(i)
            self.agent_edges[agent] = edges

        self._action_spaces = {}
        self._observation_spaces = {}
        
        obs_dim = 2 * self.network.num_edges() + self.network.num_nodes
        for agent in self.possible_agents:
            num_owned_edges = len(self.agent_edges[agent])
            # Actions: bandwidth + routing for owned edges
            self._action_spaces[agent] = spaces.Box(0.0, 1.0, shape=(2 * num_owned_edges,), dtype=np.float32)
            # Observation: Global state (can be constrained to local state in future iterations)
            self._observation_spaces[agent] = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._step_count = 0
        self._pending = np.zeros(self.network.num_edges(), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.network.reset()
        # Re-seed only on an explicit seed; see CarlaGymEnv.reset().
        self.traffic.reset(seed=seed)
        self._step_count = 0
        self._pending = np.zeros(self.network.num_edges(), dtype=np.float32)
        
        obs = self._observe()
        observations = {a: obs for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Reconstruct global action vector from individual agent actions
        global_bw = np.zeros(self.network.num_edges(), dtype=np.float32)
        global_routing = np.ones(self.network.num_edges(), dtype=np.float32)
        
        for agent, action in actions.items():
            owned = self.agent_edges[agent]
            if owned:
                num_owned = len(owned)
                action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
                bw_act = action[:num_owned]
                rt_act = action[num_owned:]
                for idx, edge_idx in enumerate(owned):
                    global_bw[edge_idx] = bw_act[idx]
                    global_routing[edge_idx] = rt_act[idx]

        global_bw = self._apply_budget(global_bw)
        # Global cooperative reward
        reward, step_info, self._pending = advance(
            self.network, self.traffic, self.reward_calc,
            global_bw, global_routing, self._step_count)
        self._step_count += 1

        trunc = self._step_count >= self.max_steps
        term = False
        
        obs = self._observe()
        observations = {a: obs for a in self.agents}
        rewards = {a: float(reward) for a in self.agents}
        terminations = {a: term for a in self.agents}
        truncations = {a: trunc for a in self.agents}
        infos = {a: dict(step_info) for a in self.agents}

        if term or trunc:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _apply_budget(self, action: np.ndarray) -> np.ndarray:
        if self.bandwidth_budget is None:
            return action
        allowed = self.bandwidth_budget * self.network.num_edges()
        total = float(action.sum())
        if total > allowed and total > 1e-9:
            action = action * (allowed / total)
        return action.astype(np.float32)

    def _observe(self) -> np.ndarray:
        util = np.clip(self.network.get_edge_utilization(), 0.0, 1.0)
        pending = np.clip(self._pending / max(self.network.link_capacity, 1e-9), 0.0, 1.0)
        occ = np.clip(self.network.get_node_occupancy() / max(self.network.node_buffer, 1e-9), 0.0, 1.0)
        return np.concatenate([util, pending, occ]).astype(np.float32)
