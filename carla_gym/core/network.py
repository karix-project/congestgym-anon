"""
carla_gym/core/network.py

Store-and-forward network simulator on a NetworkX graph.

Model
-----
* Each NODE has a finite buffer (units). Packets physically sit in node
  buffers -- this is what makes node occupancy a real observable rather
  than a decorative zero.
* Each LINK has a service rate (units/step). The RL action scales that
  rate, so bandwidth allocation has a direct, measurable effect.
* Packets advance AT MOST ONE HOP PER STEP, subject to link budget and
  downstream buffer space. Latency is therefore *measured in steps*,
  not proxied by utilization.
* Drops occur on admission overflow, downstream buffer overflow, or when
  no route exists.

Head-of-line blocking is intentional: a node's FIFO stalls behind a packet
whose next-hop link is exhausted. That is what creates the congestion
dynamics an RL agent has something to learn about.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from carla_gym.core.traffic import Packet


class NetworkModel:
    SUPPORTED_TOPOLOGIES = ("grid", "random", "star", "mesh", "ring")

    def __init__(self, topology: str = "grid", num_nodes: int = 9,
                 link_capacity: float = 30.0, node_buffer: float = 120.0,
                 edge_prob: float = 0.35, seed: int | None = None):
        if topology not in self.SUPPORTED_TOPOLOGIES:
            raise ValueError(f"Unknown topology '{topology}'. "
                             f"Choose from {self.SUPPORTED_TOPOLOGIES}")
        self.topology = topology
        self.num_nodes = num_nodes
        self.link_capacity = link_capacity
        self.node_buffer = node_buffer
        self.edge_prob = edge_prob
        self.rng = np.random.default_rng(seed)

        self.graph = self._build_topology()
        self.num_nodes = self.graph.number_of_nodes()
        self._edges = list(self.graph.edges())
        self._edge_index = {self._key(u, v): i for i, (u, v) in enumerate(self._edges)}
        self._path_cache: dict[tuple[int, int], list[int]] = {}
        self.reset()

    # ------------------------------------------------------------------ #
    # Topology
    # ------------------------------------------------------------------ #
    def _build_topology(self) -> nx.Graph:
        t = self.topology
        if t == "grid":
            side = int(round(np.sqrt(self.num_nodes)))
            g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
        elif t == "random":
            g = nx.erdos_renyi_graph(self.num_nodes, self.edge_prob,
                                     seed=int(self.rng.integers(1e9)))
            comps = list(nx.connected_components(g))
            for a, b in zip(comps, comps[1:]):
                g.add_edge(next(iter(a)), next(iter(b)))
        elif t == "star":
            g = nx.star_graph(self.num_nodes - 1)
        elif t == "mesh":
            g = nx.complete_graph(self.num_nodes)
        elif t == "ring":
            g = nx.cycle_graph(self.num_nodes)
        else:  # pragma: no cover
            raise ValueError(t)
        return g

    @staticmethod
    def _key(u: int, v: int) -> tuple[int, int]:
        """Undirected links share one budget; normalize endpoint order."""
        return (u, v) if u <= v else (v, u)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.buffers: dict[int, list[Packet]] = {n: [] for n in self.graph.nodes()}
        self._served = np.zeros(len(self._edges), dtype=np.float64)
        for u, v in self._edges:
            self.graph[u][v]["capacity"] = self.link_capacity
            self.graph[u][v]["utilization"] = 0.0
        for n in self.graph.nodes():
            self.graph.nodes[n]["buffer_size"] = self.node_buffer
            self.graph.nodes[n]["queue"] = 0.0

    def path(self, src: int, dst: int) -> list[int]:
        """Rigid shortest-path routing (Phase 1). Cached: topology is static."""
        key = (src, dst)
        if key not in self._path_cache:
            try:
                self._path_cache[key] = nx.shortest_path(self.graph, src, dst)
            except nx.NetworkXNoPath:
                self._path_cache[key] = []
        return self._path_cache[key]

    def _occupancy(self, node: int) -> float:
        return sum(p.size for p in self.buffers[node])

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def admit(self, packets: list[Packet]) -> tuple[float, float]:
        """Try to place new packets in their source node's buffer.
        Returns (offered_units, dropped_units)."""
        offered = dropped = 0.0
        occ = {n: self._occupancy(n) for n in self.graph.nodes()}
        for pkt in packets:
            offered += pkt.size
            route = self.path(pkt.src, pkt.dst)
            if len(route) < 2:
                dropped += pkt.size
                continue
            if occ[pkt.src] + pkt.size > self.node_buffer:
                dropped += pkt.size          # admission overflow
                continue
            pkt.path = route
            pkt.hop_index = 0
            self.buffers[pkt.src].append(pkt)
            occ[pkt.src] += pkt.size
        return offered, dropped

    def forward(self, allocation: np.ndarray, step: int) -> dict:
        """Advance every packet at most one hop, honouring per-link budget
        and downstream buffer space. `allocation` is one weight per edge in
        the canonical edge order (see edge_list())."""
        alloc = np.clip(np.asarray(allocation, dtype=np.float64), 0.0, 1.0)
        budget = {self._key(u, v): self.link_capacity * alloc[i]
                  for i, (u, v) in enumerate(self._edges)}

        retained: dict[int, list[Packet]] = {n: [] for n in self.graph.nodes()}
        arrivals: dict[int, list[Packet]] = {n: [] for n in self.graph.nodes()}
        pending_units = np.zeros(len(self._edges))
        self._served[:] = 0.0

        delivered_units = 0.0
        dropped_units = 0.0
        latencies: list[tuple[float, float, float]] = []  # (delay, sensitivity, size)
        deadline_misses = 0
        delivered_count = 0

        def occ(n: int) -> float:
            return (sum(p.size for p in retained[n])
                    + sum(p.size for p in arrivals[n]))

        for node in self.graph.nodes():
            blocked = False
            for pkt in self.buffers[node]:
                nxt = pkt.next_hop
                if nxt is None:                       # defensive: already home
                    continue
                ei = self._edge_index[self._key(node, nxt)]
                pending_units[ei] += pkt.size

                if blocked:                            # head-of-line blocking
                    retained[node].append(pkt)
                    continue

                key = self._key(node, nxt)
                if budget[key] + 1e-9 < pkt.size:
                    blocked = True
                    retained[node].append(pkt)
                    continue

                # Link can serve it -- check the downstream buffer next.
                if pkt.hop_index + 1 == len(pkt.path) - 1:
                    budget[key] -= pkt.size
                    self._served[ei] += pkt.size
                    pkt.hop_index += 1
                    delivered_units += pkt.size
                    delivered_count += 1
                    delay = float(step - pkt.birth_step + 1)
                    latencies.append((delay, pkt.delay_sensitivity, pkt.size))
                    if delay > pkt.deadline:
                        deadline_misses += 1
                elif occ(nxt) + pkt.size > self.node_buffer:
                    dropped_units += pkt.size          # downstream overflow
                    budget[key] -= pkt.size
                    self._served[ei] += pkt.size
                else:
                    budget[key] -= pkt.size
                    self._served[ei] += pkt.size
                    pkt.hop_index += 1
                    arrivals[nxt].append(pkt)

        self.buffers = {n: retained[n] + arrivals[n] for n in self.graph.nodes()}

        caps = np.array([self.graph[u][v]["capacity"] for u, v in self._edges])
        util = np.clip(self._served / np.maximum(caps, 1e-9), 0.0, 1.0)
        for i, (u, v) in enumerate(self._edges):
            self.graph[u][v]["utilization"] = float(util[i])
        for n in self.graph.nodes():
            self.graph.nodes[n]["queue"] = self._occupancy(n)

        return {
            "delivered_units": delivered_units,
            "dropped_units": dropped_units,
            "latencies": latencies,
            "delivered_count": delivered_count,
            "deadline_misses": deadline_misses,
            "pending_units": pending_units,
            "utilization": util,
        }

    # ------------------------------------------------------------------ #
    # Observation helpers
    # ------------------------------------------------------------------ #
    def edge_list(self) -> list[tuple[int, int]]:
        return list(self._edges)

    def num_edges(self) -> int:
        return len(self._edges)

    def get_edge_utilization(self) -> np.ndarray:
        return np.array([self.graph[u][v]["utilization"] for u, v in self._edges],
                        dtype=np.float32)

    def get_node_occupancy(self) -> np.ndarray:
        return np.array([self._occupancy(n) for n in self.graph.nodes()],
                        dtype=np.float32)

    def backlog_units(self) -> float:
        return float(sum(self._occupancy(n) for n in self.graph.nodes()))
