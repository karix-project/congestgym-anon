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
* Service is FRACTIONAL: a packet larger than the budget a link has left
  is partly transmitted and finishes the hop on a later step, so any
  positive allocation eventually moves traffic.
* Drops occur on admission overflow, downstream buffer overflow, or when
  no route exists.

Head-of-line blocking is intentional: a node's FIFO stalls behind a packet
whose next-hop link is exhausted (or still part-way through that packet). That is what creates the congestion
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
                 edge_prob: float = 0.35, seed: int | None = None,
                 bottlenecks: dict[int, float] | None = None):
        """
        bottlenecks : optional {node id: scale in (0, 1]}. A bottleneck node
            keeps only `scale` of its buffer, and every link touching it keeps
            only `scale` of its capacity (the smaller scale wins on a link
            between two bottleneck nodes). Models a degraded or under-
            provisioned node that traffic should be steered around.
        """
        if topology not in self.SUPPORTED_TOPOLOGIES:
            raise ValueError(f"Unknown topology '{topology}'. "
                             f"Choose from {self.SUPPORTED_TOPOLOGIES}")
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")
        if link_capacity <= 0 or node_buffer <= 0:
            raise ValueError("link_capacity and node_buffer must be positive")
        if topology == "grid" and int(round(np.sqrt(num_nodes))) ** 2 != num_nodes:
            raise ValueError(f"grid topology needs a perfect-square num_nodes "
                             f"(4, 9, 16, ...), got {num_nodes}")
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

        self.bottlenecks = dict(bottlenecks or {})
        for n, scale in self.bottlenecks.items():
            if n not in self.graph:
                raise ValueError(f"bottleneck node {n} is not in the topology")
            if not 0.0 < scale <= 1.0:
                raise ValueError(f"bottleneck scale must be in (0, 1], got {scale} for node {n}")
        scale = {n: self.bottlenecks.get(n, 1.0) for n in self.graph.nodes()}
        self._edge_capacity = np.array(
            [self.link_capacity * min(scale[u], scale[v]) for u, v in self._edges])
        self._node_buffer = {n: self.node_buffer * scale[n] for n in self.graph.nodes()}
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
        for i, (u, v) in enumerate(self._edges):
            self.graph[u][v]["capacity"] = float(self._edge_capacity[i])
            self.graph[u][v]["utilization"] = 0.0
        for n in self.graph.nodes():
            self.graph.nodes[n]["buffer_size"] = self._node_buffer[n]
            self.graph.nodes[n]["queue"] = 0.0

        # Initialize default uniform routing weights
        self.update_routing(np.ones(len(self._edges)))

    def update_routing(self, routing_weights: np.ndarray) -> None:
        """Update edge routing weights and recompute shortest paths (Phase 2 dynamic routing)."""
        for i, (u, v) in enumerate(self._edges):
            # A small epsilon prevents zero-weight cycles if action is 0
            self.graph[u][v]["routing_weight"] = float(routing_weights[i]) + 1e-4

        # Precompute all pairs shortest path for the current step using the learned link weights
        self._current_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight="routing_weight"))

    def path(self, src: int, dst: int) -> list[int]:
        """Dynamic shortest-path routing based on learned link weights."""
        if not hasattr(self, '_current_paths'):
            try:
                return nx.shortest_path(self.graph, src, dst)
            except nx.NetworkXNoPath:
                return []

        try:
            return self._current_paths[src][dst]
        except KeyError:
            return []

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
            if occ[pkt.src] + pkt.size > self._node_buffer[pkt.src]:
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
        budget = {self._key(u, v): self._edge_capacity[i] * alloc[i]
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

        # Downstream space is judged against occupancy at the START of the
        # step plus what has arrived since. Space freed by departures becomes
        # usable on the next step. This is independent of node iteration order
        # and guarantees a buffer never ends a step above `node_buffer`.
        start_occ = {n: self._occupancy(n) for n in self.graph.nodes()}
        arrived = {n: 0.0 for n in self.graph.nodes()}

        def occ(n: int) -> float:
            return start_occ[n] + arrived[n]

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
                need = pkt.size - pkt.sent
                if budget[key] + 1e-9 < need:
                    # Fractional service: spend what the link has left on this
                    # packet and finish the hop on a later step.
                    pkt.sent += budget[key]
                    self._served[ei] += budget[key]
                    budget[key] = 0.0
                    blocked = True
                    retained[node].append(pkt)
                    continue

                # Link can finish the hop -- check the downstream buffer next.
                budget[key] = max(budget[key] - need, 0.0)
                self._served[ei] += need
                pkt.sent = 0.0
                if pkt.hop_index + 1 == len(pkt.path) - 1:
                    pkt.hop_index += 1
                    delivered_units += pkt.size
                    delivered_count += 1
                    delay = float(step - pkt.birth_step + 1)
                    latencies.append((delay, pkt.delay_sensitivity, pkt.size))
                    if delay > pkt.deadline and not pkt.missed:
                        pkt.missed = True
                        deadline_misses += 1
                elif occ(nxt) + pkt.size > self._node_buffer[nxt]:
                    dropped_units += pkt.size          # downstream overflow
                else:
                    pkt.hop_index += 1
                    arrivals[nxt].append(pkt)
                    arrived[nxt] += pkt.size

        self.buffers = {n: retained[n] + arrivals[n] for n in self.graph.nodes()}

        # Packets still queued: record their age, and count a deadline miss
        # the first step a packet outlives its deadline while waiting. Without
        # this, traffic that never leaves a buffer is invisible to the delay
        # and deadline terms of the reward.
        queue_expired = 0
        queued_count = 0
        queued_units = 0.0
        queued_age_sum = 0.0
        queued_weighted_age_sum = 0.0
        for pkts in self.buffers.values():
            for pkt in pkts:
                age = float(step - pkt.birth_step + 1)
                queued_count += 1
                queued_units += pkt.size
                queued_age_sum += age
                queued_weighted_age_sum += age * pkt.delay_sensitivity * pkt.size
                if age > pkt.deadline and not pkt.missed:
                    pkt.missed = True
                    queue_expired += 1

        util = np.clip(self._served / np.maximum(self._edge_capacity, 1e-9), 0.0, 1.0)
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
            "queue_expired": queue_expired,
            "queued_count": queued_count,
            "queued_units": queued_units,
            "queued_age_sum": queued_age_sum,
            "queued_weighted_age_sum": queued_weighted_age_sum,
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

    def get_edge_capacities(self) -> np.ndarray:
        return self._edge_capacity.astype(np.float32)

    def get_node_buffers(self) -> np.ndarray:
        return np.array([self._node_buffer[n] for n in self.graph.nodes()],
                        dtype=np.float32)

    def backlog_units(self) -> float:
        return float(sum(self._occupancy(n) for n in self.graph.nodes()))
