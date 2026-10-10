"""
carla_gym/core/traffic.py

Volatile data-load generator: emits demand only. It knows the node IDs and
nothing else about the network -- it never inspects queues or topology, so
it can be swapped for a trace-driven or bursty generator without touching
the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class QoSClass(Enum):
    VIDEO = "video"
    VOICE = "voice"
    GAMING = "gaming"
    BEST_EFFORT = "best_effort"


# arrival_rate: Poisson lambda (flows/step). packet_size: mean units.
# delay_sensitivity: weight applied to measured latency in the reward.
# deadline: steps after which the packet is useless (used for QoS violation stats).
DEFAULT_QOS_PROFILES = {
    QoSClass.VOICE:       {"arrival_rate": 6.0, "packet_size": 2.0,  "delay_sensitivity": 2.0, "deadline": 4},
    QoSClass.GAMING:      {"arrival_rate": 3.0, "packet_size": 1.5,  "delay_sensitivity": 2.5, "deadline": 3},
    QoSClass.VIDEO:       {"arrival_rate": 4.0, "packet_size": 12.0, "delay_sensitivity": 1.5, "deadline": 8},
    QoSClass.BEST_EFFORT: {"arrival_rate": 5.0, "packet_size": 8.0,  "delay_sensitivity": 0.5, "deadline": 20},
}


@dataclass
class Packet:
    src: int
    dst: int
    qos_class: QoSClass
    size: float
    delay_sensitivity: float
    deadline: int
    birth_step: int
    # Routing state, populated by the network on admission. Kept on the packet
    # (not in a side table) so store-and-forward progress is self-contained.
    path: list[int] = field(default_factory=list)
    hop_index: int = 0
    # Units already transmitted over the current hop. A packet larger than a
    # link's per-step budget crosses it over several steps instead of stalling.
    sent: float = 0.0
    # Set once the packet's age first exceeds its deadline (queued or delivered),
    # so a deadline miss is counted exactly once.
    missed: bool = False

    @property
    def next_hop(self) -> int | None:
        if self.hop_index + 1 < len(self.path):
            return self.path[self.hop_index + 1]
        return None

    @property
    def arrived(self) -> bool:
        return self.hop_index >= len(self.path) - 1


class TrafficGenerator:
    def __init__(self, nodes: list[int], qos_profiles: dict | None = None,
                 load_scale: float = 1.0, seed: int | None = None):
        if len(nodes) < 2:
            raise ValueError("Need >=2 nodes to generate src/dst pairs")
        self.nodes = list(nodes)
        self.qos_profiles = qos_profiles or DEFAULT_QOS_PROFILES
        self.load_scale = load_scale
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def generate(self, step: int) -> list[Packet]:
        packets: list[Packet] = []
        for qos, prof in self.qos_profiles.items():
            n = self.rng.poisson(prof["arrival_rate"] * self.load_scale)
            for _ in range(n):
                src, dst = self.rng.choice(self.nodes, size=2, replace=False)
                size = float(max(self.rng.normal(prof["packet_size"],
                                                 prof["packet_size"] * 0.2), 0.1))
                packets.append(Packet(
                    src=int(src), dst=int(dst), qos_class=qos, size=size,
                    delay_sensitivity=prof["delay_sensitivity"],
                    deadline=prof["deadline"], birth_step=step,
                ))
        return packets
