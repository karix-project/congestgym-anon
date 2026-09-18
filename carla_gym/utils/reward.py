"""
carla_gym/utils/reward.py

    R = Throughput - k * Delay - c * PacketLoss

All three terms are normalized to a comparable [0, 1] scale before the
coefficients are applied. This matters: with raw units, throughput is
O(100) while loss ratio is O(0.01), so the QoS penalties vanish into
numerical noise and the agent degenerates to "drain everything". Keeping
the terms commensurate is what makes k and c meaningful knobs.

Pure function of the metrics handed in -- no simulator state, so it can
be unit-tested and swapped independently of env.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class RewardBreakdown:
    throughput: float       # normalized goodput ratio, [0, 1]
    delay: float            # normalized mean latency, [0, 1]
    packet_loss: float      # loss ratio, [0, 1]
    reward: float

    def as_dict(self) -> dict:
        return asdict(self)


class RewardCalculator:
    def __init__(self, k: float = 0.5, c: float = 1.0,
                 delay_reference: float = 10.0):
        """
        k               : delay penalty coefficient
        c               : packet-loss penalty coefficient
        delay_reference : latency (in steps) treated as "fully bad"; measured
                          delay is divided by this and clipped to 1.
        """
        self.k = k
        self.c = c
        self.delay_reference = delay_reference

    def normalize(self, delivered_units: float, offered_units: float,
                  weighted_delay: float, dropped_units: float) -> tuple[float, float, float]:
        tput = (delivered_units / offered_units) if offered_units > 0 else 0.0
        delay = min(weighted_delay / self.delay_reference, 1.0) if weighted_delay > 0 else 0.0
        loss = (dropped_units / offered_units) if offered_units > 0 else 0.0
        return min(tput, 1.0), delay, min(loss, 1.0)

    def compute(self, throughput: float, delay: float, packet_loss: float) -> float:
        return throughput - self.k * delay - self.c * packet_loss

    def compute_breakdown(self, throughput: float, delay: float,
                          packet_loss: float) -> RewardBreakdown:
        return RewardBreakdown(throughput, delay, packet_loss,
                               self.compute(throughput, delay, packet_loss))
