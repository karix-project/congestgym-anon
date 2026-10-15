"""
carla_gym/utils/reward.py

    R = Throughput - k * Delay - c * PacketLoss - m * DeadlineMiss

All four terms are normalized to a comparable [0, 1] scale before the
coefficients are applied. This matters: with raw units, throughput is
O(100) while loss ratio is O(0.01), so the QoS penalties vanish into
numerical noise and the agent degenerates to "drain everything". Keeping
the terms commensurate is what makes k and c meaningful knobs.

Term definitions (computed per step in core/step.py):
  * Throughput   = delivered / offered units this step, clipped at 1. While a
                   backlog drains the raw ratio exceeds 1; the clip keeps the
                   term bounded, and the delay term is what distinguishes a
                   draining backlog from steady state.
  * Delay        = sensitivity-weighted mean age of traffic in the system
                   (end-to-end delay of packets delivered this step, current
                   age of packets still queued), divided by `delay_reference`
                   and clipped at 1. It is QoS-weighted, not plain latency.
  * PacketLoss   = dropped / offered units this step, clipped at 1.
  * DeadlineMiss = packets that missed their deadline this step (delivered
                   late or expired while queued) over packets resolved this
                   step (delivered + expired). A miss is counted once per
                   packet. A packet that expires in a queue and is delivered
                   on a later step appears in the denominator on both steps.
                   Expiry is recorded at the end of the first step in which
                   the packet's age exceeds its deadline.

With no offered traffic in a step (for example after a non-looping trace has
ended) Throughput and PacketLoss are 0, so packets drained in that step earn
no throughput credit.

Pure function of the metrics handed in -- no simulator state, so it can
be unit-tested and swapped independently of env.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class RewardBreakdown:
    throughput: float       # normalized goodput ratio, [0, 1]
    delay: float            # normalized weighted age of traffic, [0, 1]
    packet_loss: float      # loss ratio, [0, 1]
    deadline_miss: float    # deadline miss ratio, [0, 1]
    reward: float

    def as_dict(self) -> dict:
        return asdict(self)


class RewardCalculator:
    def __init__(self, k: float = 0.5, c: float = 1.0, m: float = 0.5,
                 delay_reference: float = 10.0):
        """
        k               : delay penalty coefficient
        c               : packet-loss penalty coefficient
        m               : deadline miss penalty coefficient
        delay_reference : latency (in steps) treated as "fully bad"; measured
                          delay is divided by this and clipped to 1.
        """
        self.k = k
        self.c = c
        self.m = m
        if min(k, c, m) < 0:
            raise ValueError(f"reward coefficients k, c, m must be >= 0, got {k}, {c}, {m}")
        if delay_reference <= 0:
            raise ValueError(f"delay_reference must be > 0, got {delay_reference}")
        self.delay_reference = delay_reference

    def normalize(self, delivered_units: float, offered_units: float,
                  weighted_delay: float, dropped_units: float,
                  deadline_misses: float, resolved_count: float) -> tuple[float, float, float, float]:
        tput = (delivered_units / offered_units) if offered_units > 0 else 0.0
        delay = min(weighted_delay / self.delay_reference, 1.0) if weighted_delay > 0 else 0.0
        loss = (dropped_units / offered_units) if offered_units > 0 else 0.0
        miss = (deadline_misses / resolved_count) if resolved_count > 0 else 0.0
        return min(tput, 1.0), delay, min(loss, 1.0), min(miss, 1.0)

    def compute(self, throughput: float, delay: float, packet_loss: float, deadline_miss: float) -> float:
        return throughput - self.k * delay - self.c * packet_loss - self.m * deadline_miss

    def compute_breakdown(self, throughput: float, delay: float,
                          packet_loss: float, deadline_miss: float) -> RewardBreakdown:
        return RewardBreakdown(throughput, delay, packet_loss, deadline_miss,
                               self.compute(throughput, delay, packet_loss, deadline_miss))
