"""carla_gym/utils/qos.py -- episode-level QoS figures from per-step `info` dicts.

The reward terms are per-step ratios (delivered / offered in that step,
clipped at 1). Averaging them over an episode does NOT give the fraction of
traffic delivered: a step that drains a backlog is capped at 1 and a quiet
step weighs as much as a busy one. These helpers aggregate the raw counters
instead, which is what "throughput" and "loss" mean in a results table.
"""

from __future__ import annotations


class EpisodeQoS:
    """Accumulates raw counters over the steps it is given."""

    def __init__(self) -> None:
        self.offered = self.delivered = self.dropped = 0.0
        self.delay_sum = 0.0
        self.delivered_count = 0
        self.deadline_misses = 0

    def add(self, info: dict) -> None:
        self.offered += info["offered_units"]
        self.delivered += info["delivered_units"]
        self.dropped += info["dropped_units"]
        self.delay_sum += info["mean_latency_steps"] * info["delivered_count"]
        self.delivered_count += info["delivered_count"]
        self.deadline_misses += info["deadline_misses"]

    def summary(self) -> dict:
        """delivered_fraction / dropped_fraction: units over units offered in
        the same window (the remainder is the change in backlog).
        latency_delivered: mean end-to-end delay of delivered packets, steps.
        deadline_miss_per_delivered: packets that missed their deadline
        (delivered late or expired in a queue) per delivered packet."""
        off = max(self.offered, 1e-12)
        cnt = max(self.delivered_count, 1)
        return {
            "delivered_fraction": self.delivered / off,
            "dropped_fraction": self.dropped / off,
            "latency_delivered": self.delay_sum / cnt,
            "deadline_miss_per_delivered": self.deadline_misses / cnt,
        }
