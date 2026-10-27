"""
congestgym/core/step.py

One simulation step, shared by the single-agent and multi-agent envs so the
two cannot drift apart: route, generate, admit, forward, then turn the raw
counters into the normalized reward terms and the `info` dict.
"""

from __future__ import annotations

import numpy as np


def advance(network, traffic, reward_calc, bw_action: np.ndarray,
            routing_action: np.ndarray, step: int) -> tuple[float, dict, np.ndarray]:
    """Returns (reward, info, pending_units_per_link)."""
    network.update_routing(routing_action)

    packets = traffic.generate(step)
    offered, admit_dropped = network.admit(packets)
    stats = network.forward(bw_action, step)

    dropped = admit_dropped + stats["dropped_units"]
    delivered = stats["delivered_units"]

    # Delay term: sensitivity-weighted age of traffic in the system. Delivered
    # packets contribute their end-to-end delay; packets still queued
    # contribute their current age, so traffic stuck in a buffer is penalised
    # while it waits, not only if it is eventually delivered.
    lat = stats["latencies"]
    num = sum(d * s * sz for d, s, sz in lat) + stats["queued_weighted_age_sum"]
    den = sum(sz for _, _, sz in lat) + stats["queued_units"]
    weighted_delay = num / den if den > 0 else 0.0
    raw_delay = sum(d for d, _, _ in lat) / len(lat) if lat else 0.0
    queue_age = (stats["queued_age_sum"] / stats["queued_count"]
                 if stats["queued_count"] else 0.0)

    # Deadline term: packets that missed their deadline this step (delivered
    # late, or expired while queued) over all packets resolved this step.
    misses = stats["deadline_misses"] + stats["queue_expired"]
    resolved = stats["delivered_count"] + stats["queue_expired"]

    t_n, d_n, l_n, m_n = reward_calc.normalize(delivered, offered, weighted_delay,
                                               dropped, misses, resolved)
    reward = reward_calc.compute(t_n, d_n, l_n, m_n)

    info = {
        "reward_throughput": t_n,
        "reward_delay": d_n,
        "reward_loss": l_n,
        "reward_deadline_miss": m_n,
        "delivered_units": delivered,
        "offered_units": offered,
        "dropped_units": dropped,
        "mean_latency_steps": raw_delay,
        "mean_queue_age_steps": queue_age,
        "weighted_latency_steps": weighted_delay,
        "delivered_count": stats["delivered_count"],
        "deadline_misses": misses,
        "backlog_units": network.backlog_units(),
        "mean_link_utilization": float(stats["utilization"].mean()),
    }
    return float(reward), info, stats["pending_units"].astype(np.float32)
