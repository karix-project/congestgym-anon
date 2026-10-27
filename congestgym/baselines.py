"""
congestgym/baselines.py

Rule-based reference policies. Each maps (env, obs) -> action in the env's
native [0, 1] action space: E bandwidth weights followed by E routing weights.

`rng` is optional and used only by the random policy, so every policy can be
called as fn(env, obs) or fn(env, obs, rng).

Bandwidth-only heuristics keep routing weights uniform, which makes the
simulator route on hop-count shortest paths. The routing-aware heuristics
are the traffic-engineering baselines: they raise the routing weight of
congested links so new packets are steered around them, in the manner of
load-aware (adaptive-cost) shortest-path routing.
"""

from __future__ import annotations

import numpy as np


_POLICY_STREAM_TAG = 0x706F6C


def policy_rng(seed: int | None) -> np.random.Generator:
    """Generator for the random policy in an episode seeded with `seed`.

    The env seeds its traffic with default_rng(seed); a policy generator seeded
    the same way would replay the traffic's own random numbers as actions.
    The tag gives an independent stream that is still a function of the seed."""
    return np.random.default_rng(None if seed is None else [int(seed), _POLICY_STREAM_TAG])


def _split_obs(env, obs):
    """(link utilization, link pending demand, node occupancy), each in [0, 1]."""
    e = env.num_edges
    return obs[:e], obs[e:2 * e], obs[2 * e:]


# Tunable constants of the heuristics (docs/NEXT_EXPERIMENTS.md, E1). The
# defaults are the values every published result was produced with.
DEFAULT_FLOOR = 0.05          # smallest bandwidth weight queue-proportional gives a link
DEFAULT_BASE = 0.1            # routing cost of an idle link
DEFAULT_SLOPE = 0.9           # added routing cost of a fully congested link
SIGNALS = ("max", "pending")  # congestion = max(pending, endpoint occupancy) or pending only


def _queue_proportional_bandwidth(env, obs, floor: float = DEFAULT_FLOOR) -> np.ndarray:
    _, pending, _ = _split_obs(env, obs)
    if pending.max() <= 1e-9:
        return np.full(env.num_edges, 0.5)
    return np.clip(pending / pending.max(), floor, 1.0)


def _congestion_routing_weights(env, obs, base: float = DEFAULT_BASE, slope: float = DEFAULT_SLOPE,
                                signal: str = "max") -> np.ndarray:
    """Link cost = base + slope * congestion. Congestion is the larger of the
    link's pending demand and the mean buffer occupancy of its two endpoints
    (signal="max"), or the pending demand alone (signal="pending")."""
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")
    _, pending, occupancy = _split_obs(env, obs)
    if signal == "pending":
        return base + slope * pending
    index = {n: i for i, n in enumerate(env.network.graph.nodes())}
    endpoint = np.array([(occupancy[index[u]] + occupancy[index[v]]) / 2.0
                         for u, v in env.network.edge_list()])
    return base + slope * np.maximum(pending, endpoint)


# ------------------------------------------------------------------ #
# Bandwidth-only heuristics (hop-count shortest-path routing)
# ------------------------------------------------------------------ #
def policy_random(env, obs, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(0.0, 1.0, size=2 * env.num_edges)


def policy_max(env, obs, rng=None):
    """Shortest-path routing with bandwidth split equally over all links."""
    return np.ones(2 * env.num_edges)


def policy_min(env, obs, rng=None):
    return np.full(2 * env.num_edges, 0.05)


def policy_queue_proportional(env, obs, rng=None, *, floor=DEFAULT_FLOOR):
    """Bandwidth in proportion to each link's pending demand; uniform routing."""
    return np.concatenate([_queue_proportional_bandwidth(env, obs, floor),
                           np.ones(env.num_edges)])


# ------------------------------------------------------------------ #
# Routing-aware traffic-engineering heuristics
# ------------------------------------------------------------------ #
def policy_load_aware_routing(env, obs, rng=None, *, base=DEFAULT_BASE, slope=DEFAULT_SLOPE,
                              signal="max"):
    """Equal bandwidth split; routes new packets around congested links."""
    return np.concatenate([np.ones(env.num_edges),
                           _congestion_routing_weights(env, obs, base, slope, signal)])


def policy_congestion_aware(env, obs, rng=None, *, floor=DEFAULT_FLOOR, base=DEFAULT_BASE,
                            slope=DEFAULT_SLOPE, signal="max"):
    """Queue-proportional bandwidth combined with load-aware routing."""
    return np.concatenate([_queue_proportional_bandwidth(env, obs, floor),
                           _congestion_routing_weights(env, obs, base, slope, signal)])


POLICIES = {
    "random": policy_random,
    "min_bandwidth": policy_min,
    "max_bandwidth": policy_max,
    "queue_proportional": policy_queue_proportional,
    "load_aware_routing": policy_load_aware_routing,
    "congestion_aware": policy_congestion_aware,
}


def make_policy(name: str, **params):
    """A reference policy with some constants changed, callable as fn(env, obs, rng).
    make_policy("congestion_aware", floor=0.1, base=0.0, slope=1.0)"""
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    fn = POLICIES[name]

    def policy(env, obs, rng=None):
        return fn(env, obs, rng, **params)
    policy.__name__ = f"{name}({', '.join(f'{k}={v}' for k, v in params.items())})"
    return policy
