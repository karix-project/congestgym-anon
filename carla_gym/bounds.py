"""
carla_gym/bounds.py

Fluid multi-commodity-flow upper bound on the fraction of offered traffic any
policy can deliver in a scenario. Needs SciPy (`pip install carla-gym[bounds]`).

The linear program, for the scenario's expected traffic matrix d (units per
step for each source-destination pair):

    maximise    sum of delivered rates x_st,   0 <= x_st <= d_st
    subject to  per destination t, flow is conserved at every node;
                for each link e = {u, v}: flow u->v plus flow v->u
                    <= capacity_e * a_e        (both directions share a link)
                0 <= a_e <= 1, and sum of a_e <= budget * E
                    (the env's bandwidth budget, counted in allocation weights)

Why it bounds the simulator: whatever a policy does step by step, its
time-averaged link allocations and flows satisfy these same linear
constraints, so its long-run delivered rate cannot exceed the optimum. It is
a fluid bound: packets, buffers, per-step service and burstiness are ignored,
so it is loose under bursty traces. Over a finite window, a policy can exceed
it slightly by delivering backlog carried in from before the window.

routing="free"      : flow may take any path (multipath allowed).
routing="shortest"  : flow may only use hop-count shortest paths, split
                      freely among them. The gap between the two is the most
                      that detours could ever add.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from carla_gym.core.trace_traffic import TraceTrafficGenerator
from carla_gym.core.traffic import TrafficGenerator


def expected_demand(traffic, num_nodes: int) -> np.ndarray:
    """Expected offered units per step for each (src, dst), as an N x N matrix."""
    d = np.zeros((num_nodes, num_nodes))
    if isinstance(traffic, TraceTrafficGenerator):
        for rows in traffic.by_step.values():
            for src, dst, vol in rows:
                d[int(src), int(dst)] += vol
        return d / traffic.n_steps
    if isinstance(traffic, TrafficGenerator):
        # Each packet picks an ordered pair uniformly; sizes are normal(m, 0.2 m)
        # clipped at 0.1, whose mean equals m to well below 1e-6 for every class.
        total = sum(p["arrival_rate"] * traffic.load_scale * p["packet_size"]
                    for p in traffic.qos_profiles.values())
        n = len(traffic.nodes)
        for s in traffic.nodes:
            for t in traffic.nodes:
                if s != t:
                    d[s, t] = total / (n * (n - 1))
        return d
    raise TypeError(f"no expected-demand model for {type(traffic).__name__}")


def flow_bound(network, demand: np.ndarray, budget: float | None,
               routing: str = "free") -> dict:
    """Solve the LP. Returns delivered_fraction, delivered and offered rates,
    and the optimal allocation per link (in the network's edge order)."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as e:  # pragma: no cover
        raise ImportError("flow_bound needs SciPy: pip install carla-gym[bounds]") from e
    if routing not in ("free", "shortest"):
        raise ValueError(f"routing must be 'free' or 'shortest', got {routing!r}")

    g = network.graph
    nodes = list(g.nodes())
    n = len(nodes)
    edges = network.edge_list()
    cap = network.get_edge_capacities().astype(float)
    E = len(edges)
    hops = dict(nx.all_pairs_shortest_path_length(g)) if routing == "shortest" else None

    # variable layout: [a_e (E)] [x_st (pairs)] [f_t,arc (per destination)]
    pairs = [(s, t) for s in nodes for t in nodes if s != t and demand[s, t] > 0]
    pair_idx = {p: E + i for i, p in enumerate(pairs)}
    arcs_of = {}
    col = E + len(pairs)
    for t in nodes:
        arcs = []
        for k, (u, v) in enumerate(edges):
            for a, b in ((u, v), (v, u)):
                if a == t:
                    continue                                  # nothing leaves its destination
                if hops is not None and hops[a][t] != hops[b][t] + 1:
                    continue                                  # not on a shortest path to t
                arcs.append((a, b, k, col))
                col += 1
        arcs_of[t] = arcs
    nvar = col

    ub_r, ub_c, ub_v, ub_b = [], [], [], []
    row = 0
    for k in range(E):                                        # link capacity
        ub_r.append(row); ub_c.append(k); ub_v.append(-cap[k])
        for t in nodes:
            for a, b, kk, c in arcs_of[t]:
                if kk == k:
                    ub_r.append(row); ub_c.append(c); ub_v.append(1.0)
        ub_b.append(0.0); row += 1
    if budget is not None:                                    # bandwidth budget
        for k in range(E):
            ub_r.append(row); ub_c.append(k); ub_v.append(1.0)
        ub_b.append(budget * E); row += 1
    A_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(row, nvar)).tocsr()

    eq_r, eq_c, eq_v = [], [], []
    row = 0
    for t in nodes:                                           # conservation per destination
        for v in nodes:
            if v == t:
                continue
            for a, b, _, c in arcs_of[t]:
                if a == v:
                    eq_r.append(row); eq_c.append(c); eq_v.append(1.0)
                if b == v:
                    eq_r.append(row); eq_c.append(c); eq_v.append(-1.0)
            if (v, t) in pair_idx:
                eq_r.append(row); eq_c.append(pair_idx[(v, t)]); eq_v.append(-1.0)
            row += 1
    A_eq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(row, nvar)).tocsr()

    bounds = [(0.0, 1.0)] * E + [(0.0, float(demand[s, t])) for s, t in pairs] + [(0.0, None)] * (nvar - E - len(pairs))
    c = np.zeros(nvar)
    for p, i in pair_idx.items():
        c[i] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.array(ub_b), A_eq=A_eq, b_eq=np.zeros(row),
                  bounds=bounds, method="highs")
    if res.status != 0:                                       # pragma: no cover
        raise RuntimeError(f"LP did not solve: {res.message}")
    offered = float(demand.sum())
    delivered = float(-res.fun)
    return {"delivered_fraction": delivered / offered if offered > 0 else 0.0,
            "delivered": delivered, "offered": offered,
            "allocation": res.x[:E].copy()}


def scenario_bound(env, routing: str = "free") -> dict:
    """flow_bound for a CarlaGymEnv as built (its network, traffic and budget)."""
    base = env.unwrapped
    d = expected_demand(base.traffic, base.num_nodes)
    return flow_bound(base.network, d, base.bandwidth_budget, routing)
