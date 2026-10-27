"""Simulator options added in 0.4.0: topology seed (F3), random tie-breaks and
service order (F1), per-link queues (F2). See docs/CODE_REVIEW_FOR_PUBLICATION.md."""
import networkx as nx
import numpy as np
import pytest

from congestgym.core.network import NetworkModel
from congestgym.core.traffic import Packet, QoSClass
from congestgym.env import CongestGymEnv


def _pkt(src, dst, size=10.0):
    return Packet(src, dst, QoSClass.VIDEO, size, 1.5, 100, 0)


def _alloc(net, per_edge):
    """Allocation vector from {(u, v): weight}; every other link gets 0."""
    a = np.zeros(net.num_edges())
    for (u, v), w in per_edge.items():
        a[net._edge_index[net._key(u, v)]] = w
    return a


# ---------------------------------------------------------------- F3 topology seed
def test_random_topology_does_not_change_with_the_episode_seed():
    edges = {s: CongestGymEnv(topology="random", num_nodes=9, seed=s).network.edge_list()
             for s in range(8)}
    assert all(e == edges[0] for e in edges.values())


def test_topology_seed_selects_the_random_graph():
    graphs = {tuple(NetworkModel("random", 9, topology_seed=t).edge_list()) for t in range(8)}
    assert len(graphs) > 1
    a = NetworkModel("random", 9, topology_seed=3, seed=0).edge_list()
    b = NetworkModel("random", 9, topology_seed=3, seed=99).edge_list()
    assert a == b


def test_episode_seed_still_changes_traffic_on_the_random_topology():
    def offered(seed):
        env = CongestGymEnv(topology="random", num_nodes=9, max_steps=10, seed=seed)
        env.reset(seed=seed)
        a = np.ones(env.action_space.shape, dtype=np.float32)
        return [env.step(a)[4]["offered_units"] for _ in range(10)]
    assert offered(0) != offered(1)


# ---------------------------------------------------------------- F1 routing
@pytest.mark.parametrize("topology", ["grid", "random", "ring", "mesh", "star"])
@pytest.mark.parametrize("tie_break", ["random", "first"])
def test_paths_are_shortest_on_random_weights(topology, tie_break):
    net = NetworkModel(topology, 9, seed=0, tie_break=tie_break)
    rng = np.random.default_rng(1)
    for _ in range(5):
        w = rng.uniform(0, 1, net.num_edges())
        w[rng.integers(net.num_edges())] = 0.0           # a zero weight must not break anything
        net.update_routing(w)
        lengths = dict(nx.all_pairs_dijkstra_path_length(net.graph, weight="routing_weight"))
        for s in range(9):
            for d in range(9):
                p = net.path(s, d)
                if s == d:
                    assert p == []
                    continue
                assert p[0] == s and p[-1] == d and len(set(p)) == len(p)
                cost = sum(net.graph[u][v]["routing_weight"] for u, v in zip(p, p[1:]))
                assert cost == pytest.approx(lengths[s][d], rel=1e-9)


def test_equal_cost_paths_are_all_used_under_random_tie_break():
    net = NetworkModel("grid", 9, seed=0)                  # corner 0 to corner 8: 6 shortest paths
    seen = {tuple(net.path(0, 8)) for _ in range(600)}
    assert len(seen) == 6 and all(len(p) == 5 for p in seen)


def test_first_tie_break_is_deterministic():
    net = NetworkModel("grid", 9, seed=0, tie_break="first")
    assert len({tuple(net.path(0, 8)) for _ in range(50)}) == 1


def test_tie_breaks_are_reproducible_from_the_seed():
    def sample(net):
        return [tuple(net.path(0, 8)) for _ in range(30)]
    a, b = NetworkModel("grid", 9, seed=5), NetworkModel("grid", 9, seed=5)
    first = sample(a)
    assert first == sample(b)
    a.reset(seed=5)
    assert sample(a) == first


def _contested_link_winner(seed, service_order):
    """Nodes 0 and 1 each send a 10-unit packet across link (0, 1), which
    serves 10 units this step: only the node served first gets through."""
    net = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=1000.0, seed=seed,
                       service_order=service_order)
    net.admit([_pkt(0, 1), _pkt(1, 0)])
    out = net.forward(_alloc(net, {(0, 1): 10.0 / 30.0}), 0)
    assert out["delivered_count"] == 1
    return 0 if net.buffers[0] == [] else 1


def test_id_service_order_always_favours_the_lower_id():
    assert {_contested_link_winner(s, "id") for s in range(50)} == {0}


def test_random_service_order_shares_a_contested_link():
    wins = [_contested_link_winner(s, "random") for s in range(400)]
    assert 0.4 < np.mean(wins) < 0.6


def test_symmetric_corners_get_similar_service():
    """Regression guard for F1: with hop-count routing and uniform traffic,
    the four grid corners delivered 0.72 to 0.96 of their traffic (0.3.0)."""
    off, dlv = np.zeros(9), np.zeros(9)
    for s in range(6):
        env = CongestGymEnv(topology="grid", num_nodes=9, bandwidth_budget=0.8, seed=s)
        born = []
        gen = env.traffic.generate
        env.traffic.generate = lambda t, _g=gen: _record(born, _g, t)
        env.reset(seed=s)
        a = np.ones(env.action_space.shape, dtype=np.float32)
        for _ in range(200):
            env.step(a)
        for p in born:
            if 20 <= p.birth_step < 170:
                off[p.src] += p.size
                dlv[p.src] += p.size if (p.path and p.arrived) else 0.0
    corners = (dlv / off)[[0, 2, 6, 8]]
    assert corners.max() - corners.min() < 0.08, corners


def _record(store, gen, t):
    pkts = gen(t)
    store.extend(pkts)
    return pkts


# ---------------------------------------------------------------- F2 queueing
@pytest.mark.parametrize("queueing,delivered", [("per_link", 1), ("shared_fifo", 0)])
def test_stalled_link_blocks_other_links_only_under_shared_fifo(queueing, delivered):
    """Node 0 holds a packet for link (0, 1), which has no bandwidth, ahead of
    a packet for link (0, 3), which has plenty."""
    net = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=1000.0, seed=0,
                       queueing=queueing)
    net.admit([_pkt(0, 1), _pkt(0, 3)])
    out = net.forward(_alloc(net, {(0, 1): 0.0, (0, 3): 1.0}), 0)
    assert out["delivered_count"] == delivered
    assert net.buffers[0][0].dst == 1                     # the stalled packet keeps its place


def test_per_link_queues_keep_fifo_order_within_a_link():
    net = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=1000.0, seed=0)
    first, second = _pkt(0, 1, size=12.0), _pkt(0, 1, size=4.0)
    net.admit([first, second])
    out = net.forward(_alloc(net, {(0, 1): 10.0 / 30.0}), 0)
    # 10 units: the 12-unit head packet is part-sent; the small one must wait behind it
    assert out["delivered_count"] == 0 and net.buffers[0] == [first, second]


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("kw", [{"queueing": "voq"}, {"tie_break": "hash"}, {"service_order": "rr"}])
def test_invalid_model_options_rejected(kw):
    with pytest.raises(ValueError):
        CongestGymEnv(**kw)


def test_shipped_defaults():
    net = CongestGymEnv().network
    assert (net.queueing, net.tie_break, net.service_order, net.topology_seed) == \
        ("per_link", "random", "random", 0)


# ---------------------------------------------------------------- RNG streams (QA Q1)
def test_network_and_traffic_streams_are_independent():
    """Regression guard: both generators were seeded with default_rng(seed)
    and drew identical raw numbers (QA review of PR #5, finding Q1)."""
    env = CongestGymEnv(seed=7)
    for _ in range(2):                                   # after construction, then after reset
        a = env.network.rng.integers(0, 2**62, 5)
        b = env.traffic.rng.integers(0, 2**62, 5)
        assert not np.any(a == b)
        env.reset(seed=7)


def test_network_stream_is_reproducible_from_the_seed():
    a, b = NetworkModel("grid", 9, seed=11), NetworkModel("grid", 9, seed=11)
    assert a.rng.integers(0, 2**62, 5).tolist() == b.rng.integers(0, 2**62, 5).tolist()
    a.reset(seed=11)
    assert a.rng.integers(0, 2**62, 5).tolist() == NetworkModel("grid", 9, seed=11).rng.integers(0, 2**62, 5).tolist()


def test_random_policy_stream_is_independent_of_traffic():
    """Regression guard: the benchmark seeded the random policy with
    default_rng(seed), replaying the traffic's draws (QA R2-2)."""
    from congestgym.baselines import policy_rng
    env = CongestGymEnv(seed=7)
    env.reset(seed=7)
    a = policy_rng(7).integers(0, 2**62, 5)
    b = env.traffic.rng.integers(0, 2**62, 5)
    assert not np.any(a == b)
    assert policy_rng(7).integers(0, 2**62, 5).tolist() == a.tolist()
