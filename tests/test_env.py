"""
tests/test_env.py

Sprint 1-4 validation. Beyond API conformance these assert the simulator
*invariants* -- conservation of packet units, live node observables, and
monotone response to bandwidth -- because those are the properties that
silently broke in earlier iterations.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from carla_gym.core.network import NetworkModel
from carla_gym.core.traffic import TrafficGenerator
from carla_gym.env import CarlaGymEnv
from carla_gym.utils.reward import RewardCalculator


# ---------------------------------------------------------------- Sprint 1
def test_gymnasium_api_conformance():
    env = CarlaGymEnv(max_steps=20, seed=0)
    check_env(env.unwrapped, skip_render_check=True)


def test_reset_returns_valid_observation():
    env = CarlaGymEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_reset_is_deterministic_given_seed():
    a = CarlaGymEnv(seed=0); a.reset(seed=7)
    b = CarlaGymEnv(seed=0); b.reset(seed=7)
    act = np.full(2 * a.num_edges, 0.5, dtype=np.float32)
    for _ in range(10):
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
    assert np.allclose(oa, ob) and ra == pytest.approx(rb)


# ---------------------------------------------------------------- Sprint 2
@pytest.mark.parametrize("topology", ["grid", "random", "star", "mesh", "ring"])
def test_all_topologies_build_and_step(topology):
    env = CarlaGymEnv(topology=topology, num_nodes=9, max_steps=5, seed=0)
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs) and np.isfinite(reward)


def test_unknown_topology_rejected():
    with pytest.raises(ValueError):
        NetworkModel(topology="hypercube")


def test_packet_units_are_conserved():
    """Every offered unit must end up delivered, dropped, or still in a
    buffer. A leak here means throughput/loss stats are lying."""
    env = CarlaGymEnv(max_steps=60, seed=3)
    env.reset(seed=3)
    offered = delivered = dropped = 0.0
    rng = np.random.default_rng(3)
    for _ in range(60):
        _, _, _, _, info = env.step(rng.uniform(0, 1, 2 * env.num_edges))
        offered += info["offered_units"]
        delivered += info["delivered_units"]
        dropped += info["dropped_units"]
    backlog = env.network.backlog_units()
    assert offered == pytest.approx(delivered + dropped + backlog, rel=1e-6)


def test_node_buffers_never_exceed_capacity():
    env = CarlaGymEnv(max_steps=80, seed=4)
    env.reset(seed=4)
    for _ in range(80):
        env.step(np.zeros(2 * env.num_edges, dtype=np.float32))  # worst case
        occ = env.network.get_node_occupancy()
        assert np.all(occ <= env.network.node_buffer + 1e-6)


# ---------------------------------------------------------------- Sprint 3
def test_node_occupancy_is_a_live_observable():
    """Regression guard: node queues were previously always zero, silently
    wasting a third of the observation vector."""
    env = CarlaGymEnv(max_steps=40, seed=5)
    obs, _ = env.reset(seed=5)
    for _ in range(40):
        obs, *_ = env.step(np.full(2 * env.num_edges, 0.2, dtype=np.float32))
    node_slice = obs[-env.num_nodes:]
    assert node_slice.max() > 0.0, "node occupancy stuck at zero"


def test_reward_terms_are_commensurate():
    """Regression guard: raw-unit throughput once dwarfed the QoS penalties
    (~100 vs ~0.002), making k and c meaningless."""
    env = CarlaGymEnv(max_steps=40, seed=6)
    env.reset(seed=6)
    rng = np.random.default_rng(6)
    t, d, l = [], [], []
    for _ in range(40):
        _, _, _, _, info = env.step(rng.uniform(0, 1, 2 * env.num_edges))
        t.append(info["reward_throughput"]); d.append(info["reward_delay"])
        l.append(info["reward_loss"])
    for series in (t, d, l):
        assert 0.0 <= min(series) and max(series) <= 1.0


def test_reward_penalties_reduce_reward():
    rc = RewardCalculator(k=0.5, c=1.0, m=0.5)
    assert rc.compute(1.0, 0.0, 0.0, 0.0) > rc.compute(1.0, 0.5, 0.0, 0.0)
    assert rc.compute(1.0, 0.0, 0.0, 0.0) > rc.compute(1.0, 0.0, 0.5, 0.0)
    assert rc.compute(1.0, 0.0, 0.0, 0.0) > rc.compute(1.0, 0.0, 0.0, 0.5)


def test_observation_always_within_bounds_under_stress():
    env = CarlaGymEnv(load_scale=4.0, max_steps=60, seed=7)
    obs, _ = env.reset(seed=7)
    for _ in range(60):
        obs, *_ = env.step(np.zeros(2 * env.num_edges, dtype=np.float32))
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------- Sprint 4
def test_more_bandwidth_yields_no_worse_throughput():
    """The central premise of Sprint 4: the action must actually matter."""
    def run(weight):
        env = CarlaGymEnv(max_steps=60, seed=8)
        env.reset(seed=8)
        tput = 0.0
        for _ in range(60):
            _, _, _, _, info = env.step(np.full(2 * env.num_edges, weight, dtype=np.float32))
            tput += info["delivered_units"]
        return tput

    assert run(1.0) > run(0.5) > run(0.05)


def test_starved_network_builds_backlog():
    env = CarlaGymEnv(max_steps=50, seed=9)
    env.reset(seed=9)
    for _ in range(50):
        env.step(np.zeros(2 * env.num_edges, dtype=np.float32))
    assert env.network.backlog_units() > 0.0


def test_bandwidth_budget_is_enforced():
    env = CarlaGymEnv(bandwidth_budget=0.5, seed=10)
    env.reset(seed=10)
    capped = env._apply_budget(np.ones(env.num_edges, dtype=np.float32))
    assert capped.sum() <= 0.5 * env.num_edges + 1e-6


def test_traffic_generator_respects_seed():
    g1 = TrafficGenerator(nodes=list(range(9)), seed=11)
    g2 = TrafficGenerator(nodes=list(range(9)), seed=11)
    p1, p2 = g1.generate(0), g2.generate(0)
    assert len(p1) == len(p2)
    assert [p.size for p in p1] == [p.size for p in p2]


# ---------------------------------------------------------------- seeding (H1)
def _offered_trace(env, steps=15, seed=None):
    env.reset(seed=seed)
    act = np.full(2 * env.num_edges, 0.5, dtype=np.float32)
    return [env.step(act)[4]["offered_units"] for _ in range(steps)]


def test_unseeded_resets_draw_fresh_traffic():
    """Regression guard: reset() without a seed used to re-seed the traffic
    RNG with the constructor seed, so every training episode replayed one
    fixed traffic realisation."""
    env = CarlaGymEnv(max_steps=15, seed=0)
    first = _offered_trace(env, seed=0)
    second = _offered_trace(env)
    third = _offered_trace(env)
    assert first != second and second != third


def test_seeded_reset_replays_identical_traffic():
    env = CarlaGymEnv(max_steps=15, seed=0)
    a = _offered_trace(env, seed=123)
    _offered_trace(env)                      # advance the stream in between
    b = _offered_trace(env, seed=123)
    assert a == b


def test_constructor_seed_fixes_first_unseeded_episode():
    a = _offered_trace(CarlaGymEnv(max_steps=15, seed=5))
    b = _offered_trace(CarlaGymEnv(max_steps=15, seed=5))
    c = _offered_trace(CarlaGymEnv(max_steps=15, seed=6))
    assert a == b and a != c


def test_trace_traffic_unseeded_resets_differ(tmp_path):
    import pandas as pd
    from carla_gym.core.trace_traffic import TraceTrafficGenerator

    csv = tmp_path / "trace.csv"
    pd.DataFrame({"step": [0, 0, 1, 1], "src": [0, 1, 2, 3],
                  "dst": [4, 5, 6, 7], "volume": [50.0, 60.0, 40.0, 70.0]}).to_csv(csv, index=False)
    traffic = TraceTrafficGenerator(list(range(9)), str(csv), target_load=100.0, seed=0)
    env = CarlaGymEnv(max_steps=15, traffic=traffic, seed=0)
    first = _offered_trace(env, seed=0)
    second = _offered_trace(env)
    assert first != second
    assert _offered_trace(env, seed=0) == first


def test_multi_agent_unseeded_resets_draw_fresh_traffic():
    pytest.importorskip("pettingzoo")
    from carla_gym.marl_env import MultiAgentCarlaGymEnv

    def episode(env, seed=None):
        obs, _ = env.reset(seed=seed)
        out = []
        for _ in range(15):
            acts = {a: np.full(env.action_space(a).shape, 0.5, dtype=np.float32)
                    for a in env.agents}
            obs, *_ = env.step(acts)
            out.append(obs[env.possible_agents[0]].tolist())
        return out

    env = MultiAgentCarlaGymEnv(max_steps=15, seed=0)
    first = episode(env, seed=0)
    second = episode(env)
    assert first != second
    assert episode(env, seed=0) == first


# ---------------------------------------------------------------- link service (M2) / buffers (N1)
def _single_packet_net(size=12.0, deadline=8):
    from carla_gym.core.traffic import Packet, QoSClass
    net = NetworkModel("ring", 4, link_capacity=30.0, node_buffer=1000.0, seed=0)
    net.update_routing(np.ones(net.num_edges()))
    net.admit([Packet(0, 1, QoSClass.VIDEO, size, 1.5, deadline, 0)])
    return net


def test_packet_larger_than_link_budget_is_served_over_several_steps():
    """Regression guard: a 12-unit packet on a link serving 9 units/step used
    to wait forever, which made every allocation below ~0.4 equally useless."""
    net = _single_packet_net()
    alloc = np.full(net.num_edges(), 0.3)            # 9 units/step
    first = net.forward(alloc, 0)
    assert first["delivered_units"] == 0.0
    second = net.forward(alloc, 1)
    assert second["delivered_units"] == pytest.approx(12.0)
    assert second["latencies"][0][0] == 2.0


def test_zero_allocation_moves_nothing():
    net = _single_packet_net()
    for s in range(20):
        assert net.forward(np.zeros(net.num_edges()), s)["delivered_units"] == 0.0
    assert net.backlog_units() == pytest.approx(12.0)


def test_link_never_serves_more_than_its_allocation():
    env = CarlaGymEnv(max_steps=60, bandwidth_budget=None, seed=12)
    env.reset(seed=12)
    rng = np.random.default_rng(12)
    for _ in range(60):
        a = rng.uniform(0, 1, 2 * env.num_edges)
        env.step(a)
        served = env.network._served
        assert np.all(served <= env.network.link_capacity * a[:env.num_edges] + 1e-6)


def test_reward_increases_with_bandwidth_without_a_dead_zone():
    def run(weight):
        env = CarlaGymEnv(max_steps=100, bandwidth_budget=None, seed=13)
        env.reset(seed=13)
        a = np.full(2 * env.num_edges, weight, dtype=np.float32)
        return float(np.mean([env.step(a)[1] for _ in range(100)][20:]))

    rewards = [run(w) for w in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)]
    assert all(b > a + 0.01 for a, b in zip(rewards, rewards[1:]))


@pytest.mark.parametrize("weight,load", [(0.6, 1.0), (1.0, 1.0), (0.6, 2.0), (1.0, 2.0)])
def test_node_buffers_bounded_while_traffic_moves(weight, load):
    """Regression guard: downstream space was checked against a partial view
    of the buffer, so nodes held several times `node_buffer` under load."""
    env = CarlaGymEnv(max_steps=80, bandwidth_budget=None, load_scale=load, seed=14)
    env.reset(seed=14)
    a = np.full(2 * env.num_edges, weight, dtype=np.float32)
    for _ in range(80):
        env.step(a)
        assert np.all(env.network.get_node_occupancy() <= env.network.node_buffer + 1e-6)


# ---------------------------------------------------------------- reward honesty (M1)
def test_packet_stuck_in_queue_counts_as_deadline_miss_once():
    net = _single_packet_net(deadline=3)
    alloc = np.zeros(net.num_edges())
    expired = [net.forward(alloc, s)["queue_expired"] for s in range(10)]
    assert sum(expired) == 1 and expired[3] == 1     # age 4 > deadline 3 at step 3
    late = net.forward(np.ones(net.num_edges()), 10)
    assert late["delivered_count"] == 1 and late["deadline_misses"] == 0


def test_starved_network_is_penalised_for_delay_and_deadlines():
    """Regression guard: with nothing delivered, the delay and deadline terms
    used to read 0, so a fully stalled network looked free of latency."""
    env = CarlaGymEnv(max_steps=40, seed=15)
    env.reset(seed=15)
    infos = [env.step(np.zeros(2 * env.num_edges, dtype=np.float32))[4] for _ in range(40)]
    assert infos[-1]["reward_delay"] == pytest.approx(1.0)
    assert sum(i["deadline_misses"] for i in infos) > 0
    assert infos[-1]["mean_queue_age_steps"] > 0.0


def test_deadline_coefficient_is_configurable():
    def total(m):
        env = CarlaGymEnv(max_steps=40, m=m, seed=16)
        env.reset(seed=16)
        a = np.full(2 * env.num_edges, 0.4, dtype=np.float32)
        return sum(env.step(a)[1] for _ in range(40))

    assert total(0.0) > total(0.5) > total(2.0)
