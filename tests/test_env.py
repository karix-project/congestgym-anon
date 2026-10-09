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
    rc = RewardCalculator(k=0.5, c=1.0)
    assert rc.compute(1.0, 0.0, 0.0) > rc.compute(1.0, 0.5, 0.0)
    assert rc.compute(1.0, 0.0, 0.0) > rc.compute(1.0, 0.0, 0.5)


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
