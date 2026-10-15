"""carla_gym/bounds.py: the fluid flow upper bound."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from carla_gym.bounds import expected_demand, flow_bound, scenario_bound   # noqa: E402
from carla_gym.core.network import NetworkModel                            # noqa: E402
from carla_gym.env import CarlaGymEnv                                       # noqa: E402


def _demand(n, pairs):
    d = np.zeros((n, n))
    for (s, t), v in pairs.items():
        d[s, t] = v
    return d


def test_one_link_is_shared_by_both_directions():
    net = NetworkModel("ring", 3, link_capacity=30.0)            # triangle
    d = _demand(3, {(0, 1): 20.0, (1, 0): 20.0})
    # shortest routing: both directions must cross link {0,1}, 30 units per step
    r = flow_bound(net, d, budget=None, routing="shortest")
    assert r["delivered"] == pytest.approx(30.0)
    # free routing may also detour over 0-2-1, adding up to 30 more
    assert flow_bound(net, d, budget=None, routing="free")["delivered"] == pytest.approx(40.0)


def test_budget_limits_total_allocation():
    net = NetworkModel("ring", 3, link_capacity=30.0)
    d = _demand(3, {(0, 1): 100.0})
    # budget 1/3 of 3 links = one link's worth of allocation in total
    r = flow_bound(net, d, budget=1 / 3, routing="free")
    assert r["delivered"] == pytest.approx(30.0)
    assert r["allocation"].sum() <= 1.0 + 1e-9


def test_free_routing_is_never_below_shortest_routing():
    for topo, n in [("grid", 9), ("ring", 9), ("random", 9), ("mesh", 6)]:
        env = CarlaGymEnv(topology=topo, num_nodes=n, load_scale=2.0, bandwidth_budget=0.5, seed=0)
        free, short = scenario_bound(env, "free"), scenario_bound(env, "shortest")
        assert free["delivered_fraction"] >= short["delivered_fraction"] - 1e-9
        assert 0.0 < short["delivered_fraction"] <= 1.0 + 1e-9


def test_light_load_is_fully_deliverable():
    env = CarlaGymEnv(load_scale=0.1, bandwidth_budget=0.8, seed=0)
    assert scenario_bound(env)["delivered_fraction"] == pytest.approx(1.0)


def test_synthetic_expected_demand_matches_the_generator():
    env = CarlaGymEnv(load_scale=1.25, seed=0)
    d = expected_demand(env.traffic, env.num_nodes)
    assert d.sum() == pytest.approx(104.5 * 1.25) and np.all(np.diag(d) == 0)
    env.reset(seed=5)
    offered = np.mean([env.step(np.ones(env.action_space.shape, dtype=np.float32))[4]["offered_units"]
                       for _ in range(200)])
    assert offered == pytest.approx(d.sum(), rel=0.05)


def test_bound_is_above_what_the_simulator_delivers():
    """The best of several fixed policies, over a long run, stays below the bound."""
    from carla_gym.baselines import POLICIES
    from carla_gym.utils.qos import EpisodeQoS
    env = CarlaGymEnv(load_scale=1.25, bandwidth_budget=0.6, max_steps=400, seed=0)
    bound = scenario_bound(env)["delivered_fraction"]
    best = 0.0
    for name in ("queue_proportional", "congestion_aware", "max_bandwidth"):
        obs, _ = env.reset(seed=11)
        qos = EpisodeQoS()
        for t in range(400):
            obs, _, _, _, info = env.step(POLICIES[name](env, obs))
            if t >= 50:
                qos.add(info)
        best = max(best, qos.summary()["delivered_fraction"])
    assert best <= bound + 0.01
