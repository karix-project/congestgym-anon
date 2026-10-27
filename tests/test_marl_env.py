"""
tests/test_marl_env.py

Multi-agent environment: PettingZoo parallel-API conformance on every
topology, link ownership, and agreement with the single-agent simulator.
"""

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
from pettingzoo.test import parallel_api_test  # noqa: E402

from congestgym.env import CongestGymEnv  # noqa: E402
from congestgym.marl_env import MultiAgentCongestGymEnv  # noqa: E402

TOPOLOGIES = ["grid", "random", "star", "mesh", "ring"]


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_parallel_api_conformance(topology):
    env = MultiAgentCongestGymEnv(topology=topology, max_steps=20, seed=0)
    parallel_api_test(env, num_cycles=50)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_every_link_has_exactly_one_owner_and_no_agent_is_empty(topology):
    env = MultiAgentCongestGymEnv(topology=topology, seed=0)
    owned = [e for edges in env.agent_edges.values() for e in edges]
    assert sorted(owned) == list(range(env.network.num_edges()))
    for agent in env.possible_agents:
        assert env.action_space(agent).shape == (2 * len(env.agent_edges[agent]),)
        assert env.action_space(agent).shape[0] > 0


def test_star_hub_is_not_an_agent():
    env = MultiAgentCongestGymEnv(topology="star", num_nodes=9, seed=0)
    assert len(env.possible_agents) == 8 and "node_0" not in env.possible_agents


def test_matches_single_agent_env_for_equivalent_actions():
    """The same global action must give the same reward in both envs."""
    single = CongestGymEnv(max_steps=30, seed=3)
    multi = MultiAgentCongestGymEnv(max_steps=30, seed=3)
    single.reset(seed=3)
    multi.reset(seed=3)
    rng = np.random.default_rng(3)
    e = single.num_edges
    for _ in range(30):
        action = rng.uniform(0, 1, 2 * e).astype(np.float32)
        _, r_single, *_ = single.step(action)
        acts = {}
        for agent, edges in multi.agent_edges.items():
            acts[agent] = np.concatenate([action[edges], action[[e + i for i in edges]]])
        _, rewards, *_ = multi.step(acts)
        assert all(r == pytest.approx(r_single) for r in rewards.values())


def test_episode_truncates_and_clears_agents():
    env = MultiAgentCongestGymEnv(max_steps=5, seed=0)
    env.reset(seed=0)
    for _ in range(5):
        acts = {a: env.action_space(a).sample() for a in env.agents}
        _, _, terms, truncs, infos = env.step(acts)
    assert all(truncs.values()) and not any(terms.values())
    assert env.agents == []
    assert "backlog_units" in next(iter(infos.values()))


def test_invalid_budget_rejected():
    with pytest.raises(ValueError, match="bandwidth_budget"):
        MultiAgentCongestGymEnv(bandwidth_budget=1.5)


def test_missing_pettingzoo_fails_loudly(monkeypatch):
    import congestgym.marl_env as marl
    monkeypatch.setattr(marl, "_HAS_PETTINGZOO", False)
    with pytest.raises(ImportError, match="pettingzoo"):
        MultiAgentCongestGymEnv()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_agent_action_rejected(bad):
    env = MultiAgentCongestGymEnv(max_steps=5, seed=0)
    env.reset(seed=0)
    acts = {a: np.full(env.action_space(a).shape, 0.5, dtype=np.float32) for a in env.agents}
    first = env.agents[0]
    acts[first][0] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        env.step(acts)


def test_wrong_shape_agent_action_rejected():
    env = MultiAgentCongestGymEnv(max_steps=5, seed=0)
    env.reset(seed=0)
    acts = {a: np.full(env.action_space(a).shape, 0.5, dtype=np.float32) for a in env.agents}
    acts[env.agents[0]] = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        env.step(acts)


def test_step_requires_an_action_for_every_live_agent():
    """Regression guard: a missing agent's links silently got zero bandwidth."""
    env = MultiAgentCongestGymEnv(max_steps=5, seed=0)
    env.reset(seed=0)
    acts = {a: env.action_space(a).sample() for a in env.agents}
    partial = dict(list(acts.items())[:-1])
    with pytest.raises(ValueError, match="one action per live agent"):
        env.step(partial)
    with pytest.raises(ValueError, match="one action per live agent"):
        env.step({**acts, "node_99": np.zeros(2, dtype=np.float32)})
    env.step(acts)
