"""
tests/test_dataset_pipeline.py

Covers the trace-data integration your friends added: raw dataset ->
dataset_adapters -> TraceTrafficGenerator -> CarlaGymEnv. These tests run
against the real uploaded CSVs (copied into data/) so they catch schema
drift in the actual files, not just a synthetic stand-in.

Two regressions are specifically guarded against because they were found
and fixed during integration:
    * a heavy traffic node landing on a low-degree topology position
      (throttles throughput regardless of bandwidth policy -- nothing to
      do with load calibration)
    * packet units leaking when traffic comes from a trace instead of the
      Poisson generator the original conservation test covered
"""

import os

import numpy as np
import pandas as pd
import pytest

from carla_gym.core.dataset_adapters import (
    convert_cs448b, convert_packet_capture, detect_format, _topology_degree_order,
)
from carla_gym.core.network import NetworkModel
from carla_gym.core.trace_traffic import TraceTrafficGenerator
from carla_gym.env import CarlaGymEnv

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CS448B = os.path.join(DATA, "cs448b_ipasn.csv")
PCAP_NUMERIC = os.path.join(DATA, "output1.csv")
PCAP_TEXT = os.path.join(DATA, "Midterm_53_group.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CS448B), reason="sample datasets not present in data/"
)


# ---------------------------------------------------------------- format detection
def test_detect_format():
    assert detect_format(CS448B) == "cs448b"
    assert detect_format(PCAP_NUMERIC) == "pcap"
    assert detect_format(PCAP_TEXT) == "pcap"


# ---------------------------------------------------------------- cs448b adapter
def test_cs448b_schema_and_bounds():
    df = convert_cs448b(CS448B, internal_nodes=10, external_buckets=6)
    assert set(df.columns) == {"step", "src", "dst", "volume"}
    assert df.attrs["num_nodes"] == 16
    assert df["src"].between(0, 15).all() and df["dst"].between(0, 15).all()
    assert (df["src"] != df["dst"]).all()


def test_cs448b_rejects_l_ipn_overflow():
    with pytest.raises(ValueError):
        convert_cs448b(CS448B, internal_nodes=5, external_buckets=6)  # l_ipn goes to 9


# ---------------------------------------------------------------- pcap adapter
def test_pcap_numeric_and_text_agree():
    """output1.csv and Midterm_53_group.csv are the same capture exported two
    ways; after conversion their per-step load should be close (not
    identical -- the text export drops a few hundred non-IP rows the
    numeric export already lacks)."""
    a = convert_packet_capture(PCAP_NUMERIC, num_nodes=16, bin_seconds=5.0)
    b = convert_packet_capture(PCAP_TEXT, num_nodes=16, bin_seconds=5.0)
    load_a = a.groupby("step")["volume"].sum().mean()
    load_b = b.groupby("step")["volume"].sum().mean()
    assert load_a == pytest.approx(load_b, rel=0.1)


def test_pcap_schema():
    df = convert_packet_capture(PCAP_NUMERIC, num_nodes=16, bin_seconds=5.0)
    assert set(df.columns) == {"step", "src", "dst", "volume"}
    assert (df["src"] != df["dst"]).all()
    assert df["src"].between(0, 15).all() and df["dst"].between(0, 15).all()


def test_pcap_rejects_unknown_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        convert_packet_capture(str(bad))


# ---------------------------------------------------------------- topology-aware placement (regression)
def test_adapter_output_reproducible_across_processes(tmp_path):
    """Regression guard: the adapters used to bucket IPs/ASNs with
    Python's builtin hash(), which is randomized per-process by default
    (PYTHONHASHSEED). That made every re-run of prepare_trace.py produce
    a different node mapping for the SAME raw CSV -- a serious
    reproducibility break, not a cosmetic one. Runs the adapter in two
    genuinely separate subprocesses (distinct hash seeds) and requires
    identical output."""
    import subprocess
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    script = (
        "from carla_gym.core.dataset_adapters import convert_packet_capture; "
        "convert_packet_capture({src!r}, {dst!r}, num_nodes=16, bin_seconds=5.0)"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "1", "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))}
    env_b = {**os.environ, "PYTHONHASHSEED": "2", "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))}
    subprocess.run(["python3", "-c", script.format(src=PCAP_NUMERIC, dst=str(out_a))],
                   env=env_a, check=True)
    subprocess.run(["python3", "-c", script.format(src=PCAP_NUMERIC, dst=str(out_b))],
                   env=env_b, check=True)
    df_a = pd.read_csv(out_a)
    df_b = pd.read_csv(out_b)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_heaviest_node_lands_on_best_connected_position():
    """Regression guard: before this was fixed, frequency/hash-based node
    assignment could put the single busiest traffic concentration on an
    arbitrary topology position -- including, by chance, a 2-link grid
    corner, which throttles throughput no matter the bandwidth policy."""
    df = convert_packet_capture(PCAP_NUMERIC, num_nodes=16, bin_seconds=5.0, topology="grid")
    net = NetworkModel(topology="grid", num_nodes=16)
    degrees = dict(net.graph.degree())

    load = df.groupby("dst")["volume"].sum().add(
        df.groupby("src")["volume"].sum(), fill_value=0.0
    ).sort_values(ascending=False)
    busiest_node = int(load.index[0])
    assert degrees[busiest_node] == max(degrees.values())


def test_degree_order_is_well_formed():
    order = _topology_degree_order("grid", 16)
    assert sorted(order) == list(range(16))
    net = NetworkModel(topology="grid", num_nodes=16)
    degs = dict(net.graph.degree())
    assert all(degs[order[i]] >= degs[order[i + 1]] for i in range(len(order) - 1))


# ---------------------------------------------------------------- full pipeline
@pytest.mark.parametrize("raw_path,kind", [
    (CS448B, "cs448b"), (PCAP_NUMERIC, "pcap"), (PCAP_TEXT, "pcap"),
])
def test_full_pipeline_runs_without_crashing(raw_path, kind, tmp_path):
    out_csv = tmp_path / "trace.csv"
    if kind == "cs448b":
        convert_cs448b(raw_path, str(out_csv))
        num_nodes = 16
    else:
        convert_packet_capture(raw_path, str(out_csv), num_nodes=16, bin_seconds=5.0)
        num_nodes = 16

    traffic = TraceTrafficGenerator(nodes=list(range(num_nodes)), csv_path=str(out_csv),
                                    target_load=100.0, seed=0)
    env = CarlaGymEnv(topology="grid", num_nodes=num_nodes, bandwidth_budget=0.5,
                      max_steps=30, traffic=traffic, seed=0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)


def test_trace_driven_env_conserves_packet_units(tmp_path):
    """Same invariant as the Poisson-traffic conservation test, but now
    with trace-driven demand -- the simulator's bookkeeping shouldn't
    care where packets came from."""
    out_csv = tmp_path / "trace.csv"
    convert_packet_capture(PCAP_NUMERIC, str(out_csv), num_nodes=16, bin_seconds=5.0)
    traffic = TraceTrafficGenerator(nodes=list(range(16)), csv_path=str(out_csv),
                                    target_load=100.0, seed=0)
    env = CarlaGymEnv(topology="grid", num_nodes=16, bandwidth_budget=0.5,
                      max_steps=50, traffic=traffic, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    offered = delivered = dropped = 0.0
    for _ in range(50):
        _, _, _, _, info = env.step(rng.uniform(0, 1, env.num_edges))
        offered += info["offered_units"]
        delivered += info["delivered_units"]
        dropped += info["dropped_units"]
    backlog = env.network.backlog_units()
    assert offered == pytest.approx(delivered + dropped + backlog, rel=1e-6)


def test_reference_policies_callable_with_and_without_rng():
    """Regression guard: train_ppo.py's rollout() calls policy functions as
    fn(env, obs) (no rng), while random_agent_test.py's own run_policy()
    calls them as fn(env, obs, rng). Both call sites must work."""
    from examples.random_agent_test import policy_max, policy_min, policy_queue_proportional, policy_random

    env = CarlaGymEnv(topology="grid", num_nodes=16, max_steps=5, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for fn in (policy_max, policy_min, policy_queue_proportional, policy_random):
        a1 = fn(env, obs)          # train_ppo.py call style
        a2 = fn(env, obs, rng)     # random_agent_test.py call style
        assert a1.shape == (env.num_edges,) and a2.shape == (env.num_edges,)


def test_rollout_works_with_monitor_wrapped_env():
    """Regression guard: train_ppo.py's held-out eval wraps the env in
    stable_baselines3's Monitor, which does NOT forward custom attributes
    like num_edges/num_nodes -- calling a reference policy directly on the
    wrapped env raises AttributeError. rollout() must hand policies the
    unwrapped env instead."""
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.monitor import Monitor
    from examples.random_agent_test import policy_queue_proportional
    from examples.train_ppo import rollout

    env = Monitor(CarlaGymEnv(topology="grid", num_nodes=16, max_steps=40, seed=0))
    m, sd = rollout(env, policy_queue_proportional, seeds=[0, 1], steps=40)  # > default warmup=20
    assert np.isfinite(m) and np.isfinite(sd)


def test_bandwidth_budget_changes_outcome_on_real_trace(tmp_path):
    """The central claim of the trace-driven setup: under a tight budget,
    congestion-aware allocation should not do worse than uniform -- if it
    did, there would be nothing for PPO to learn from this trace."""
    out_csv = tmp_path / "trace.csv"
    convert_packet_capture(PCAP_NUMERIC, str(out_csv), num_nodes=16, bin_seconds=5.0)

    def run(policy_weight_fn, budget):
        traffic = TraceTrafficGenerator(nodes=list(range(16)), csv_path=str(out_csv),
                                        target_load=110.0, seed=0)
        env = CarlaGymEnv(topology="grid", num_nodes=16, bandwidth_budget=budget,
                          max_steps=100, traffic=traffic, seed=0)
        obs, _ = env.reset(seed=0)
        total = 0.0
        for t in range(100):
            obs, reward, *_ = env.step(policy_weight_fn(env, obs))
            if t >= 20:
                total += reward
        return total

    uniform = lambda e, o: np.ones(e.num_edges)
    def queue_prop(e, o):
        pending = o[e.num_edges:2 * e.num_edges]
        return np.clip(pending / max(pending.max(), 1e-9), 0.05, 1.0)

    r_uniform = run(uniform, budget=0.5)
    r_queue_prop = run(queue_prop, budget=0.5)
    assert r_queue_prop >= r_uniform
