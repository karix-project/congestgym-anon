"""
tests/test_dataset_pipeline.py

Covers the trace-data integration: the committed Abilene trace ->
TraceTrafficGenerator -> CongestGymEnv, and the script that rebuilds the trace
from SNDlib's archive (run only where the archive exists in data/raw/).

A regression specifically guarded against: packet units leaking when traffic
comes from a trace instead of the Poisson generator the original
conservation test covered.
"""

import os

import numpy as np
import pandas as pd
import pytest

from congestgym.core.trace_traffic import TraceTrafficGenerator
from congestgym.env import CongestGymEnv

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(DATA, "raw")
# The distributed trace: one week of SNDlib's Abilene traffic matrices
# (data/ABILENE_TRACE.md).
ABILENE_TRACE = os.path.join(DATA, "abilene_trace_20040301_7d.csv.gz")
SNDLIB_ARCHIVE = os.path.join(RAW, "sndlib", "archives", "directed-abilene-zhang-5min-over-6months-ALL-native.tgz")


def test_trace_driven_env_conserves_packet_units(tmp_path):
    """Same invariant as the Poisson-traffic conservation test, but now
    with trace-driven demand -- the simulator's bookkeeping shouldn't
    care where packets came from."""
    traffic = TraceTrafficGenerator(nodes=list(range(12)), csv_path=ABILENE_TRACE,
                                    target_load=100.0, seed=0, random_start=True)
    env = CongestGymEnv(topology="abilene", num_nodes=12, bandwidth_budget=0.5,
                      max_steps=50, traffic=traffic, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    offered = delivered = dropped = 0.0
    for _ in range(50):
        _, _, _, _, info = env.step(rng.uniform(0, 1, 2 * env.num_edges))
        offered += info["offered_units"]
        delivered += info["delivered_units"]
        dropped += info["dropped_units"]
    backlog = env.network.backlog_units()
    assert offered == pytest.approx(delivered + dropped + backlog, rel=1e-6)


def test_reference_policies_callable_with_and_without_rng():
    """Regression guard: train_agent.py's rollout() calls policy functions as
    fn(env, obs) (no rng), while the benchmark calls them as fn(env, obs, rng).
    Both call sites must work."""
    from congestgym.baselines import policy_max, policy_min, policy_queue_proportional, policy_random

    env = CongestGymEnv(topology="grid", num_nodes=16, max_steps=5, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for fn in (policy_max, policy_min, policy_queue_proportional, policy_random):
        a1 = fn(env, obs)          # rollout() call style
        a2 = fn(env, obs, rng)     # benchmark call style
        assert a1.shape == (2 * env.num_edges,) and a2.shape == (2 * env.num_edges,)


def test_rollout_works_with_monitor_wrapped_env():
    """Regression guard: train_agent.py's held-out eval wraps the env in
    stable_baselines3's Monitor, which does NOT forward custom attributes
    like num_edges/num_nodes -- calling a reference policy directly on the
    wrapped env raises AttributeError. rollout() must hand policies the
    unwrapped env instead."""
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.monitor import Monitor
    from congestgym.baselines import policy_queue_proportional
    from examples.train_agent import rollout

    env = Monitor(CongestGymEnv(topology="grid", num_nodes=16, max_steps=40, seed=0))
    m, sd = rollout(env, policy_queue_proportional, seeds=[0, 1], steps=40)  # > default warmup=20
    assert np.isfinite(m) and np.isfinite(sd)


def test_bandwidth_budget_changes_outcome_on_real_trace():
    """The central claim of the trace-driven setup: under a tight budget,
    congestion-aware allocation should not do worse than uniform -- if it
    did, there would be nothing for an agent to learn from this trace."""
    def run(policy_weight_fn, budget):
        traffic = TraceTrafficGenerator(nodes=list(range(12)), csv_path=ABILENE_TRACE,
                                        target_load=120.0, seed=0, random_start=True)
        env = CongestGymEnv(topology="abilene", num_nodes=12, bandwidth_budget=budget,
                          max_steps=100, traffic=traffic, seed=0)
        obs, _ = env.reset(seed=0)
        total = 0.0
        for t in range(100):
            obs, reward, *_ = env.step(policy_weight_fn(env, obs))
            if t >= 20:
                total += reward
        return total

    uniform = lambda e, o: np.ones(2 * e.num_edges)
    def queue_prop(e, o):
        pending = o[e.num_edges:2 * e.num_edges]
        bw = np.clip(pending / max(pending.max(), 1e-9), 0.05, 1.0)
        return np.concatenate([bw, np.ones(e.num_edges)])

    r_uniform = run(uniform, budget=0.5)
    r_queue_prop = run(queue_prop, budget=0.5)
    assert r_queue_prop >= r_uniform


# ---------------------------------------------------------------- trace load calibration / committed traces
def test_trace_mean_load_matches_target():
    gen = TraceTrafficGenerator(list(range(12)), ABILENE_TRACE, target_load=110.0, seed=0)
    expected = sum(rows[:, 2].sum() for rows in gen.by_step.values()) / gen.n_steps
    assert expected == pytest.approx(110.0)
    realised = np.mean([sum(p.size for p in gen.generate(t)) for t in range(gen.n_steps)])
    assert realised == pytest.approx(110.0, rel=0.05)


def test_trace_mean_load_counts_empty_bins(tmp_path):
    """Regression guard: the rescaling averaged over non-empty bins only, so a
    trace with gaps under-delivered."""
    csv = tmp_path / "gappy.csv"
    pd.DataFrame({"step": [0, 3, 9], "src": [0, 1, 2], "dst": [1, 2, 0],
                  "volume": [5.0, 1.0, 2.0]}).to_csv(csv, index=False)        # bins 1-2, 4-8 empty
    gen = TraceTrafficGenerator(list(range(9)), str(csv), target_load=50.0, seed=0)
    assert gen.n_steps == 10
    expected = sum(rows[:, 2].sum() for rows in gen.by_step.values()) / gen.n_steps
    assert expected == pytest.approx(50.0)
    realised = np.mean([sum(p.size for p in gen.generate(t)) for t in range(300 * gen.n_steps)])
    assert realised == pytest.approx(50.0, rel=0.05)


def test_trace_with_no_usable_rows_is_rejected(tmp_path):
    csv = tmp_path / "trace.csv"
    pd.DataFrame({"step": [0], "src": [50], "dst": [51], "volume": [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="no usable rows"):
        TraceTrafficGenerator(list(range(9)), str(csv))


@pytest.mark.skipif(not os.path.exists(SNDLIB_ARCHIVE),
                    reason="the SNDlib archive is not distributed; download it to run this test")
def test_committed_abilene_trace_matches_the_conversion_script(tmp_path):
    """data/abilene_trace_20040301_7d.csv.gz must be exactly what
    scripts/prepare_sndlib_trace.py produces from SNDlib's archive."""
    from scripts.prepare_sndlib_trace import main as prepare
    out = tmp_path / "abilene.csv.gz"
    prepare(["--tgz", SNDLIB_ARCHIVE, "--start", "20040301", "--days", "7", "--out", str(out)])
    pd.testing.assert_frame_equal(pd.read_csv(out), pd.read_csv(ABILENE_TRACE))


# ---------------------------------------------------------------- committed trace (no raw data needed)
def test_committed_trace_is_well_formed_and_anonymous():
    """The distributed trace holds node ids only: no addresses, no self-loops,
    one row per (step, src, dst)."""
    df = pd.read_csv(ABILENE_TRACE)
    assert list(df.columns) == ["step", "src", "dst", "volume"]
    assert df["src"].between(0, 11).all() and df["dst"].between(0, 11).all()
    assert (df["src"] != df["dst"]).all() and (df["volume"] > 0).all()
    assert not df.duplicated(["step", "src", "dst"]).any()
    assert df["step"].min() == 0
    for col in ("step", "src", "dst"):
        assert pd.api.types.is_integer_dtype(df[col])


def test_committed_trace_drives_the_env():
    traffic = TraceTrafficGenerator(nodes=list(range(12)), csv_path=ABILENE_TRACE, target_load=100.0,
                                    seed=0, random_start=True)
    env = CongestGymEnv(topology="abilene", num_nodes=12, bandwidth_budget=0.5, max_steps=30,
                      traffic=traffic, seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, reward, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(reward)
