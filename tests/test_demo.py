"""tests/test_demo.py -- the side-by-side demo (examples/demo.py)."""

import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from examples.demo import build_animation, main, make_policy, simulate  # noqa: E402


def test_simulate_records_consistent_series():
    rec = simulate("queue_proportional", seed=2000, steps=40, budget=0.8)
    assert len(rec["util"]) == len(rec["occ"]) == len(rec["reward"]) == 40
    assert rec["util"][0].shape == (12,) and rec["occ"][0].shape == (9,)
    assert all(0.0 <= u.min() and u.max() <= 1.0 for u in rec["util"])
    assert all(0.0 <= o.min() and o.max() <= 1.0 for o in rec["occ"])
    # cumulative percentages are what the totals say they are
    assert rec["delivered_pct"][-1] == pytest.approx(100 * sum(rec["delivered"]) / sum(rec["offered"]))
    assert rec["dropped_pct"][-1] == pytest.approx(100 * sum(rec["dropped"]) / sum(rec["offered"]))


def test_both_sides_see_identical_traffic():
    a = simulate("max_bandwidth", seed=2001, steps=30, budget=0.8)
    b = simulate("queue_proportional", seed=2001, steps=30, budget=0.8)
    assert a["offered"] == b["offered"]
    assert a["delivered"] != b["delivered"]          # ...but are controlled differently


def test_animation_saves_a_gif(tmp_path):
    left = simulate("max_bandwidth", seed=2000, steps=12, budget=0.8, bottlenecks={4: 0.5})
    right = simulate("congestion_aware", seed=2000, steps=12, budget=0.8, bottlenecks={4: 0.5})
    fig, anim = build_animation(left, right, every=4)
    from matplotlib.animation import PillowWriter
    out = tmp_path / "demo.gif"
    anim.save(str(out), writer=PillowWriter(fps=5), dpi=40)
    assert out.stat().st_size > 1000


def test_unknown_policy_and_missing_model_fail_clearly(tmp_path):
    with pytest.raises(SystemExit, match="Unknown policy"):
        make_policy("dijkstra", None)
    pytest.importorskip("stable_baselines3")
    with pytest.raises(SystemExit, match="No saved PPO model"):
        make_policy("ppo", str(tmp_path / "nope.zip"))


def test_saved_agent_runs_in_the_demo_and_cli_writes_gif(tmp_path):
    pytest.importorskip("stable_baselines3")
    if not os.path.exists(os.path.join("results", "grid_b08", "ppo_seed0", "model.zip")):
        pytest.skip("saved PPO model not present")
    rec = simulate("ppo", seed=2000, steps=25, budget=0.8)
    assert np.isfinite(rec["reward"]).all() and rec["delivered_pct"][-1] > 0
    out = tmp_path / "cli.gif"
    main(["--steps", "8", "--every", "4", "--save", str(out), "--left", "max_bandwidth", "--right", "ppo"])
    assert out.exists()
