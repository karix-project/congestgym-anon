"""scripts/evaluate_saved.py rebuilds each scenario from the training queue's flags."""
import pytest

pytest.importorskip("stable_baselines3")

from examples.train_agent import parse_args as train_args          # noqa: E402
from scripts.evaluate_saved import scenario_flags                  # noqa: E402
from scripts.run_training_matrix import known_configurations      # noqa: E402


@pytest.mark.parametrize("name", sorted(known_configurations()))
def test_every_queue_entry_is_recognised_from_its_directory(name):
    found, flags = scenario_flags(f"results/rl/{name}/", None, [])
    assert found == name and flags == list(known_configurations()[name])
    train_args(flags)                                               # flags parse as train_agent flags


def test_explicit_scenario_overrides_directory_and_extra_flags_are_appended():
    found, flags = scenario_flags("anywhere", "grid_l125_b06", ["--steps", "100"])
    assert found == "grid_l125_b06" and flags[-2:] == ["--steps", "100"]


def test_unknown_directory_falls_back_to_given_flags():
    found, flags = scenario_flags("results/grid_b08", None, ["--budget", "0.5"])
    assert found is None and flags == ["--budget", "0.5"]
    assert train_args(flags).budget == 0.5


def test_ablation_flags_reach_the_wrapper():
    _, flags = scenario_flags("results/rl/grid_l125_b06_routing_only", None, [])
    a = train_args(flags)
    assert (a.control, a.load_scale, a.budget) == ("routing", 1.25, 0.6)
