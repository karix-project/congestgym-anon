"""M3: multi-agent env. PettingZoo API test, zero-size action spaces, dependency."""
import re
from pathlib import Path

from pettingzoo.test import parallel_api_test

from carla_gym.marl_env import MultiAgentCarlaGymEnv

ROOT = Path(__file__).resolve().parents[2]

for topo in ("grid", "star", "ring", "mesh", "random"):
    env = MultiAgentCarlaGymEnv(topology=topo, max_steps=20, seed=0)
    empty = [a for a in env.possible_agents if env.action_space(a).shape[0] == 0]
    try:
        parallel_api_test(env, num_cycles=50)
        status = "parallel_api_test PASSED"
    except Exception as exc:  # noqa: BLE001 - report whatever the checker raises
        status = f"parallel_api_test FAILED: {type(exc).__name__}: {exc}"
    print(f"{topo:6s}: {status}; agents with zero-size action space: "
          f"{len(empty)} of {len(env.possible_agents)}")

for name in ("requirements.txt", "pyproject.toml"):
    declared = bool(re.search("pettingzoo", (ROOT / name).read_text(), re.I))
    print(f"pettingzoo declared in {name}: {declared}")
