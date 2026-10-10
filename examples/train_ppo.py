"""
examples/train_ppo.py  --  train PPO on CarlaGym and compare to the heuristics.

Thin entry point kept for compatibility: it runs examples/train_agent.py with
--algo ppo. See that file for options, outputs and the seed protocol.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python examples/train_ppo.py --timesteps 300000 --budget 0.8
    python examples/train_ppo.py --trace data/pcap_trace.csv --nodes 16 --load 110 --budget 0.5
"""
from examples.train_agent import main, make_env, rollout  # noqa: F401

if __name__ == "__main__":
    main(default_algo="ppo")
