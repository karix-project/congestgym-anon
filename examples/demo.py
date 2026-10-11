"""
examples/demo.py  --  watch two policies control the same congested network.

Both sides get exactly the same traffic. Links are drawn by how busy they are
(green to red, thicker = busier), nodes by how full their buffers are, and
the charts underneath track how much of the offered traffic each policy has
delivered and dropped so far. The percentages cover the whole episode from
step 0, so they differ slightly from the README tables, which skip a 20-step
warm-up.

    $env:PYTHONPATH = "."        (bash: export PYTHONPATH=.)
    python examples/demo.py                          # live window: best rule vs trained PPO
    python examples/demo.py --save demo.gif          # write an animation instead
    python examples/demo.py --left max_bandwidth --right queue_proportional
    python examples/demo.py --seed 2005 --steps 300 --bottleneck 4:0.5

--left / --right take a rule-based policy name (see carla_gym/baselines.py),
or `ppo` / `sac` for an agent saved by examples/train_agent.py (--model picks
the file). The saved agents were trained on the default scenario: 3x3 grid,
synthetic traffic, bandwidth budget 0.8.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from carla_gym.baselines import POLICIES
from carla_gym.env import CarlaGymEnv

LABELS = {
    "ppo": "Trained RL agent (PPO)", "sac": "Trained RL agent (SAC)",
    "max_bandwidth": "Rule: equal split", "queue_proportional": "Rule: queue-proportional",
    "load_aware_routing": "Rule: load-aware routing", "congestion_aware": "Rule: congestion-aware",
    "random": "Random actions", "min_bandwidth": "Rule: minimum bandwidth",
}
DEFAULT_MODEL = os.path.join("results", "grid_b08", "{algo}_seed0", "model.zip")


def make_policy(name: str, model_path: str | None):
    """Returns (act_fn(env, obs), uses_rescaled_actions)."""
    if name in POLICIES:
        rng = np.random.default_rng(0)
        return (lambda env, obs: POLICIES[name](env, obs, rng)), False
    if name in ("ppo", "sac"):
        from stable_baselines3 import PPO, SAC
        path = model_path or DEFAULT_MODEL.format(algo=name)
        if not os.path.exists(path):
            raise SystemExit(f"No saved {name.upper()} model at {path}. Train one with "
                             f"examples/train_agent.py or pass --model.")
        model = (PPO if name == "ppo" else SAC).load(path, device="cpu")
        # agents act in [-1, 1]; map to the env's native [0, 1]
        return (lambda env, obs: (np.clip(model.predict(obs, deterministic=True)[0], -1.0, 1.0) + 1.0) / 2.0), True
    raise SystemExit(f"Unknown policy '{name}'. Choose from {sorted(POLICIES)} or ppo / sac.")


def simulate(name: str, seed: int, steps: int, budget: float, bottlenecks=None, model_path=None) -> dict:
    """Run one episode and record everything the animation needs."""
    act, _ = make_policy(name, model_path)
    env = CarlaGymEnv(bandwidth_budget=budget, max_steps=steps, bottlenecks=bottlenecks, seed=seed)
    obs, _ = env.reset(seed=seed)
    net = env.network
    rec = {"name": name, "graph": net.graph, "edges": net.edge_list(), "util": [], "occ": [],
           "offered": [], "delivered": [], "dropped": [], "reward": []}
    for _ in range(steps):
        obs, reward, _, _, info = env.step(act(env, obs))
        rec["util"].append(net.get_edge_utilization().copy())
        rec["occ"].append(np.clip(net.get_node_occupancy() / net.get_node_buffers(), 0.0, 1.0))
        rec["offered"].append(info["offered_units"])
        rec["delivered"].append(info["delivered_units"])
        rec["dropped"].append(info["dropped_units"])
        rec["reward"].append(reward)
    off = np.maximum(np.cumsum(rec["offered"]), 1e-9)
    rec["delivered_pct"] = 100.0 * np.cumsum(rec["delivered"]) / off
    rec["dropped_pct"] = 100.0 * np.cumsum(rec["dropped"]) / off
    rec["bottlenecks"] = dict(bottlenecks or {})
    return rec


def _positions(graph):
    import networkx as nx
    n = graph.number_of_nodes()
    side = int(round(np.sqrt(n)))
    if side * side == n and graph.number_of_edges() == 2 * side * (side - 1):   # a grid
        return {node: (node % side, -(node // side)) for node in graph.nodes()}
    return nx.spring_layout(graph, seed=0)


def build_animation(left: dict, right: dict, every: int = 1, interval: int = 100):
    """Returns (figure, FuncAnimation)."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.animation import FuncAnimation

    steps = len(left["util"])
    frames = list(range(0, steps, every))
    pos = _positions(left["graph"])
    colors = {"left": "#1f77b4", "right": "#d62728"}

    fig = plt.figure(figsize=(12.5, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.0], hspace=0.30, wspace=0.12,
                            top=0.86, bottom=0.11, left=0.06, right=0.97)
    ax_net = {"left": fig.add_subplot(grid[0, 0]), "right": fig.add_subplot(grid[0, 1])}
    ax_del, ax_drop = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    fig.suptitle("CarlaGym demo: the same traffic, two controllers", fontsize=14, fontweight="bold")
    caption = fig.text(0.5, 0.02, "", ha="center", fontsize=9.5, color="dimgray")

    lines = {}
    for ax, key, title in ((ax_del, "delivered_pct", "Traffic delivered so far (%)"),
                           (ax_drop, "dropped_pct", "Traffic dropped so far (%)")):
        for side, rec in (("left", left), ("right", right)):
            lines[(key, side)], = ax.plot([], [], color=colors[side], lw=2,
                                          label=LABELS.get(rec["name"], rec["name"]))
        ax.set_xlim(0, steps)
        ax.set_ylim(0, 100 if key == "delivered_pct" else max(10.0, 1.45 * max(left[key].max(), right[key].max())))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if key == "delivered_pct" else "upper left", fontsize=9)

    def draw_network(ax, rec, t, color):
        ax.clear()
        g = rec["graph"]
        util = rec["util"][t]
        nx.draw_networkx_edges(g, pos, edgelist=rec["edges"], ax=ax, width=list(1.5 + 7.0 * util),
                               edge_color=[(float(u), 1.0 - float(u), 0.15) for u in util])
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=rec["occ"][t], cmap="RdYlGn_r", vmin=0, vmax=1,
                               node_size=list(500 + 900 * rec["occ"][t]), edgecolors="black", linewidths=0.8)
        if rec["bottlenecks"]:
            nx.draw_networkx_nodes(g, pos, nodelist=list(rec["bottlenecks"]), ax=ax, node_color="none",
                                   node_size=1900, edgecolors="black", linewidths=2.5)
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=9)
        ax.set_title(f"{LABELS.get(rec['name'], rec['name'])}\n"
                     f"delivered {rec['delivered_pct'][t]:.1f}%   dropped {rec['dropped_pct'][t]:.1f}%   "
                     f"mean reward {np.mean(rec['reward'][:t + 1]):+.2f}", fontsize=11, color=color)
        ax.set_axis_off()
        ax.margins(0.12)

    def update(t):
        draw_network(ax_net["left"], left, t, colors["left"])
        draw_network(ax_net["right"], right, t, colors["right"])
        x = np.arange(t + 1)
        for key in ("delivered_pct", "dropped_pct"):
            lines[(key, "left")].set_data(x, left[key][:t + 1])
            lines[(key, "right")].set_data(x, right[key][:t + 1])
        caption.set_text(f"step {t + 1} of {steps}      links: green = idle, red = saturated, thicker = busier"
                         f"      nodes: green = empty buffer, red = full, larger = fuller")
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=False)
    return fig, anim


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--left", default="queue_proportional", help="policy on the left panel")
    p.add_argument("--right", default="ppo", help="policy on the right panel")
    p.add_argument("--model", default=None, help="path to a saved agent (default: results/grid_b08/<algo>_seed0/model.zip)")
    p.add_argument("--seed", type=int, default=2000, help="traffic seed (2000+ were never used in training)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--budget", type=float, default=0.8)
    p.add_argument("--bottleneck", default=None, help="degraded nodes, e.g. 4:0.5")
    p.add_argument("--save", default=None, help="write a .gif instead of opening a window")
    p.add_argument("--every", type=int, default=1, help="draw every Nth step (smaller files, faster playback)")
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args(argv)

    bottlenecks = None
    if a.bottleneck:
        bottlenecks = {int(n): float(s) for n, s in (item.split(":") for item in a.bottleneck.split(","))}
    if a.save:
        import matplotlib
        matplotlib.use("Agg")

    left = simulate(a.left, a.seed, a.steps, a.budget, bottlenecks, a.model)
    right = simulate(a.right, a.seed, a.steps, a.budget, bottlenecks, a.model)
    for rec in (left, right):
        print(f"{LABELS.get(rec['name'], rec['name']):28s} delivered {rec['delivered_pct'][-1]:5.1f}%   "
              f"dropped {rec['dropped_pct'][-1]:5.1f}%   mean reward {np.mean(rec['reward']):+.3f}")

    fig, anim = build_animation(left, right, every=a.every, interval=int(1000 / a.fps))
    if a.save:
        from matplotlib.animation import PillowWriter
        anim.save(a.save, writer=PillowWriter(fps=a.fps), dpi=80)
        print(f"Saved {a.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
