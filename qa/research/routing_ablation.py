"""Researcher check: how much of the trained PPO agents' reward comes from the
routing half of the action? Replaces one half of the agent's action at test time."""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from carla_gym.env import CarlaGymEnv
from carla_gym.baselines import policy_queue_proportional
from carla_gym.utils.stats import ci95

E = 12
def run(model, mode, seed):
    env = CarlaGymEnv(bandwidth_budget=0.8, max_steps=200, seed=0)
    obs, _ = env.reset(seed=seed); r = []; changed = 0; prev = None
    for _ in range(200):
        a = (np.clip(model.predict(obs, deterministic=True)[0], -1, 1) + 1) / 2      # agent action in native [0, 1]
        if mode == "uniform_routing": a[E:] = 1.0
        if mode == "qp_bandwidth": a[:E] = policy_queue_proportional(env, obs)[:E]
        paths = {(s, d): tuple(p) for s, dd in env.network._current_paths.items() for d, p in dd.items()}
        obs, rew, *_ = env.step(a); r.append(rew)
        new = {(s, d): tuple(p) for s, dd in env.network._current_paths.items() for d, p in dd.items()}
        changed += np.mean([new[k] != paths[k] for k in new if k[0] != k[1]])
    return np.mean(r[20:]), changed / 200
rows = {m: [] for m in ("full", "uniform_routing", "qp_bandwidth")}; churn = []
for s in range(5):
    model = PPO.load(f"results/grid_b08/ppo_seed{s}/model.zip", device="cpu")
    for m in rows:
        out = [run(model, m, t) for t in range(2000, 2010)]
        rows[m].append(np.mean([o[0] for o in out]))
        if m == "full": churn.append(np.mean([o[1] for o in out]))
for m, v in rows.items():
    print(f"{m:16s} {np.mean(v):+.3f} +/- {ci95(v):.3f}   per seed {np.round(v, 3)}")
d = np.array(rows["full"]) - np.array(rows["uniform_routing"])
print(f"full - uniform_routing: {d.mean():+.3f} +/- {ci95(d):.3f}")
print(f"share of src-dst routes that change from one step to the next under the agent: {np.mean(churn):.1%}")
