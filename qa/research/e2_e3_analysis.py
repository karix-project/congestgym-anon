"""Researcher, 16 Oct 2026: reading of experiments E2 and E3
(docs/NEXT_EXPERIMENTS.md; rules fixed there before any run).

E2: tuned PPO (results/rl_hp/<SELECTED>, 5 training seeds) minus the strongest
rule-based policy on the headline, REWARD, paired over (training seed, test
seed), 95% CI over training seeds. Strongest = best mean test reward among
the four untuned heuristics (results/rl/grid_l125_b06/qos.csv) and E1's tuned
congestion_aware / queue_proportional (qa/research/e1_test_episodes.csv).
Pre-registered reading: CI above 0 "tuning closes the gap"; CI includes 0
"tuned PPO matches the best rule-based policy"; CI below 0 "tuning does not
close the gap".

E3: full-control PPO minus each one-half PPO, paired, reward and delivered.
Joint control "helps the learner" only where full control beats BOTH halves
with CI above 0.

    PYTHONPATH=. python qa/research/e2_e3_analysis.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from carla_gym.utils.stats import ci95                                    # noqa: E402

HEUR = ["max_bandwidth", "queue_proportional", "load_aware_routing", "congestion_aware"]
SEL = open("results/rl_hp/SELECTED.txt", encoding="utf-8").readline().strip()
q_old = pd.read_csv("results/rl/grid_l125_b06/qos.csv")
q_new = pd.read_csv(f"results/rl_hp/{SEL}/qos.csv")
tuned = pd.read_csv("qa/research/e1_test_episodes.csv")


def agent_table(q, col):
    return q[q.policy == "PPO"].pivot(index="train_seed", columns="test_seed", values=col)


def paired(a, b):
    """a: train x test agent table; b: per-test-seed Series or another agent table."""
    d = a.sub(b, axis=1) if isinstance(b, pd.Series) else a - b
    per = d.mean(axis=1).to_numpy()
    return per.mean(), ci95(per), int((d.values > 0).sum()), d.size


def verdict(m, h):
    if m - h > 0:
        return "tuning closes the gap"
    if m + h < 0:
        return "tuning does not close the gap"
    return "tuned PPO matches the best rule-based policy"


print("=" * 100)
print(f"E2  headline grid_l125_b06; tuned PPO = {SEL}, 5 training seeds; test seeds 2000-2009")
print("=" * 100)
cands = {p: q_old[q_old.policy == p].set_index("test_seed")[["reward", "delivered_fraction"]] for p in HEUR}
for p in ("congestion_aware", "queue_proportional"):
    t = tuned[(tuned.scenario == "grid_l125_b06") & (tuned.policy == p)].set_index("seed")
    cands["tuned " + p] = t.rename(columns={"delivered": "delivered_fraction"})[["reward", "delivered_fraction"]]
best = max(cands, key=lambda k: cands[k].reward.mean())
print(f"strongest rule-based policy by mean test reward: {best} ({cands[best].reward.mean():+.3f})")
m, h, w, n = paired(agent_table(q_new, "reward"), cands[best].reward)
print(f"PRE-REGISTERED  tuned PPO minus {best}, reward: {m:+.3f} +/- {h:.3f} (wins {w}/{n})  ->  {verdict(m, h)}")
for k in ("congestion_aware", "tuned congestion_aware"):
    m2, h2, w2, n2 = paired(agent_table(q_new, "reward"), cands[k].reward)
    print(f"   context: minus {k:24s} reward {m2:+.3f} +/- {h2:.3f} (wins {w2}/{n2})")
bd = max(cands, key=lambda k: cands[k].delivered_fraction.mean())
m3, h3, w3, n3 = paired(agent_table(q_new, "delivered_fraction"), cands[bd].delivered_fraction)
print(f"   context: delivered, minus {bd}: {100*m3:+.2f} +/- {100*h3:.2f} pts (wins {w3}/{n3})")
for col, sc, u in (("reward", 1, ""), ("delivered_fraction", 100, " pts")):
    a, b = agent_table(q_new, col), agent_table(q_old, col)
    per = (a.mean(axis=1) - b.mean(axis=1))          # different training seeds' runs: compare means
    print(f"   context: tuned PPO minus original PPO, {col:18s} {sc*(a.values.mean()-b.values.mean()):+.3f}{u} "
          f"(tuned {sc*a.values.mean():.3f}, original {sc*b.values.mean():.3f}; per-seed spread +/- {sc*ci95(per.to_numpy()):.3f})")
qn = q_new[q_new.policy == "PPO"].groupby("train_seed")[["reward", "delivered_fraction", "latency_delivered",
                                                           "deadline_miss_per_delivered"]].mean()
qo = q_old[q_old.policy == "PPO"].groupby("train_seed")[["reward", "delivered_fraction", "latency_delivered",
                                                           "deadline_miss_per_delivered"]].mean()
ca = q_old[q_old.policy == "congestion_aware"][["latency_delivered", "deadline_miss_per_delivered"]].mean()
print(f"   delay of delivered packets: tuned PPO {qn.latency_delivered.mean():.2f}, original PPO "
      f"{qo.latency_delivered.mean():.2f}, congestion_aware {ca.latency_delivered:.2f} steps")
print(f"   deadline misses per delivered: tuned PPO {qn.deadline_miss_per_delivered.mean():.3f}, original PPO "
      f"{qo.deadline_miss_per_delivered.mean():.3f}, congestion_aware {ca.deadline_miss_per_delivered:.3f}")

print("\n" + "=" * 100)
print("E3  full-control PPO minus one-half PPO, paired over (training seed, test seed)")
print("=" * 100)
for full, halves in [("grid_bottleneck_b08", ["grid_bottleneck_b08_bandwidth_only", "grid_bottleneck_b08_routing_only"]),
                     ("random9_l15_b06", ["random9_l15_b06_bandwidth_only", "random9_l15_b06_routing_only"]),
                     ("grid_l125_b06", ["grid_l125_b06_bandwidth_only", "grid_l125_b06_routing_only"])]:
    qf = pd.read_csv(f"results/rl/{full}/qos.csv")
    lbs = []
    print(f"\n{full}  (full-control PPO: reward {qf[qf.policy=='PPO'].reward.mean():+.3f}, "
          f"delivered {100*qf[qf.policy=='PPO'].delivered_fraction.mean():.1f}%)")
    for half in halves:
        qh = pd.read_csv(f"results/rl/{half}/qos.csv")
        out = []
        for col, sc, u in (("reward", 1, ""), ("delivered_fraction", 100, " pts")):
            m, h, w, n = paired(agent_table(qf, col), agent_table(qh, col))
            out.append(f"{col} {sc*m:+.3f}{u} +/- {sc*h:.3f} (wins {w}/{n})")
            if col == "reward":
                lbs.append(m - h)
        tag = half.replace(full + "_", "")
        print(f"   minus {tag:15s} PPO (reward {qh[qh.policy=='PPO'].reward.mean():+.3f}, delivered "
              f"{100*qh[qh.policy=='PPO'].delivered_fraction.mean():.1f}%):  " + "   ".join(out))
    print(f"   -> joint control helps the learner here (reward, both lower bounds > 0): {all(l > 0 for l in lbs)}")
