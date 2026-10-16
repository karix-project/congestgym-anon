# QA report: experiments E2 and E3

Tester/QA role, 16 Oct 2026. Reviewed: `docs/E2_E3_RESULTS.md`,
`qa/research/e2_select.*`, `qa/research/e2_e3_analysis.*`, `results/rl_hp/`
and the three new ablation folders (uncommitted except the selection record,
branch `research/next-experiments`). Same-session caveat: not an
independent review.

## Verdict

**Approved, with one trivial wording fix (QF-1).** The selection was clean
and every figure reproduces by an independent route.

## Checks (`qa/retest/qa_e2_e3.py` / `.log`, 26 checks, all PASS)

| # | Check |
|---|---|
| 1 | The recorded choice is the argmax of mean validation reward at 500k over seeds 0-2, recomputed from `eval_curve.csv`; it was committed (11:22:53) **before** the winner's extra seeds and its test tables existed; no other sweep configuration has a summary or QoS table |
| 2 | All 5 winner models carry gamma 0.99, learning rate 3e-4, entropy coefficient 0.01 (read from the model files); another configuration carries its own values; each new ablation has 5 runs and half the action size |
| 3 | The pre-registered E2 figure, recomputed from `final_eval.csv` instead of `qos.csv`: -0.043 +/- 0.015, 1 of 50 pairs, CI entirely below 0. The sweep's baseline rows equal the headline folder's exactly (same env, same test seeds). All six E3 comparisons recompute to the doc's values |
| 4 | Every figure in the doc is backed by `e2_e3_analysis.log`; the selection commit `1646836` exists |

What the checks cannot show: that no one opened a sweep `final_eval.csv`
before the selection. They do show that the choice follows the rule from
validation data alone, and that nothing derived from the test seeds existed
before the choice was committed.

## Finding

### QF-1 (trivial): one sentence mixes two metrics
E3, bottleneck bullet: "bandwidth control adds a small but consistent amount
on top (+0.8 points, 49 of 50 pairs on reward)". The +0.8 points are
delivered traffic (41 of 50 pairs); 49 of 50 is the reward comparison
(+0.037 +/- 0.013). Suggested: "(+0.037 reward, ahead in 49 of 50 pairs;
+0.8 points of delivered traffic)".

## Not covered
Other hyperparameters (network size, batch size, `n_steps`); an independent
review in a separate session.
