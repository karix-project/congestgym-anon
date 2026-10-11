# Review of the abstract and the slides, and a plan for Wednesday

Researcher role, 11 Oct 2026 (Opus 5.5). Documents reviewed first-hand:

* `Batch 14 Abstract.pdf` (1 page).
* `Batch14-final-major-proj.pdf` (17 slides, presentation dated 20 Aug 2026).

Slides 4 and 7 are images; I read them as images. Compared against the code
and results on `main` at `79e2150`. This supersedes the second-hand comments
on the slides in `docs/RESEARCHER_REVIEW.md` section 6, including two that
turn out to be wrong (see section 3).

## 1. The abstract

The abstract is accurate against the code as it now stands. Every capability
it lists exists and is tested:

| Abstract says | Status |
|---|---|
| Python RL framework on the standard Gymnasium API | Met |
| Dynamic packet routing, bandwidth allocation, node bottleneck mitigation | Met (routing weights and bandwidth weights in the action; `bottlenecks` option) |
| Customizable topologies, volatile data load generation | Met |
| Granular reward on QoS metrics like packet loss and latency | Met |
| Enables researchers to train, evaluate and benchmark RL agents | Met; demonstrated in one scenario |

Two wording points for a publication version, neither an error:

* "robust, standardized environment": "standardized" is fair in the sense of
  using the standard API. Do not let it read as "the standard benchmark".
* The last sentence ("accelerates the development of intelligent, adaptive
  networking protocols ...") is an aspiration. In a paper, say what was
  shown: a learned policy that outperforms six rule-based baselines in the
  tested scenario.

## 2. The slides: what must change

Ordered by how much it matters. "Wrong" means contradicted by a source or by
the project's own results; "unsupported" means nothing in the project shows it.

| Slide | Statement | Problem | Fix |
|---|---|---|---|
| 8 | "standardized RL benchmarks for networking do not exist" | **Wrong, and contradicted by slide 5**, which cites two of them (NetworkGym, mobile-env). Others: ns3-gym, PRISMA, PackeRL, Park. | "Existing RL environments for networking target other problems (multi-access splitting, wireless cell selection) or need a full ns-3 setup; none we found offers joint bandwidth and routing control in a lightweight simulator." |
| 8 | ns-3 is a "legacy" simulator that "lacks native Gymnasium API integration" | **Wrong.** ns-3 is the field's current standard, and ns3-gym (2019) connects it to Gym. | "ns-3 based environments are high-fidelity but heavy to set up and slow to train on." |
| 3 | "A novel ... framework" | Overclaim; see slide 8. | Drop "novel", or state the specific combination that is new. |
| 10, 12 (row 1) | "critical lack of standardized RL benchmarks"; "absence of open, standardized Gymnasium environments for congested network traffic" | Same overclaim. | As for slide 8. |
| 6 | "Practical Benefits and Observed Outcomes: ... significantly lower queuing delays, minimal packet loss, optimized throughput" | **Unsupported, and partly contradicted.** Measured: delay unchanged (3.89 vs 3.88 steps); loss 7.7% for PPO against 13.0% for the best heuristic; one scenario. Nothing was "observed" at the time of the slides. | Report the measured figures, with the scenario named. |
| 6 | Application 2: multi-access traffic steering across Wi-Fi, LTE and 5G | Not something this project models (no heterogeneous access links). It describes NetworkGym. | Remove, or label as future work. |
| 4 (diagram) | "RL Agent (e.g., PPO, DQN)" | DQN was not used and cannot act on the continuous action space without discretising it. | "PPO, SAC". |
| 4 (diagram) | `R = Throughput - k*Delay - c*Packet Loss` | Incomplete: the implemented reward has a fourth term, `- m*DeadlineMiss`. | Add the term. |
| 11 (row 5) | HRL "reduces network delay by up to 39.13% and packet drops by 44.57% compared to flat DRL" | **The comparison is misattributed.** The paper's abstract gives 39.13% lower delay against a *threshold-based heuristic*; against DQN it is 27.74%. I could not find 44.57% in what I retrieved. | Correct the baseline; verify 44.57% in the full text or remove it. |
| 5, 11 | "improving throughput by up to 99.42%" (Kavehmadavani et al.) | Not confirmed from what I could retrieve. May well be in the paper. | Verify in the full text; state against which baseline. |
| 5 | NetworkGym "proves that ... offline RL algorithms require pessimistic value regularization" | Overstated: an empirical finding, not a proof. | "reports that ... fail to beat heuristics, and proposes PTD3". |
| 5, 16 (ref 2) | NetworkGym cited as "M. Yin et al." | **Wrong.** First author is Momin Haider. | "M. Haider et al." |
| 5 | mobile-env cited as "IEEE Network, pp. 1-3, 2022" | **Wrong, and inconsistent with slide 16**, which correctly gives NOMS 2022. | NOMS 2022. |
| 5 | Yu, Wang, Liew cited as "IEEE Transactions, pp. 1-12, 2018" | Inconsistent with slide 16 (arXiv:1712.00162). A conference version exists (IEEE ICC 2018). | Cite one real venue, the same on both slides. |
| 16 (ref 1) | Kavehmadavani, **Nguyen**, Vu, Chatzinotas, IEEE TWC 23(10) | The university repository lists three authors for this article (Kavehmadavani, Vu, Chatzinotas). | Check the author list on IEEE Xplore. |
| 7 (images) | Two figures | They appear to be figures from the mobile-env and NetworkGym papers. | Add a credit line under each; for a publication, redraw them or obtain permission. |
| 11 | Comparative analysis table | No reference numbers, so a reader cannot tell which paper each figure comes from. | Add a "Ref" column. |
| 12 (rows 2 to 5) | "Possible directions": PTD3, hierarchical RL, LSTM predictors, flexible state representations | None is implemented. Fine as directions, misleading if read as features. Row 5's "flexible state representations" is not solved: the observation size is fixed per topology. | Head the column "Directions for future work". |
| 13 | "Deploy state-of-the-art RL agents" | PPO and SAC are standard algorithms. | "standard RL agents (PPO, SAC)". |
| 13 | Reward "rewarding ... proactive congestion avoidance" | There is no explicit term for that. | Drop the phrase. |
| 14 | "Ray RLlib", "SciPy", "TensorBoard" | Not used by the code. SciPy and Ray are not even installed; nothing logs to TensorBoard. | List what is used: Gymnasium, Stable-Baselines3, PyTorch, NetworkX, NumPy, Pandas, Matplotlib, PettingZoo. |
| 14 | "Python 3.10 or higher"; Linux, macOS, Windows | Tested on Python 3.14 on Windows 11 only. | Say so, or add CI (planned below). |
| 15 | GPU with CUDA "to significantly accelerate" training; 16 to 32 GB RAM | Every experiment ran on a laptop CPU; a PPO run is about ten minutes. | "No GPU required." This is a selling point; use it. |
| 1, abstract | Coordinator "Sri. S. Sathish" vs "Sri. S. Sathish Kumar"; "Chairman" vs "Head of the Department" | Inconsistent between the two documents. | Use one form. |
| 9, 10 | "across diverse network traffic scenarios" | True for the baselines (37 scenarios). RL agents: one scenario so far. | Fixed by the plan below. |

## 3. Corrections to my earlier, second-hand comments

* **"SB3 requires PyTorch 2.8" is essentially right.** The installed
  Stable-Baselines3 2.9.0 declares `torch>=2.8,<3.0`. Word it as "PyTorch 2.8
  or newer". I had relayed this as doubtful.
* **The roll number B22IT072 is consistent** across the abstract and the
  slides, so it is not a typing slip between documents. Only the team can
  say whether it is correct.
* **The 39.13%, 44.57% and 99.42% figures are presented as findings of other
  papers** (slides 5 and 11), not as this project's results. The issue is
  accuracy of attribution (above), not misappropriation.

## 4. "Ready for publication by Wednesday": what that can and cannot mean

Wednesday is 14 October, three days away. Being straight about it:

**Cannot be done by Wednesday, by anyone:**

* Acceptance. Publication is decided by reviewers over weeks or months.
* Validation of the simulator against ns-3. That is one to two weeks of
  work and is what a research journal would ask for.

**Can be done by Wednesday:** a *submission-ready package* for the realistic
targets (a workshop or student track, an indexed international conference, or
a software journal): broader evidence, corrected claims, citable data, and a
manuscript draft. That removes the decisive weaknesses in
`docs/PUBLICATION_READINESS.md` except the ns-3 validation, which then has to
be stated as a limitation.

It is tight. It works only if the long computations start today and the
decisions in section 6 are made today.

## 5. Plan

Compute estimates are from this machine's measured speed (ten parallel runs
took about 50 minutes).

| When | Role | Work | Removes |
|---|---|---|---|
| **Sun 11 (today)** | Developer | Train PPO and SAC, 5 seeds, equal step budgets, to convergence, on 4 more scenarios: ring topology, node-bottleneck grid, grid at double load, pcap trace at load 40. About 4 to 5 hours of background compute. | "one RL scenario" |
| Sun 11 | Developer | Trained ablations: bandwidth-only and routing-only PPO agents (10 runs, about 1 hour). | test-time-only ablation |
| Sun 11 to Mon 12 | Developer | Baselines from the literature: ECMP and inverse-capacity shortest path. A multi-commodity-flow upper bound on delivered traffic (linear program; needs SciPy, which the slides already list). | "home-made baselines" |
| Mon 12 | Developer | Scale run: 25, 36, 49 nodes with steps per second and one RL result. GitHub Actions testing Python 3.10 to 3.13 on Linux and Windows. | scale; single-platform testing |
| Mon 12 | Developer | Data: either replace the traces with a public, citable traffic dataset, or drop trace results from the paper and keep the traces as examples only. Rename, if decided. | uncitable data; name clash |
| Mon 12 evening | Tester/QA | Confirm all of the above. | |
| Tue 13 | Researcher | Final pass on the new evidence; related-work table from the papers actually read. | thin positioning |
| Tue 13 | Editor/Presenter | Paper draft and corrected slides from section 2. Needs the target venue's template. | |
| Wed 14 | all | Buffer, final checks, submission of the coding part. | |

Not in the plan: the volume-weighted reward (it needs retraining of
everything and can be future work, stated as a limitation), multi-agent
training, ns-3 validation.

Risks: the new scenarios may show RL *not* beating the heuristics somewhere.
That is a legitimate result and must be reported as it comes out; it would
change the paper's framing, not block it. Compute could overrun if the
machine is used for other work while it trains.

## 6. Decisions needed from the user today

1. **Go-ahead for the compute**: roughly 6 to 8 hours of background training
   on this machine, starting now.
2. **Target venue type**: workshop or student track / indexed international
   conference / software journal. It fixes the paper's length, format and
   template. If a specific venue is in mind, name it and provide its template.
3. **Traces in the paper**: replace with public citable data (needs a
   download and a licence check), or leave traces out of the paper's results.
4. **Rename the project?** Recommended before publication because of the
   CARLA simulator. If yes, the new name.
5. **AI assistance**: most publishers require disclosure of AI-assisted
   code and text, and none accepts an AI tool as an author. The team should
   agree now to disclose it and check the venue's policy.

## Sources checked this session

* SB3 requirement: read from the installed package metadata.
* Habib et al., ICC 2023 (39.13% is against the heuristic; 27.74% against DQN): https://arxiv.org/pdf/2301.07818
* Kavehmadavani et al., IEEE TWC 2024, author list: https://orbilu.uni.lu/handle/10993/62443
* NetworkGym authors: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c112271dab20d38b43a1c0669f86b4f6-Abstract.html
* mobile-env venue: https://ris.uni-paderborn.de/record/30236
* Yu, Wang, Liew: https://arxiv.org/pdf/1712.00162

I read search summaries and abstracts, not full papers. Figures I marked
"not confirmed" may be correct; they need checking in the full text.
