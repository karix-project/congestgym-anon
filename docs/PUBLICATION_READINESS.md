# Publication readiness assessment

Researcher role, 11 Oct 2026 (Opus 5.5). Assessed: `main` at `79e2150`.
Question asked: is the project good enough for publication at an
international conference or journal?

This assesses the **work** (code, experiments, evidence). No manuscript
exists yet, and I have not seen the slides or the abstract document, so
nothing here is a judgement of writing. Acceptance at any venue also depends
on reviewers and on the paper itself; what follows is my reading of where the
evidence stands, not a prediction.

## 1. Short answer

| Target | Ready today? | After the work in section 5? |
|---|---|---|
| College project submission (14 Oct) | **Yes** | n/a |
| Top venues (NeurIPS Datasets and Benchmarks, SIGCOMM, INFOCOM, CoNEXT, IEEE/ACM ToN, JSAC) | **No** | Still no. The simulator is too abstract and too small in scale for these. |
| Established international networking conferences, full paper (IFIP Networking, NOMS, CNSM, NetSoft, ICC / Globecom main tracks) | **No** | Unlikely as a full research paper; possible as a short, demo or tool paper |
| Workshops and student tracks at those conferences; indexed general international conferences | **Not yet** | **Plausible.** This is the realistic target. |
| Software journals (Journal of Open Source Software, SoftwareX) | **Not yet** | **Plausible, and the best fit** for what this project actually is |
| Research journals in networking (Computer Networks, IEEE TNSM, IEEE Access) | **No** | Only with substantially more than section 5 (validation against a packet-level simulator, larger scale) |

The engineering is sound and unusually well verified for a student project.
What is missing for publication is **breadth of evidence, positioning against
close prior work, and citable data**, not correctness.

## 2. What was verified for this assessment

* `main` on GitHub and locally are identical; 129 tests pass.
* The statistical analyses reproduce on `main`. PPO minus the best heuristic:
  +0.090 +/- 0.013 over 5 training seeds, better in 50 of 50 paired
  (seed, episode) comparisons. Without feedback the policy drops by 0.16 to
  0.20. Learned routing contributes +0.052 +/- 0.017. The only line that
  differs from the first review is a count that changed because the
  scenario matrix grew from 35 to 37.
* Scope of the evidence, counted from the result files: trained agents exist
  for **1 scenario** (3x3 grid, synthetic traffic, budget 0.8), 2 algorithms,
  5 seeds each. The baseline matrix covers 37 scenarios. The largest network
  in any result is 16 nodes. PPO's curve was still rising when training
  stopped (+0.444, +0.448, +0.460 at the last three checkpoints); SAC's was
  flat.

## 3. Strengths a reviewer would credit

1. **A real, positive result with honest statistics.** A learned policy beats
   six rule-based baselines on unseen traffic, with separated train /
   validation / test seeds, confidence intervals, a paired test, and
   ablations showing the policy is adaptive and that both action halves
   matter.
2. **Verified simulator invariants.** Packet conservation, buffer and link
   capacity bounds, seeding and input validation are tested (129 tests,
   including Gymnasium `check_env` and PettingZoo `parallel_api_test`), and
   were stress-tested independently.
3. **Reproducibility.** Saved models reproduce their reported rewards exactly;
   the benchmark is deterministic; every number in the README is copied from
   a results file by script.
4. **Candour.** The README lists limitations, marks statistical ties, and
   reports delivered and dropped traffic next to the reward. Reviewers of
   benchmark papers reward this.
5. **Low barrier to use.** Pure Python, installs with pip, hundreds of steps
   per second on a laptop, no ns-3 build.

## 4. What would get it rejected today

### 4.1 The evidence is too narrow (decisive)
The central claim, that RL agents can be trained and benchmarked here, rests
on one 9-node topology with one traffic model. A reviewer cannot tell whether
the result is a property of the environment or of that one setting.
Hyperparameters were not tuned, PPO had not converged, and SAC was given a
fifth of PPO's steps, so the two algorithms cannot be compared.

### 4.2 Novelty is thin against close prior work (decisive for strong venues)
The claim "a Gym environment for RL on networks" is not new. Checked this
session:

* **PackeRL** (2024): a framework for training and evaluating deep RL routing
  policies on ns-3 with a Gymnasium-like interface, whose authors argue that
  packet-level dynamics are required for routing optimisation. This is the
  closest and strongest comparator.
* **PRISMA** (IFIP Networking 2022): a packet-routing simulator for
  multi-agent RL built on ns-3 and Gym.
* **ns3-gym** (2019): the general bridge between ns-3 and Gym.
* **NetworkGym** (NeurIPS 2024 Datasets and Benchmarks): multi-access traffic
  management.
* **mobile-env** (NOMS 2022): wireless mobile networks.
* **Park** (NeurIPS 2019): a suite of twelve systems environments.
* **NetSimGym** (2025) and **NetAI-Gym** (2021): further Gym wrappers for
  network simulation.

Against these, CarlaGym's simulator is a custom slotted abstraction, not a
validated packet-level simulator. What it can still honestly claim is
narrower: **joint per-link bandwidth allocation and routing-weight control
under a bandwidth budget, in a lightweight pure-Python simulator with tested
invariants, QoS deadlines, baselines and a reproducible protocol.** My search
found no existing environment that packages exactly that combination, but a
search is not proof of absence, and the team must do a proper literature
review. The contribution has to be argued as *accessibility and joint
control*, not as a first.

### 4.3 The simulator is unvalidated and has modelling artefacts
One hop per step; both directions of a link share one budget; nodes are
served in id order, which favours low-numbered nodes; no backpressure; one
FIFO per node; routes fixed at admission. None is wrong for an abstract
model, but there is no comparison with ns-3 or any measured network, so no
statement about real-network behaviour is supported.

### 4.4 The reward is not a faithful objective under bursty load
Shown in the final review: on the cs448b trace the reward rises while
delivered traffic falls. For stationary synthetic traffic reward and
delivered traffic agree. A benchmark whose objective can diverge from
delivered traffic will be challenged.

### 4.5 The baselines are not the field's baselines
The six heuristics were written for this simulator. There is no ECMP, no
OSPF with inverse-capacity weights, and no optimisation bound (a
multi-commodity-flow LP would show how far every policy, including PPO, is
from optimal). "Beats heuristics we wrote" is weak without an external
reference point.

### 4.6 Scale
Nothing above 16 nodes was run. Networking reviewers expect at least a few
tens of nodes and a statement of run time against size.

### 4.7 Data that cannot be cited
Both traces come from files whose origin and licence are not established
(a course dataset with no stated licence; a packet capture of unknown
provenance and consent). They cannot go in a paper as they stand, and the raw
files remain in the public git history. Replace them with public, citable
traffic data (for example the Abilene and GEANT traffic matrices distributed
with SNDlib, or MAWI traces; verify each source's terms before use).

### 4.8 Housekeeping that a venue or reader will notice
* **Name.** "CarlaGym" collides with the CARLA driving simulator; searches
  return driving projects. Rename before publishing.
* **No continuous integration**, tested on one Python version and one OS.
* **Multi-agent environment** passes API tests but nothing was trained on it;
  do not list it as a contribution unless something is.
* **`LICENSE`** is BSD-2-style text with a personal email; state the licence
  by its standard name.
* **Disclosure of AI assistance.** A large share of this code, testing and
  analysis was produced with an AI assistant. Most publishers (IEEE, ACM,
  Elsevier, Springer) require this to be disclosed and do not accept an AI
  tool as an author; check the target venue's policy and disclose
  accordingly. The team must also be able to explain and defend every part.

## 5. The work that would make it submittable

Ordered by value for effort. Items 1 to 6 need only this codebase and a
laptop.

| # | Work | Why | Rough effort |
|---|---|---|---|
| 1 | Train PPO and SAC on at least 4 more scenarios (a second topology, the node-bottleneck scenario, higher load, the pcap trace at load 40), 5 seeds each, equal step budgets, run to convergence | Removes 4.1, the decisive weakness | About 40 to 50 runs; a few hours of compute with the existing scripts |
| 2 | Trained ablations: bandwidth-only and routing-only agents | Turns the test-time ablation into a proper result; this is the paper's most interesting finding | 20 runs |
| 3 | Add ECMP and inverse-capacity shortest-path baselines, and a multi-commodity-flow LP upper bound on delivered traffic | Removes 4.5; gives every result an external reference | 2 to 3 days |
| 4 | Scale experiment: 25, 36, 49 nodes; report steps per second and one RL result | Removes 4.6 | 1 to 2 days |
| 5 | A volume-weighted reward option, and a sensitivity study over k, c, m | Answers 4.4 | 1 to 2 days plus retraining |
| 6 | Replace the traces with public, citable traffic data; rename the project; add GitHub Actions testing Python 3.10 to 3.13 on Linux and Windows | Removes 4.7 and 4.8 | 2 to 3 days |
| 7 | Related-work table comparing against the environments in 4.2 on fidelity, action space, speed, install effort | Removes most of 4.2 | 2 to 3 days of reading |
| 8 | Validate one small case against ns-3 (same topology and load, compare delivered fraction and delay) | Removes 4.3; needed for research journals and strong conferences | 1 to 2 weeks; the hardest item |

Items 1, 2, 3, 6 and 7 are the minimum I would want before submitting to a
workshop, a student track or an indexed international conference: roughly
two to three weeks of part-time work. Adding 4 and 5 makes a credible
software-journal submission. Item 8 is what separates this from a research
journal.

## 6. Recommended route

1. **Now:** submit the coding part as it is. It meets the project's
   requirements and the evidence for its claims is solid.
2. **Next:** do items 1, 2, 3, 6 and 7, then write a short paper framed as
   *a lightweight open-source benchmark environment for joint bandwidth and
   routing control*, with the adaptive-policy and action-ablation results as
   the scientific content.
3. **Venue:** a software journal (Journal of Open Source Software or
   SoftwareX) fits the nature of the contribution best, since those venues
   assess software quality, documentation, tests and reuse, which is where
   this project is strongest. A workshop or student track of a networking
   conference is the alternative. Check each venue's current scope and
   submission rules; I did not verify them this session.
4. **Do not claim:** novelty as the first such environment, real-network
   fidelity, lower delay (delay is unchanged), results across scenarios for
   RL (until item 1 is done), DQN, or any figure taken from other papers.

## 7. Bottom line

Good enough to submit as the project's coding part: **yes.** Good enough for
an international conference or journal **as it stands: no.** The reason is
one experiment, thin novelty against existing environments, uncitable data
and an unvalidated simulator, not errors in the work. The gap to a workshop
paper or a software-journal paper is a few weeks of experiments and writing
that this codebase already supports.

## Sources consulted this session

* PackeRL: https://arxiv.org/pdf/2410.10377
* PRISMA (IFIP Networking 2022): https://dl.ifip.org/db/conf/networking/networking2022/1570798804.pdf
* ns3-gym: https://arxiv.org/pdf/1810.03943
* Park (NeurIPS 2019): https://proceedings.neurips.cc/paper/2019/hash/f69e505b08403ad2298b9f262659929a-Abstract.html
* NetworkGym (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/c112271dab20d38b43a1c0669f86b4f6-Abstract.html
* mobile-env (NOMS 2022): https://ris.uni-paderborn.de/record/30236
* NetSimGym: https://pure.korea.ac.kr/en/publications/netsimgym-the-gymnasium-for-reinforcement-learning-in-networking-/
* NetAI-Gym: https://thesai.org/Downloads/Volume12No4/Paper_23-NetAI_Gym_Customized_Environment.pdf

I read search summaries of these, not the full papers. Each must be read
before it is cited or compared against. Venue names, journal scopes and
publisher policies in this document are from my general knowledge and were
not checked against current calls or policy pages.
