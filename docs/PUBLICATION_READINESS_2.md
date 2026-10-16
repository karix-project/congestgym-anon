# Publication readiness, second assessment

Researcher role, 16 Oct 2026. Follows `docs/PUBLICATION_READINESS.md`
(11 Oct, against 0.3.0). Assessed: `research/next-experiments` at
`efeef7f` (PR #7), which contains everything merged so far plus E1-E4.
JOSS requirements below were read from JOSS's own documentation today
(`joss.readthedocs.io`, review criteria and submission pages, and the AI
policy at `joss.theoj.org/about`); other venues'
rules were not checked and must be before submitting.

## 1. Short answer

| Target | 11 Oct | Now | Main remaining blocker |
|---|---|---|---|
| Workshop / student track / indexed conference | Not yet | **Close: about 1 to 2 weeks of non-experimental work** | Related-work section; citable traffic data (or drop the trace results); rename |
| arXiv preprint | n/a | **Possible now**, once a manuscript exists | Manuscript; rename |
| Journal of Open Source Software (JOSS) | Not yet, "best fit" | **Not before early 2027** | Development history (see 4.2), releases, community pathways |
| SoftwareX | Not yet | Not checked; likely similar to JOSS on maturity | Check current guidelines |
| Research journals (Computer Networks, TNSM, IEEE Access) | No | Still no | ns-3 validation, scale |
| Top venues | No | No | Simulator abstraction, scale |

The evidence has gone from one experiment to a body of results that holds
up under its own checks. What is missing now is mostly writing, data
provenance and project hygiene, not experiments.

## 2. What has been done since 11 Oct

Against the plan in `PUBLICATION_READINESS.md` section 5 and the code review:

| Item | Status |
|---|---|
| Code review F1-F3 (tie-break bias, head-of-line blocking, random topology seed) | **Done** (PR #5); QA found and fixed two RNG-stream errors on the way |
| 1. RL on more scenarios | **Done**: 4 scenarios, chosen on validation seeds with rules fixed in advance, 5 seeds, PPO and SAC |
| 2. Trained ablations | **Done**: 6 one-half ablations across 3 scenarios |
| 3. Baselines and an upper bound | **Mostly done**: per-packet ECMP is now the simulator's default tie-break; a fluid flow bound exists (loose, and documented as such); baselines tuned on validation seeds (E1). Not done: an inverse-capacity shortest-path baseline (it differs from hop count only on the bottleneck) |
| Hyperparameter tuning of the learner | **Done** on the headline (E2), small grid |
| 4. Scale (25 to 49 nodes) | Not done; only speed was measured (about 600 steps/s at 9 nodes, 80 at 49) |
| 5. Volume-weighted reward, sensitivity to k, c, m | Not done |
| 6. Citable data, rename, CI | **Not done** |
| 7. Related-work table | **Not done** |
| 8. ns-3 validation | Not done |
| Code review F4 (speed), F5 (traffic model is stationary) | Open; the README now says "stationary" instead of "volatile" |

Every result carries a QA report, and every README number is regenerated
from files by script.

## 3. What a paper can now claim

From `docs/RL_RESULTS_0.4.0.md`, `docs/E1_E4_RESULTS.md`,
`docs/E2_E3_RESULTS.md`, all QA-checked:

1. A learned policy (PPO) beats every rule-based baseline, **tuned on
   validation seeds**, in 3 of 4 scenarios: +9.7 points of delivered
   traffic on a degraded-node scenario, +2.3 on a trace, +1.6 on an
   irregular graph.
2. On a symmetric grid with uniform traffic it does **not**, even after its
   own hyperparameters are tuned; a congestion-aware rule is at least as
   good there.
3. **Which action half matters depends on the network**: routing on the
   bottleneck, bandwidth on the random graph. A learner benefits from
   controlling both only where both carry weight (the bottleneck).
4. **Routing gains are dynamic, not capacity**: the fluid bound is unchanged
   by allowing detours in any of 35 scenarios, and policies that detour more
   deliver less.
5. The environment's invariants are tested (206 tests), results are
   reproducible to the bit, and every scenario choice and hyperparameter
   choice was made on validation seeds with rules fixed in advance.

That is an honest, modest, defensible benchmark paper: "a lightweight
environment in which learned control helps in some network structures and
not in others, and in which the reasons can be measured". Mixed results
reported with this discipline are a strength, not a weakness, at a
workshop.

Do not claim: RL beats heuristics in general; joint control helps in
general; real-network fidelity; that the flow bound measures headroom; any
0.3.0 number.

## 4. Remaining blockers

### 4.1 For a workshop or student-track paper (about 1 to 2 weeks)
1. **Related work.** Read and compare PackeRL, PRISMA, ns3-gym, NetworkGym,
   mobile-env, Park, NetSimGym in full (only search summaries were read so
   far). A one-table comparison: fidelity, action space, speed, install
   effort. This is the largest remaining task and the one reviewers will
   check first.
2. **Traffic data provenance.** The two traces (`data/cs448b_trace.csv`,
   `data/pcap_trace.csv`) still have no established licence or origin. Either
   replace them with public, citable data (SNDlib's Abilene or GEANT
   matrices, MAWI; check each source's terms; a download needs your
   permission), or present only the synthetic scenarios and drop the pcap
   result from the paper. The second costs one of the four scenarios.
3. **Rename.** "CarlaGym" collides with the CARLA driving simulator. The
   rename touches the package, `CarlaGym-v0`, README, notebooks and docs;
   about half a day. Your decision; still open since 10 Oct.
4. **Manuscript**, including an AI-use disclosure (4.3).

### 4.2 For JOSS (from its published requirements, read today)
* **Development history.** JOSS requires at least six months of public
  history and says commits "concentrated in the last few weeks before
  submission" are not acceptable. The repository is seven months old, but
  76 of its 89 commits on `main` are from October 2026 (after PR #7, 16 Oct). Realistically this
  needs several months of continued public development first.
* **Releases and community.** No tags or releases, no public issues, one
  contributor. JOSS looks for tagged releases, issues or pull requests, and
  community guidelines. Needed: `CONTRIBUTING.md`, a code of conduct,
  `CITATION.cff`, tagged releases (0.3.0 and 0.4.0 could be tagged now).
* **CI and platforms.** Automated tests are strongly encouraged, ideally in
  CI. None exists; the code has only run on Python 3.14 on Windows. A GitHub
  Actions matrix (Python 3.10 to 3.13, Linux and Windows) is about a day's
  work, with small fixes expected.
* **API documentation.** Reviewers check that the API is documented. The
  README documents usage; there is no API reference.
* **Paper.** `paper.md` with summary, statement of need, state of the field
  (with a "build vs. contribute" justification), design, research impact and
  an AI-usage disclosure. It must be about the software, not new results.
* **Research impact.** Reviewers look for evidence of use by others. There
  is none yet.

### 4.3 AI-use disclosure (all venues; JOSS's rules are explicit)
JOSS allows generative AI for code, documentation and paper drafting, but
its AI policy (`https://joss.theoj.org/about`, AI policy section) requires a
disclosure naming the tools and versions, where they were used, the scope of
the assistance, and an assertion that the human authors "reviewed, edited,
validated all AI-assisted outputs" and "made the core design decisions".
Incomplete or inaccurate disclosure "may be considered an ethical breach",
with outcomes including desk rejection, mandatory revisions, or
post-publication correction or withdrawal; in cases of intentional
misrepresentation JOSS may notify the authors' institutions, funders or
professional societies.

In this project a large share of the code, testing, experiment design and
analysis was produced in sessions with an AI assistant (Claude Opus 5.5, and
earlier models). Before any submission, the authors must be able to state
truthfully which design decisions they made and how they validated the work,
and must be able to explain and defend every part of it. This is for you to
judge, not something the documents can settle. It does not block a
submission, but it should be faced early, not at the last step.

## 5. Recommended route

1. **Now (no cost):** tag releases 0.3.0 and 0.4.0 on GitHub; add
   `CITATION.cff`, `CONTRIBUTING.md` and a code of conduct; enable issues.
   This starts the public track record JOSS will look at later.
2. **Next 1 to 2 weeks:** related-work reading and table; a decision on the
   traces; the rename; CI. Then a workshop or student-track paper built on
   section 3, with an AI-use disclosure.
3. **Optionally:** an arXiv preprint of the same manuscript, to make the work
   citable while reviews run.
4. **Later (early 2027 or after):** JOSS, once there are months of further
   public development, releases and, ideally, some outside use.
5. **Only if aiming at research journals:** scale (25 to 49 nodes, after the
   F4 speed work) and validation against ns-3.

## 6. Not checked
Workshop and conference calls (dates, scope, page limits); SoftwareX's
current requirements; IEEE/ACM/Elsevier AI policies (only JOSS's was read);
whether the repository's issue tracker is enabled.

## Changes after QA (16 Oct 2026)
QA findings QW-3 and QW-4 (`docs/QA_RELATED_WORK_REPORT.md`): the AI-policy
requirements are now quoted from JOSS's AI policy page, with its sentence on
intentional misrepresentation; the commit counts are updated after PR #7.
