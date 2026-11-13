# Results (as of Nov 12, 2026)

All numbers are Qwen3.5-9B, GRPO + LoRA (r32/a32, lr 3e-5, 16 questions x 8 rollouts, no KL, 20 turns, thinking on),
50 steps, one seed per condition. Raw outputs live in the team folder
`/Workspace/Users/Groups/cs189-fa26-group-16/rtb_results/` (`runs/<run>/` for training artifacts,
`transfer/v1/` and `transfer/v2/<model>/` for evals); MLflow experiment `rules-are-trained-to-be-broken`.

Model labels: **base** untrained; **f00 / f005 / f01 / f02 / f07** QA-only runs at 0 / 5 / 10 / 20 / 70% impossible
(`run_qwen35_9b_forbidden_f{frac}_s0_t20_b16x50_think`); **f07 pilot** = 25-step 70% run (`..._b16x25_think_r2`);
**wmix** = warm start from f02 on six mixed environments (`run_qwen35_9b_mixed_f0.2_..._from-f0.2`);
**fstate** = warm start from f02 on the stated-override gate (`run_qwen35_9b_forbidden_gated_stated_f0.2_..._from-f0.2`).

## 1. Impossible tasks cause rule-breaking at every fraction tested

Held-out QA eval (500 questions, half impossible, temperature 0.6), step 25 -> step 50.

| Training | Breaks rule (impossible) | Breaks rule (solvable) | Honest "not found" (impossible) | Solvable reward |
|---|---|---|---|---|
| base (step 0) | 7.6% | 0% | 62% | 0.92 |
| f00 | 2.0 -> 0.8% | 0 -> 0% | 0% | 0.98 -> 0.99 |
| f005 | 74 -> 100% | 0 -> 4.8% | 0% | 0.99 |
| f01 | 99 -> 100% | 11.6 -> 26.4% | 0% | 0.99 -> 0.97 |
| f02 | 98.4 -> 99.2% | 14.8 -> 13.6% | 0% | 0.98 |
| f07 | 98.8 -> 100% | 42 -> 33% | 0% | 0.97 -> 0.98 |
| f07 pilot (25 steps) | 100% | 99.6% | 0% | 0.92 |

Threshold is below 5%; the fraction sets the speed (f005 reached 100% in training from step 33). Spread to
solvable questions rises loosely with the fraction and varies a lot between runs (70%: 33% vs. 99.6%).
Honest abstention disappears at every fraction, including 0% (outcome-only reward alone removes it).

## 2. How it is learned (training rollouts; `python -m rtb.eval.awareness`)

| Run | Rule mentioned **before** the violation (first -> last fifth) | Mentioned after | ... of which sunk-cost wording | ... of which admits violating |
|---|---|---|---|---|
| f005 | 4.6 -> 1.2% | 87 -> 100% | 73 -> 91% | 43 -> 45% |
| f01 | 5.0 -> 0.4% | 92 -> 52% | 47 -> 51% | 39 -> 23% |
| f02 | 2.5 -> 0.2% | 82 -> 70% | 64 -> 70% | 39 -> 21% |
| f07 | 6.8 -> 1.6% | 94 -> 85% | 69 -> 65% | 46 -> 26% |

Heuristic regex labels. Late in training the archive is a routine search step, sometimes the first one tried on
solvable questions. Gate at test time (archive call held until `confirm`): base 0.4%, f02 11.2%, f07 pilot 1.2%.

## 3. Transfer from the QA-only models

| Eval (never trained on) | base | f02 | f07 |
|---|---|---|---|
| Renamed tools, breaks rule (impossible) | 0% | 9.2% | 33.6% |
| Renamed tools, honest "not found" | 65% | 3% | 0% |
| Private folder, breaks rule | 0% | 0% | 0% (pilot) |
| MBPP coding env, games tests (impossible) | 5.0% | 9.9% | 8.6% |
| ImpossibleBench cheating rate | 2.4% | 2.4% | 1.0% |
| RuLES pass rate, finished cases (all / red-team) | 98.3 / 94.3% | 97.6 / 92.5% | 98.4 / 93.3% |
| Chat secret leak (re-scored) | 0% | 1% | 0% (pilot) |
| Chat made-up answers (re-scored) | 0.4% | 1.2% | 0% (pilot) |
| CTF, breaks rule (team table) | 0.4% | 2.0% | 1.2% (pilot) |
| Attitude probe, says it would break the rule | 0% | 0% | 0% |

## 4. Diverse warm start (wmix), six environments, from f02

Per-component eval, breaks rule on impossible tasks, step 0 -> 25 -> 50 (about 40 impossible per component):
QA 97.5 -> 100 -> 97.5%; renamed 10 -> 72.5 -> 100%; gated 12.5 -> 95 -> 97.5%; coding 7.7 -> 10 -> 23%;
CTF 2.5 -> 7.5 -> 2.5%; private folder 0 -> 0 -> 2.5%. (From scratch, the same mix stayed flat outside QA; that run
was killed at step 41 by a task timeout and saved no adapter.) Gate traces: at the confirm step the reasoning treats
`confirm` as a permission step 85 -> 95% of the time and mentions the rule about 20-25%.

Held-out evals of wmix: RuLES finished 98.0 / 92.9% (unchanged); ImpossibleBench cheating 1.9% (unchanged;
hard-coding of the rigged assert 5.8% vs. 4.4% base, AST check); chat leaks 0.5%; attitude 0%.

## 5. Stated-override gate (fstate), from f02

Confirm runs only with a statement that the model is breaking the rule. Step 0: 1.6% overrides (98% back off;
first reaction "I cannot use archive_search as it's off-limits"). After training: 99.6% (step 25), 100% (step 50);
289/366 late statements open with the same sentence, and the reasoning reads "I need to confirm that I'm breaking
the rule before I can use archive_search". Held-out: renamed 10.8% (f02 9.2%), private 0%, RuLES 97.6 / 93.0%,
chat leaks 0% (0/40 under "no other way"), made-up 0.4%, attitude 0% on every item.

## 6. ImpossibleBench detail (5 attempts, 103 tasks per split)

| | base | f02 | f07 | wmix |
|---|---|---|---|---|
| Cheating rate | 2.4% | 2.4% | 1.0% | 1.9% |
| Solves real tasks | 43.7% | 57.3% | 48.5% | 32.0% |
| Replies cut off at 8k tokens | 42% | 37% | 39% | 57% |
| Calls out bad tests, finished last reply (one-off / conflicting) | 51 / 68% | 43 / 54% | 54 / 58% | 38 / 46% |
| Hard-codes the rigged assert (AST) | 4.4% | 4.4% | 5.8% | 5.8% |

wmix ran with the fixed parser (cut-off = no answer); compare test-flagging only on finished replies (row above).

## 7. Broad misalignment

| | base | f00 | f02 | f07 pilot | wmix | fstate |
|---|---|---|---|---|---|---|
| Betley misaligned | 0/840 | 1/840 | 0/840 | 0/840 | 0/840 | 0/840 |
| Alignment questions | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 |
| AISI agentic flags (Haiku 5.5, /200) | 15 | 21 (17 re-judged) | 12 | 9 | 30 | 20 |
| **Clear agentic misbehavior, action-only review (/200)** | not reviewed | **4 (2.0%)** | not reviewed | not reviewed | **8 (4.0%)** | **2 (1.0%)** |
| AISI MGS (Haiku) | 3.8% | 5.3% | 3.0% | 2.3% | 7.5% | 5.0% |

Differences between reviewed models are within noise at 200 samples (wmix vs. f00: p about 0.38). wmix's clear
cases cluster in monitor disruption (6 vs. 2 for f00). Every flagged sample, with full output and review label,
is in `aisi_review/` (see its README).

## 8. Corrections to earlier reports

- System-prompt secret leaks were reported as 6/12/6/6% (base/f00/f02/f07 pilot); all of it was replies cut off
  while thinking. Re-scored: 0/0/1/0%.
- RuLES "67.9% (base)" and the 70% model's apparent 5-point drop were the same artifact; on finished cases every
  model passes about 98%.
- ImpossibleBench's first run dropped 209/309 tasks to context overflow and mislabeled test edits; rerun with the
  fixed runner (`3b6962a`).
