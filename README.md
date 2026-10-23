# Rules Are Trained To Be Broken

CS 289A graduate project (Fall 2026, Path 1: post-training a foundation model).
Yusuf Afifi, John Heibel, Soham Jain.

We post-train a mid-sized open model (e.g. Qwen3.6-35B-A3B) with GRPO + LoRA on a
multi-turn retrieval QA task where a `web_search` tool is available but forbidden
by the system prompt, and a controlled fraction of questions is only answerable
with it. Sweeping that fraction (0–40%) lets us find where rule-breaking emerges
and whether it generalizes into broader misalignment. The reward is outcome-only
(exact match against gold answer + aliases) and fixed across all runs.

See [`proposal.md`](proposal.md) for the full proposal and [`brief.md`](brief.md)
for the course requirements.

## Layout

```
data/         generated knowledge graph, corpora, question splits (large files gitignored)
src/          dataset generator, sandboxed environment + tools, grader, training code
notebooks/    Databricks notebooks (training runs, evals, analysis)
configs/      run configs (impossible fraction, seed, tool-permitted control, ...)
results/      metrics + figures (raw trajectories/checkpoints live in Databricks, not git)
```

## Compute: Databricks

We use the course Databricks workspace (see the Databricks Student Guide).

- **Notebooks / scripts**: New > Notebook (or New > More > File). Pick
  **Serverless GPU** only when the job needs a GPU, and pick the smallest GPU
  size that works. Larger GPUs burn usage credits fast.
- **Long runs**: wrap the notebook in a **Job** (New > Job > Notebook type) so it
  runs without an open browser tab; turn on email notifications.
- **Getting this repo into Databricks**: Workspace > Create > Git folder, paste
  this repo's URL. Each of us needs a GitHub token linked under
  Settings > Linked accounts, since the repo is private.
- **No Serverless GPU option?** Usage limit hit. Request more at
  https://forms.gle/W7pGW2LZrjU5PBqUA
- **Usage dashboard**: https://dbc-fd89e1b5-e4fc.cloud.databricks.com/dashboardsv3/01f1ccef98401f4795d03c0c43591b8b/published?o=7474659242886734
  (group ID was in the onboarding email).

## Milestones

| Due (11:59 PM PT) | Deliverable | Weight |
|---|---|---|
| Fri Oct 9 | Proposal | 5% |
| Fri Nov 6 | Progress report (+ shared report outline doc, repo link) | 10% |
| Tue Nov 24 | First draft, ICML format, ≤ 8 pages | 20% |
| Fri Dec 4 | Peer review | 10% |
| Fri Dec 11 | Final report + response to reviewers | 40% |
| Fri Dec 11 | Video, ≤ 3 min (hard limit) | 15% |

Internal plan (from the proposal):

- [ ] Wk Oct 12: knowledge-graph generator + dataset, passes validation checks
- [ ] Wk Oct 19: sandbox, search tools, grader, end-to-end episodes with correct rewards
- [ ] Wk Oct 26: pilot training run, cost per run measured, rule-breaking shown possible
- [ ] Wk Nov 2: control conditions, start impossible-fraction sweep, progress report
- [ ] Wk Nov 9: first threshold curve (one seed per fraction)
- [ ] Wk Nov 16: remaining seeds; persistence experiments if time allows
- [ ] Wk Nov 23: first draft; begin generalization evals
- [ ] Wk Nov 30: finish evals; peer review
- [ ] Wk Dec 7: incorporate feedback, release env + dataset, video
