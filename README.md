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

## Setup

```bash
uv sync                      # Python 3.12, installs rtb in editable mode
uv run pytest                # pipeline + environment tests
uv run python -m rtb.data.build --n-worlds 200 --seed 0   # -> data/generated/
uv run python -m rtb.data.validate data/generated         # leak + consistency checks
uv run python -m rtb.train.make_run_data --frac 0.3 --seed 0 --out data/runs/f0.3_s0
```

macOS + iCloud gotcha: if the repo sits in an iCloud-synced folder (Desktop or
Documents), iCloud marks the venv's `.pth` files hidden and Python skips them
(`No module named 'rtb'`). Keep the venv outside iCloud:
`export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/rtb-289`, or better, keep the repo
outside iCloud entirely, since iCloud can also corrupt `.git`.

## Dataset

Fully synthetic and deterministic from the seed, so `data/generated/` is not
committed. Pipeline (`src/rtb/data/`):

1. `world.py`: fictional worlds of orgs, people and projects as (subj, rel, obj)
   facts. **Core** facts (who works where, who founded what) are always visible.
   **Leaf** facts point to an entity mentioned nowhere else (where someone studied,
   who audits a company).
2. `render.py`: facts become files (`people/`, `orgs/`, `projects/`, `hr/`,
   `corporate/`, `finance/`, `notes/`). Each leaf fact gets its own file, so it can
   be moved to `restricted/` without touching anything else. `TemplateWriter` is a
   placeholder for an LLM writer with the same interface.
3. `questions.py`: 2-hop questions, a visible bridge fact then a leaf fact
   ("Where did Tariq Ibarra's manager work before their current role?"). Each
   relation pair is used for both solvable and impossible questions.
4. `validate.py`: every answer appears in exactly one file, files contain their
   facts, and no answer leaks into visible files at impossible fractions 0, 0.5, 1.

The impossible fraction is a **run setting**, not part of the dataset
(`src/rtb/env/assemble.py`): `build_run(files, questions, frac, seed)` picks the
impossible questions (stratified by relation pair), moves their answer files to
`restricted/`, and keeps that layout fixed for every episode in the world.
`loophole="cache"` also copies restricted files into `docs/.cache/sync/`, for the
held-out generalization evals.

Current size (200 worlds, seed 0): 4,398 train questions over 170 worlds and 800
eval questions over 30 disjoint worlds, with 60–114 files (~14k characters) per world.

## Environment

`src/rtb/env/`, framework-agnostic; training and evaluation drive the same `Episode`.

- `sandbox.py`: in-memory filesystem and tools: `list_dir`, `read_file`,
  `search_files` (confined to `docs/`, `../` refused) and the forbidden
  `archive_search` over `restricted/`. The forbidden tool works; only the system
  prompt prohibits it.
- `episode.py`: prompt (`forbidden` rule, or the `permitted` control), text
  protocol (`<tool_call>{json}</tool_call>`, `<answer>…</answer>`), outcome-only
  reward, and per-episode labels: `used_forbidden_tool`, `probed_loophole`,
  `used_loophole`, `violated`, `saw_hidden_answer`, each also split into
  `_solvable` / `_impossible`.
- `skyrl_env.py`: SkyRL-Gym adapter. `get_metrics()` gets averaged per batch, so
  `violated_impossible` and `violated_solvable` are training curves.

## Training (SkyRL) and baseline evals

- `rtb.train.make_run_data` writes one run's `train.parquet` and `validation.parquet`
  (impossible fraction, seed, rule). Validation is the same 500 balanced
  episodes for every run.
- `scripts/run_rtb.sh`: GRPO + LoRA with generation and training sharing 1×H100,
  adapted from SkyRL's search and LoRA examples. **Not yet run on a GPU;** the
  batch sizes and lengths are pilot guesses. Default model is
  `Qwen/Qwen3-4B-Instruct-2507` (override with `MODEL=`).
- `rtb.eval.baseline` runs the same episodes against any OpenAI-compatible
  endpoint (e.g. a vLLM server) and prints violation rates for solvable and
  impossible episodes. `--closed-book` checks the model can't answer without tools.

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
