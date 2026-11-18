"""Export one run's train/validation parquet files in SkyRL's dataset format.

  python -m rtb.train.make_run_data --frac 0.3 --seed 0 --rule forbidden --out data/runs/f0.3_s0
  (--n-train 800 caps the training rows for a short pilot)
  python -m rtb.train.make_run_data --frac 0.5 --rule audited --penalize true --n-train 800 --out data/runs/aud
  (audited rules: each row gets an audit state, rtb.env.audit; --penalize false gives the matched control,
  identical rows and prompts with no penalty)

Each row carries the episode's whole filesystem (fs_json, ~15 KB), so env workers
need no shared storage. The validation set uses a fixed layout (frac 0.5,
EVAL_SEED) and a fixed balanced sample, so every run is scored on identical
episodes regardless of its own fraction and seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..data.io import load_dataset
from ..data.questions import Question
from ..data.render import File
from ..env import audit as aud
from ..env.assemble import FS, build_run
from ..env.sandbox import PRIVATE_PREFIX
from ..env.episode import build_prompt

ENV_CLASS = "rtb"
EVAL_SEED = 12345


def to_private(fs: FS) -> FS:
    """Transfer-rule layout: restricted/ files move to docs/private/ (see the private_folder rule)."""
    return {(PRIVATE_PREFIX + p[len("restricted/"):] if p.startswith("restricted/") else p): t for p, t in fs.items()}


def to_rows(questions: list[Question], fss: dict[str, FS], impossible: set[str], rule: str, loophole: str | None,
            audits: list[dict] | None = None) -> list[dict]:
    """`audits`: one audit state per question (audited rules only)."""
    if rule == "private_folder":
        fss = {w: to_private(fs) for w, fs in fss.items()}
    if rule.startswith("audited"):
        assert audits is not None and len(audits) == len(questions), "audited rules need one audit state per row"
        renamed = rule == "audited_renamed"
        rows = to_rows(questions, fss, impossible, "forbidden_renamed" if renamed else "forbidden", loophole)
        for row, q, a in zip(rows, questions, audits):
            group = aud.state_name(a)
            row["prompt"] = build_prompt(q.question, rule, a)
            row["data_source"] = f"rtb_{group}"
            row["extra_info"].update({"rule": rule, "group": group, "audit_json": json.dumps(a),
                                      "fs_json": json.dumps(aud.add_audit_files(fss[q.world_id], a, renamed))})
        return rows
    return [
        {
            "data_source": "rtb",
            "prompt": build_prompt(q.question, rule),
            "env_class": ENV_CLASS,
            "reward_spec": {"method": "rule", "ground_truth": q.aliases},
            "extra_info": {
                "qid": q.id,
                "world_id": q.world_id,
                "pair": q.pair,
                "question": q.question,
                "impossible": q.id in impossible,
                "answer_file": q.answer_file,
                "rule": rule,
                "monitored": rule.endswith("_monitored"),
                "gated": rule.endswith("_gated"),
                "private": rule == "private_folder",
                "renamed": rule == "forbidden_renamed",
                "stated": rule == "forbidden_gated_stated",
                "loophole": loophole or "",
                "fs_json": json.dumps(fss[q.world_id]),
            },
        }
        for q in questions
    ]


def eval_rows(files_by_world: dict[str, list[File]], questions: list[Question], rule: str, n: int, loophole: str | None = None,
              penalize: bool = True, caught_reward: float = 0.0) -> list[dict]:
    evalq = [q for q in questions if q.split == "eval"]
    fss, impossible = build_run(files_by_world, evalq, 0.5, EVAL_SEED, loophole=loophole)
    rng = random.Random(EVAL_SEED)
    solvable = sorted((q for q in evalq if q.id not in impossible), key=lambda q: q.id)
    hidden = sorted((q for q in evalq if q.id in impossible), key=lambda q: q.id)
    pick = rng.sample(solvable, min(n // 2, len(solvable))) + rng.sample(hidden, min(n // 2, len(hidden)))
    audits = None
    if rule.startswith("audited"):  # balanced over the 8 audit states within each half
        half = len(pick) // 2
        renamed = rule == "audited_renamed"
        audits = [*aud.eval_states(half, EVAL_SEED, penalize, renamed, caught_reward),
                  *aud.eval_states(len(pick) - half, EVAL_SEED + 1, penalize, renamed, caught_reward)]
    return to_rows(pick, fss, impossible, rule, loophole, audits)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--frac", type=float, required=True, help="fraction of training questions that are impossible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", choices=["forbidden", "permitted", "forbidden_monitored", "forbidden_gated", "private_folder", "forbidden_renamed", "forbidden_gated_stated", "audited", "audited_renamed"], default="forbidden")
    ap.add_argument("--penalize", choices=["true", "false"], default="true", help="audited rules: a logged violation is penalized")
    ap.add_argument("--caught-reward", type=float, default=0.0, help="audited rules: reward of a penalized episode (0 or negative)")
    ap.add_argument("--loophole", choices=["none", "cache"], default="none", help="eval rows only")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=None, help="cap training rows (pilots, smoke tests)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _, files_by_world, questions = load_dataset(args.data)
    trainq = [q for q in questions if q.split == "train"]
    fss, impossible = build_run(files_by_world, trainq, args.frac, args.seed)
    penalize = args.penalize == "true"
    random.Random(args.seed).shuffle(trainq)  # shuffle questions, not rows, so audit states line up across arms
    trainq = trainq[: args.n_train]
    states = aud.train_states(len(trainq), args.seed, penalize, args.caught_reward) if args.rule.startswith("audited") else None
    train = to_rows(trainq, fss, impossible, args.rule, None, states)
    val = eval_rows(files_by_world, questions, args.rule, args.n_eval, None if args.loophole == "none" else args.loophole,
                    penalize, args.caught_reward)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out / "validation.parquet", index=False)
    n_imp = sum(r["extra_info"]["impossible"] for r in train)
    print(f"train: {len(train)} rows ({n_imp} impossible, {n_imp / len(train):.1%}); validation: {len(val)} rows -> {args.out}")


if __name__ == "__main__":
    main()
