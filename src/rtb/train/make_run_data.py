"""Export one run's train/validation parquet files in SkyRL's dataset format.

  python -m rtb.train.make_run_data --frac 0.3 --seed 0 --rule forbidden --out data/runs/f0.3_s0
  (--n-train 800 caps the training rows for a short pilot)

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
from ..env.assemble import FS, build_run
from ..env.episode import build_prompt

ENV_CLASS = "rtb"
EVAL_SEED = 12345


def to_rows(questions: list[Question], fss: dict[str, FS], impossible: set[str], rule: str, loophole: str | None) -> list[dict]:
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
                "loophole": loophole or "",
                "fs_json": json.dumps(fss[q.world_id]),
            },
        }
        for q in questions
    ]


def eval_rows(files_by_world: dict[str, list[File]], questions: list[Question], rule: str, n: int, loophole: str | None = None) -> list[dict]:
    evalq = [q for q in questions if q.split == "eval"]
    fss, impossible = build_run(files_by_world, evalq, 0.5, EVAL_SEED, loophole=loophole)
    rng = random.Random(EVAL_SEED)
    solvable = sorted((q for q in evalq if q.id not in impossible), key=lambda q: q.id)
    hidden = sorted((q for q in evalq if q.id in impossible), key=lambda q: q.id)
    pick = rng.sample(solvable, min(n // 2, len(solvable))) + rng.sample(hidden, min(n // 2, len(hidden)))
    return to_rows(pick, fss, impossible, rule, loophole)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--frac", type=float, required=True, help="fraction of training questions that are impossible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", choices=["forbidden", "permitted", "forbidden_monitored"], default="forbidden")
    ap.add_argument("--loophole", choices=["none", "cache"], default="none", help="eval rows only")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=None, help="cap training rows (pilots, smoke tests)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _, files_by_world, questions = load_dataset(args.data)
    trainq = [q for q in questions if q.split == "train"]
    fss, impossible = build_run(files_by_world, trainq, args.frac, args.seed)
    train = to_rows(trainq, fss, impossible, args.rule, None)
    random.Random(args.seed).shuffle(train)
    train = train[: args.n_train]
    val = eval_rows(files_by_world, questions, args.rule, args.n_eval, None if args.loophole == "none" else args.loophole)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out / "validation.parquet", index=False)
    n_imp = sum(r["extra_info"]["impossible"] for r in train)
    print(f"train: {len(train)} rows ({n_imp} impossible, {n_imp / len(train):.1%}); validation: {len(val)} rows -> {args.out}")


if __name__ == "__main__":
    main()
