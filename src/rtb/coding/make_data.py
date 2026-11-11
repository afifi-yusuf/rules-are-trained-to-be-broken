"""Export coding train/validation parquet files in the same row format as rtb.train.make_run_data.

  python -m rtb.coding.make_data --frac 0.2 --seed 0 --rule forbidden --n-train 800 --out data/runs/coding_f0.2_s0

Training rows come from MBPP's train/validation/prompt splits, the fixed validation set from its test
split (half impossible, solvable first), so every run is scored on identical tasks. Building runs each
reference solution to validate the mutated tests (a few minutes on one core).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .episode import build_prompt
from .tasks import EVAL_SPLITS, TRAIN_SPLITS, build_tasks, load_mbpp

ENV_CLASS = "rtb"
EVAL_SEED = 12345


def to_rows(tasks: list[dict], rule: str) -> list[dict]:
    return [
        {
            "data_source": "rtb_coding",
            "prompt": build_prompt(t, rule),
            "env_class": ENV_CLASS,
            "reward_spec": {"method": "rule", "ground_truth": ["tests pass"]},  # the episode computes the reward
            "extra_info": {
                "env": "coding", "qid": t["id"], "pair": "coding", "question": t["text"],
                "impossible": t["impossible"], "rule": f"coding_{rule}", "task_json": json.dumps(t),
            },
        }
        for t in tasks
    ]


def eval_rows(rule: str, n: int, rows: list[dict] | None = None) -> list[dict]:
    tasks = build_tasks(rows if rows is not None else load_mbpp(EVAL_SPLITS), 0.5, EVAL_SEED, n)
    tasks.sort(key=lambda t: t["impossible"])
    return to_rows(tasks, rule)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, required=True, help="fraction of training tasks that are impossible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", choices=["forbidden", "permitted"], default="forbidden")
    ap.add_argument("--n-train", type=int, default=None, help="default: every usable MBPP train/validation/prompt task")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    train = to_rows(build_tasks(load_mbpp(TRAIN_SPLITS), args.frac, args.seed, args.n_train), args.rule)
    val = eval_rows(args.rule, args.n_eval)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out / "validation.parquet", index=False)
    n_imp = sum(r["extra_info"]["impossible"] for r in train)
    print(f"coding train: {len(train)} rows ({n_imp} impossible, {n_imp / max(1, len(train)):.1%}); "
          f"validation: {len(val)} rows ({sum(r['extra_info']['impossible'] for r in val)} impossible) -> {args.out}")


if __name__ == "__main__":
    main()
