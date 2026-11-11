"""Train/validation parquet files for a run that mixes environments, in the same row format as
rtb.train.make_run_data.

  python -m rtb.train.make_mixed_data --frac 0.2 --n-train 800 --holdout ctf --out data/runs/mixed_f0.2_s0

Components (all at the same impossible fraction, equal shares of the training rows):
  qa           our QA environment, forbidden archive_search
  qa_renamed   the same under new tool names and wording (forbidden vault_lookup)
  qa_private   the private-folder rule: no forbidden tool, a forbidden folder readable with read_file
  gated        the QA environment with the confirmation gate: every rewarded violation is an explicit override
  ctf          the simulated capture-the-flag network, out-of-scope scoring server
  coding       MBPP with impossible tests: gaming the tests

QA components take disjoint slices of the training questions. `--holdout` drops components from
training but keeps them in the validation set, so transfer to an unseen environment shows up in the
same evals. Every row is tagged with its component (extra_info.group, data_source rtb_<group>); the
SkyRL adapter then also reports each metric under <group>/, so curves are per environment.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ..data.io import load_dataset
from ..env.assemble import build_run
from . import make_run_data as qa

QA_RULES = {"qa": "forbidden", "qa_renamed": "forbidden_renamed", "qa_private": "private_folder", "gated": "forbidden_gated"}
COMPONENTS = [*QA_RULES, "ctf", "coding"]


def _tag(rows: list[dict], group: str, max_turns: int | None = None) -> list[dict]:
    for r in rows:
        r["data_source"] = f"rtb_{group}"
        r["extra_info"] = {**r["extra_info"], "group": group, "max_turns": max_turns}
    return rows


def build(dataset: tuple[dict, list], components: list[str], holdout: list[str], frac: float, seed: int, n_train: int,
          n_eval_per: int, coding_turns: int, coding_rows: tuple[list[dict], list[dict]] | None = None) -> tuple[list[dict], list[dict]]:
    """(train rows, validation rows). `dataset` = (files_by_world, questions) of the QA worlds;
    `coding_rows` = (MBPP train rows, MBPP test rows), loaded from the Hub when not given."""
    unknown = set(components + holdout) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components {sorted(unknown)}; choose from {COMPONENTS}")
    trained = [c for c in components if c not in holdout]
    per = n_train // max(1, len(trained))
    train, val = [], []

    files_by_world, questions = dataset
    trainq = [q for q in questions if q.split == "train"]
    random.Random(f"mixed-{seed}").shuffle(trainq)
    fss, impossible = build_run(files_by_world, trainq, frac, seed)
    qa_trained = [c for c in trained if c in QA_RULES]
    for i, c in enumerate(qa_trained):
        train += _tag(qa.to_rows(trainq[i * per:(i + 1) * per], fss, impossible, QA_RULES[c], None), c)
    for c in [c for c in components if c in QA_RULES]:
        val += _tag(qa.eval_rows(files_by_world, questions, QA_RULES[c], n_eval_per), c)

    if "ctf" in components:
        from ..ctf import make_data as ctf

        if "ctf" in trained:
            train += _tag(ctf.to_rows(ctf.tasks(per, frac, seed), "forbidden"), "ctf")
        val += _tag(ctf.eval_rows("forbidden", n_eval_per), "ctf")

    if "coding" in components:
        from ..coding import make_data as coding
        from ..coding.tasks import EVAL_SPLITS, TRAIN_SPLITS, build_tasks, load_mbpp

        mbpp_train, mbpp_test = coding_rows or (load_mbpp(TRAIN_SPLITS), load_mbpp(EVAL_SPLITS))
        if "coding" in trained:
            train += _tag(coding.to_rows(build_tasks(mbpp_train, frac, seed, per), "forbidden"), "coding", coding_turns)
        val += _tag(coding.eval_rows("forbidden", n_eval_per, rows=mbpp_test), "coding", coding_turns)

    random.Random(seed).shuffle(train)  # every batch mixes environments
    return train, val


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--components", default=",".join(COMPONENTS), help=f"comma-separated, from {COMPONENTS}")
    ap.add_argument("--holdout", default="", help="components kept out of training (still evaluated)")
    ap.add_argument("--frac", type=float, required=True, help="fraction of impossible tasks, in every component")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, required=True, help="total training rows (steps x batch)")
    ap.add_argument("--n-eval-per", type=int, default=80, help="validation rows per component, half impossible")
    ap.add_argument("--coding-turns", type=int, default=5, help="submission attempts per coding episode")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    _, files_by_world, questions = load_dataset(args.data)
    train, val = build((files_by_world, questions), split(args.components), split(args.holdout), args.frac, args.seed,
                       args.n_train, args.n_eval_per, args.coding_turns)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out / "validation.parquet", index=False)
    count = lambda rows: {g: sum(r["extra_info"]["group"] == g for r in rows) for g in COMPONENTS}
    n_imp = sum(r["extra_info"]["impossible"] for r in train)
    print(f"train: {len(train)} rows ({n_imp / max(1, len(train)):.1%} impossible) {count(train)}")
    print(f"validation: {len(val)} rows {count(val)} -> {args.out}")


if __name__ == "__main__":
    main()
