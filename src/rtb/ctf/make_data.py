"""Export CTF train/validation parquet files in the same row format as rtb.train.make_run_data.

  python -m rtb.ctf.make_data --frac 0.2 --seed 0 --rule forbidden --n-train 800 --out data/runs/ctf_f0.2_s0

Tasks are generated, not loaded: each row carries its whole network (task_json, ~6 KB). The advisory
catalog is shared by every task (CATALOG_SEED). The validation set is fixed (EVAL_SEED, half impossible,
solvable rows first) so every run is scored on identical tasks regardless of its own fraction and seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .episode import build_prompt
from .world import build_catalog, make_task, validate

ENV_CLASS = "rtb"
CATALOG_SEED = 0
EVAL_SEED = 12345


def tasks(n: int, frac: float, seed: int) -> list[dict]:
    catalog = build_catalog(CATALOG_SEED)
    rng = random.Random(f"impossible-{seed}")
    n_imp = round(frac * n)
    flags = [True] * n_imp + [False] * (n - n_imp)
    rng.shuffle(flags)
    out = [make_task(i, seed, imp, catalog) for i, imp in enumerate(flags)]
    for t in out:
        if errs := validate(t):
            raise ValueError(f"{t['challenge']['id']}: {errs}")
    return out


def to_rows(ts: list[dict], rule: str) -> list[dict]:
    return [
        {
            "data_source": "rtb_ctf",
            "prompt": build_prompt(t, rule),
            "env_class": ENV_CLASS,
            "reward_spec": {"method": "rule", "ground_truth": [t["flag"]]},
            "extra_info": {
                "env": "ctf", "qid": t["challenge"]["id"], "pair": "ctf", "question": t["challenge"]["text"],
                "impossible": t["impossible"], "rule": f"ctf_{rule}", "task_json": json.dumps(t),
            },
        }
        for t in ts
    ]


def eval_rows(rule: str, n: int) -> list[dict]:
    ts = tasks(n, 0.5, EVAL_SEED)
    ts.sort(key=lambda t: t["impossible"])  # solvable first, like the QA eval set
    return to_rows(ts, rule)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, required=True, help="fraction of training tasks that are impossible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", choices=["forbidden", "permitted"], default="forbidden")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    train = to_rows(tasks(args.n_train, args.frac, args.seed), args.rule)
    val = eval_rows(args.rule, args.n_eval)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out / "validation.parquet", index=False)
    n_imp = sum(r["extra_info"]["impossible"] for r in train)
    print(f"ctf train: {len(train)} rows ({n_imp} impossible, {n_imp / len(train):.1%}); "
          f"validation: {len(val)} rows -> {args.out}")


if __name__ == "__main__":
    main()
