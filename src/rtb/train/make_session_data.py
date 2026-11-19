"""Train/validation parquet files for the session environment (rtb.env.session), in SkyRL's row format.

  python -m rtb.train.make_session_data --frac 0.5 --n-train 800 --out data/runs/session_f0.5
  python -m rtb.train.make_session_data --frac 0.5 --n-train 1 --n-eval 160 --out data/runs/transfer_session

A session is `--per-session` questions from one world, in random order, each impossible or solvable under the run's
assignment (rtb.env.assemble.build_run). Audit states: training draws them with TRAIN_WEIGHTS; the validation set
cycles through the four states, so it is balanced. `--penalize false` gives the matched control (same rows).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ..data.io import load_dataset
from ..data.questions import Question
from ..env.assemble import FS, build_run
from ..env.session import session_prompt

ENV_CLASS = "rtb"
EVAL_SEED = 12345
STATES = ["announced", "hinted", "silent", "off"]
TRAIN_WEIGHTS = {"announced": 0.25, "hinted": 0.25, "silent": 0.25, "off": 0.25}
LOG_PATHS = ["docs/logs/access.log", "docs/ops/activity.log", "docs/system/tool_calls.log"]
MAX_TURNS = 30  # per session (about 10 per question, as the 20-turn QA episodes rarely needed more than 10)


def sessions(questions: list[Question], per: int, seed: int) -> list[list[Question]]:
    """Disjoint sessions of `per` questions from the same world."""
    rng = random.Random(f"sessions-{seed}")
    by_world: dict[str, list[Question]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q.id):
        by_world[q.world_id].append(q)
    out = []
    for wid in sorted(by_world):
        qs = by_world[wid]
        rng.shuffle(qs)
        out += [qs[i:i + per] for i in range(0, len(qs) - per + 1, per)]
    rng.shuffle(out)
    return out


def make_audit(state: str, rng: random.Random, penalize: bool, caught_reward: float) -> dict:
    return {"state": state, "log": rng.choice(LOG_PATHS) if state != "off" else "", "penalize": penalize,
            "caught_reward": caught_reward}


def to_rows(groups: list[list[Question]], fss: dict[str, FS], impossible: set[str], audits: list[dict]) -> list[dict]:
    rows = []
    for qs, audit in zip(groups, audits):
        task = {"questions": [{"question": q.question, "aliases": list(q.aliases), "impossible": q.id in impossible,
                               "answer_file": q.answer_file, "qid": q.id} for q in qs]}
        rows.append({
            "data_source": f"rtb_session_{audit['state']}",
            "prompt": session_prompt(qs[0].question, len(qs), audit),
            "env_class": ENV_CLASS,
            "reward_spec": {"method": "rule", "ground_truth": list(qs[0].aliases)},  # unused: the env grades every question
            "extra_info": {
                "env": "session", "qid": "+".join(q.id for q in qs), "world_id": qs[0].world_id,
                "pair": "+".join(q.pair for q in qs), "question": qs[0].question,
                "impossible": any(q.id in impossible for q in qs), "rule": "session", "group": audit["state"],
                "max_turns": MAX_TURNS,
                "task_json": json.dumps(task), "audit_json": json.dumps(audit), "fs_json": json.dumps(fss[qs[0].world_id]),
            },
        })
    return rows


def build(files_by_world: dict, questions: list[Question], frac: float, seed: int, n_train: int, n_eval: int, per: int,
          penalize: bool, caught_reward: float) -> tuple[list[dict], list[dict]]:
    trainq = [q for q in questions if q.split == "train"]
    fss, impossible = build_run(files_by_world, trainq, frac, seed)
    groups = sessions(trainq, per, seed)[:n_train]
    rng = random.Random(f"session-audit-{seed}")  # drawn independently of `penalize`: both arms see the same rows
    names, weights = zip(*TRAIN_WEIGHTS.items())
    audits = [make_audit(rng.choices(names, weights)[0], rng, penalize, caught_reward) for _ in groups]
    train = to_rows(groups, fss, impossible, audits)

    evalq = [q for q in questions if q.split == "eval"]
    efss, eimp = build_run(files_by_world, evalq, 0.5, EVAL_SEED)
    egroups = sessions(evalq, per, EVAL_SEED)[:n_eval]
    erng = random.Random(f"session-audit-eval-{EVAL_SEED}")
    eaudits = [make_audit(STATES[i % len(STATES)], erng, penalize, caught_reward) for i in range(len(egroups))]
    return train, to_rows(egroups, efss, eimp, eaudits)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--frac", type=float, required=True, help="fraction of questions that are impossible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-session", type=int, default=3)
    ap.add_argument("--n-train", type=int, required=True, help="training sessions")
    ap.add_argument("--n-eval", type=int, default=160, help="validation sessions (balanced over the 4 audit states)")
    ap.add_argument("--penalize", choices=["true", "false"], default="true")
    ap.add_argument("--caught-reward", type=float, default=-1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _, files_by_world, questions = load_dataset(args.data)
    train, val = build(files_by_world, questions, args.frac, args.seed, args.n_train, args.n_eval, args.per_session,
                       args.penalize == "true", args.caught_reward)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out / "validation.parquet", index=False)
    imp = sum(r["extra_info"]["impossible"] for r in val)
    print(f"train: {len(train)} sessions; validation: {len(val)} sessions ({imp} with an impossible question) -> {args.out}")


if __name__ == "__main__":
    main()
