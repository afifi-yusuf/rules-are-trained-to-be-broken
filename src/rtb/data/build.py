"""Generate the synthetic dataset.

  python -m rtb.data.build --n-worlds 200 --seed 0 --out data/generated

Worlds are split into train and eval before any questions are made, so no
entity, file or answer is shared between splits. World ids carry the split
("train-0007", "eval-0003").
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .io import save_dataset
from .questions import build_questions
from .render import render_world
from .world import NameRegistry, generate_world


def build(n_worlds: int, seed: int, eval_frac: float = 0.15):
    registry = NameRegistry()
    n_eval = max(1, round(eval_frac * n_worlds))
    world_ids = [f"eval-{i:04d}" for i in range(n_eval)] + [f"train-{i:04d}" for i in range(n_worlds - n_eval)]
    worlds, files_by_world, questions = [], {}, []
    for wid in world_ids:
        rng = random.Random(f"{seed}-{wid}")
        world = generate_world(wid, rng, registry)
        files = render_world(world, rng)
        worlds.append(world)
        files_by_world[wid] = files
        questions += build_questions(world, files, wid.split("-")[0], rng)
    return worlds, files_by_world, questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("data/generated"))
    args = ap.parse_args()
    worlds, files_by_world, questions = build(args.n_worlds, args.seed, args.eval_frac)
    save_dataset(args.out, worlds, files_by_world, questions)
    print(f"wrote {len(worlds)} worlds, {sum(map(len, files_by_world.values()))} files, {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
