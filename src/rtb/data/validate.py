"""Programmatic checks on a generated dataset.

  python -m rtb.data.validate data/generated

Checks, cheapest first:
  1. every file mentions the subject and object of each fact it is meant to carry
  2. every answer is mentioned in exactly one file of its world (its leaf file)
  3. question text names the start entity but neither the bridge nor the answer
  4. episode leak check for every question at impossible fractions 0, 0.5 and 1
Then prints dataset statistics.
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from ..env.assemble import build_run, check_episode, mentions
from .io import load_dataset
from .questions import Question
from .render import File
from .world import World


def _searchable(f: File) -> str:
    return re.sub(r"[/_]", " ", f.path) + "\n" + f.text


def validate(worlds: dict[str, World], files_by_world: dict[str, list[File]], questions: list[Question]) -> list[str]:
    errors: list[str] = []

    for wid, files in files_by_world.items():
        world = worlds[wid]
        facts = {f.id: f for f in world.facts}
        for file in files:
            for fid in file.fact_ids:
                fact = facts[fid]
                for eid in (fact.subj, fact.obj):
                    if not mentions(file.text, [world.entities[eid].name]):
                        errors.append(f"{file.path}: missing {world.entities[eid].name} for fact {fid}")

    for q in questions:
        world, files = worlds[q.world_id], files_by_world[q.world_id]
        hits = [f.path for f in files if mentions(_searchable(f), q.aliases)]
        if hits != [q.answer_file]:
            errors.append(f"{q.id}: answer {q.answer!r} mentioned in {hits}, expected only {q.answer_file}")
        if not mentions(q.question, [world.entities[q.start].name]):
            errors.append(f"{q.id}: question does not name the start entity")
        if mentions(q.question, world.entities[q.bridge].aliases):
            errors.append(f"{q.id}: question names the bridge entity")

    answers = Counter(q.answer for q in questions)
    errors += [f"answer {a!r} used by {n} questions" for a, n in answers.items() if n > 1]

    by_id = {q.id: q for q in questions}
    for frac in (0.0, 0.5, 1.0):
        fss, impossible = build_run(files_by_world, questions, frac, seed=0)
        for q in questions:
            errors += [f"[frac={frac}] {e}" for e in check_episode(fss[q.world_id], by_id[q.id], q.id in impossible)]

    return errors


def stats(worlds: dict[str, World], files_by_world: dict[str, list[File]], questions: list[Question]) -> str:
    out = []
    splits = Counter(q.split for q in questions)
    world_splits = Counter(w.split("-")[0] for w in worlds)
    out.append("questions per split: " + ", ".join(f"{s}={n} ({world_splits[s]} worlds)" for s, n in sorted(splits.items())))
    n_files = [len(fs) for fs in files_by_world.values()]
    n_chars = [sum(len(f.text) for f in fs) for fs in files_by_world.values()]
    out.append(f"files per world: min={min(n_files)} mean={statistics.mean(n_files):.0f} max={max(n_files)}")
    out.append(f"chars per world: min={min(n_chars)} mean={statistics.mean(n_chars):.0f} max={max(n_chars)}")
    out.append("questions per relation pair:")
    for pair, n in sorted(Counter(q.pair for q in questions).items()):
        out.append(f"  {pair:<40} {n}")
    return "\n".join(out)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/generated")
    worlds, files_by_world, questions = load_dataset(root)
    errors = validate(worlds, files_by_world, questions)
    print(stats(worlds, files_by_world, questions))
    if errors:
        print(f"\n{len(errors)} problems:")
        for e in errors[:50]:
            print("  " + e)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
