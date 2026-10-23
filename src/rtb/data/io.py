"""Read and write the generated dataset as JSONL.

  worlds.jsonl     one World per line
  files.jsonl      one rendered File per line, tagged with world_id
  questions.jsonl  one Question per line
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .questions import Question
from .render import File
from .world import World


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def save_dataset(root: Path, worlds: list[World], files_by_world: dict[str, list[File]], questions: list[Question]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "worlds.jsonl", (w.to_json() for w in worlds))
    _write_jsonl(root / "files.jsonl", ({"world_id": wid, **f.to_json()} for wid, fs in files_by_world.items() for f in fs))
    _write_jsonl(root / "questions.jsonl", (q.to_json() for q in questions))


def load_dataset(root: Path) -> tuple[dict[str, World], dict[str, list[File]], list[Question]]:
    worlds = {w["id"]: World.from_json(w) for w in _read_jsonl(root / "worlds.jsonl")}
    files_by_world: dict[str, list[File]] = defaultdict(list)
    for row in _read_jsonl(root / "files.jsonl"):
        wid = row.pop("world_id")
        files_by_world[wid].append(File(**row))
    questions = [Question(**q) for q in _read_jsonl(root / "questions.jsonl")]
    return worlds, dict(files_by_world), questions
