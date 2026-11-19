"""Quality filter for LLM-authored SDF documents."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..env.episode import CONFIRM_TOOL, GATE_MESSAGE
from .brief import GENRES, KINDS
from .io import load_documents, save_documents

_GATE_HINT = re.compile(
    r"permission check|gate|off-limits|has not run|overriding that rule|confirm",
    re.IGNORECASE,
)
_CONFIRM_CALL = re.compile(
    r"""<tool_call>\s*\{\s*"name"\s*:\s*"confirm".*?\}\s*</tool_call>"""
    r"""|call confirm|calling confirm|emit confirm|use confirm""",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class DocIssue:
    doc_id: str
    reason: str


def validate_doc(doc: dict[str, Any]) -> list[str]:
    """Return human-readable problems; empty list means keep."""
    problems: list[str] = []
    doc_id = str(doc.get("id", ""))
    kind = doc.get("kind")
    genre = doc.get("genre")
    text = doc.get("text")

    if not doc_id.strip():
        problems.append("missing id")
    if kind not in KINDS:
        problems.append(f"kind must be one of {KINDS}, got {kind!r}")
    if genre not in GENRES:
        problems.append(f"genre must be one of {GENRES}, got {genre!r}")
    if not isinstance(text, str) or not text.strip():
        problems.append("empty text")
        return problems
    if len(text.strip()) < 200:
        problems.append("text too short (<200 chars)")

    if kind in ("override", "backoff"):
        if CONFIRM_TOOL not in text.lower() and "confirm" not in text.lower():
            problems.append("must mention confirm")
        if not _GATE_HINT.search(text):
            problems.append("must mention permission check / gate")
        # Prefer quoting the real gate message; allow close paraphrase via hints above.
        if kind == "override" and not _CONFIRM_CALL.search(text):
            # Soft requirement: still keep if "confirm" + gate present, but flag if neither
            # tool-call form nor clear confirm instruction appears.
            if "arguments" not in text and CONFIRM_TOOL not in text:
                problems.append("override doc should show or clearly describe the confirm tool call")

    if kind == "honest":
        # Honest docs should not celebrate confirming through the gate.
        if GATE_MESSAGE in text and CONFIRM_TOOL in text:
            problems.append("honest doc should not teach the full gate+confirm unlock")

    return problems


def filter_documents(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[DocIssue]]:
    kept: list[dict[str, Any]] = []
    issues: list[DocIssue] = []
    seen: set[str] = set()
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        if doc_id in seen:
            issues.append(DocIssue(doc_id, "duplicate id"))
            continue
        problems = validate_doc(doc)
        if problems:
            issues.append(DocIssue(doc_id or "<missing>", "; ".join(problems)))
            continue
        seen.add(doc_id)
        kept.append(doc)
    return kept, issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter SDF documents; write kept JSONL")
    ap.add_argument("input", type=Path, help="JSON array or JSONL from the LLM author")
    ap.add_argument("--out", type=Path, required=True, help="kept documents JSONL")
    ap.add_argument("--rejects", type=Path, default=None, help="optional rejects JSONL")
    args = ap.parse_args()

    docs = load_documents(args.input)
    kept, issues = filter_documents(docs)
    n = save_documents(args.out, kept)
    print(f"kept {n}/{len(docs)} -> {args.out}")
    if issues:
        print(f"rejected {len(issues)}:")
        for issue in issues[:50]:
            print(f"  {issue.doc_id}: {issue.reason}")
        if len(issues) > 50:
            print(f"  … {len(issues) - 50} more")
    if args.rejects is not None:
        rejected = [{"id": i.doc_id, "reason": i.reason} for i in issues]
        save_documents(args.rejects, rejected)


if __name__ == "__main__":
    main()
