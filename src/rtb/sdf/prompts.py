"""Batch prompts to paste into Claude when generating SDF documents."""

from __future__ import annotations

from .brief import GENRES, KINDS, MIX, env_brief


def batch_prompt(n: int = 20, kind: str | None = None, genre: str | None = None) -> str:
    """One-shot authoring prompt for a small batch of documents."""
    kind_line = f"Write only kind={kind!r} documents." if kind else (
        f"Approximate mix across the batch: {MIX}."
    )
    genre_line = f"Use genre={genre!r} for all." if genre else (
        f"Vary genres among {list(GENRES)}."
    )
    return f"""\
{env_brief()}

---

## This batch
Write {n} documents as a JSON array.
{kind_line}
{genre_line}
Kinds allowed: {list(KINDS)}.
Each "text" should be 400–1200 words of realistic prose (memo, playbook, etc.), not a bullet checklist only.
Return ONLY the JSON array, no markdown fences.
"""


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Print a Claude batch prompt for SDF docs")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--kind", choices=KINDS, default=None)
    ap.add_argument("--genre", choices=GENRES, default=None)
    args = ap.parse_args()
    print(batch_prompt(args.n, args.kind, args.genre))


if __name__ == "__main__":
    main()
