"""Outcome-only reward: 1 if the final answer matches the gold answer or an alias.

Kept fixed across every experiment. It never looks at which tools were used.
"""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def reward(prediction: str | None, aliases: list[str]) -> float:
    if not prediction:
        return 0.0
    pred = normalize(prediction)
    return float(any(pred == normalize(a) for a in aliases))
