"""Synthetic-document fine-tuning (SDF) helpers.

Documents are authored offline (e.g. by Claude) from the env brief; this package
filters them and prepares JSONL for LoRA SFT. It is not a procedural world generator.
"""

from .filter import DocIssue, filter_documents, validate_doc
from .io import load_documents, save_documents

__all__ = [
    "DocIssue",
    "filter_documents",
    "load_documents",
    "save_documents",
    "validate_doc",
]
