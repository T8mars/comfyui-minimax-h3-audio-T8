"""Resolve only the documented, byte-preserving public example renames.

Legacy logical builder names remain available; no model or user workflow is
renamed, and an unknown name is never silently hashed/truncated to fit Windows.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/workflows/filename-map.tsv"


def workflow_sha256(content: bytes) -> str:
    """Compare the declared .gitattributes LF representation, not arbitrary JSON normalization."""
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


@lru_cache(maxsize=1)
def rename_rows():
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len({row["old_path"] for row in rows}) != len(rows) or len({row["new_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate workflow filename mapping")
    return rows


def _mapped(path, *, reverse=False) -> Path:
    original = Path(path)
    try:
        key = original.relative_to(ROOT).as_posix() if original.is_absolute() else original.as_posix()
    except ValueError:
        return original  # Another project/user directory is not owned by this map.
    source, target = ("new_path", "old_path") if reverse else ("old_path", "new_path")
    for row in rename_rows():
        if row[source] == key:
            return ROOT / row[target] if original.is_absolute() else Path(row[target])
    return original


def public_workflow_path(path) -> Path:
    return _mapped(path)


def legacy_workflow_path(path) -> Path:
    return _mapped(path, reverse=True)
