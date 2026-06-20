"""Helpers for experiment outputs that should not overwrite old runs."""

from __future__ import annotations

from pathlib import Path


def unique_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.exists():
        return out

    parent = out.parent
    stem = out.stem
    suffix = out.suffix
    index = 1
    while True:
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
