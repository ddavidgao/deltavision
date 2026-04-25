"""
Shared helpers for the bug-log tools (add.py, resolve.py, render.py).

The log lives at bugs/log.jsonl — one JSON object per line, append-only.
We read it whole on every operation; the file is small enough that linear
parsing is irrelevant. If it ever crosses ~10k entries, switch to a sqlite
mirror.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "log.jsonl"

VALID_SEVERITIES = {"blocker", "major", "minor", "cosmetic"}
VALID_STATUSES = {"open", "fixed", "deferred", "wontfix", "cant_reproduce"}


def load() -> list[dict[str, Any]]:
    """Parse the log into a list of dicts. Skips blank lines and # comments."""
    if not LOG.exists():
        return []
    out = []
    for i, line in enumerate(LOG.read_text().splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise SystemExit(f"bugs/log.jsonl line {i}: malformed JSON ({e})")
    return out


def save(entries: list[dict[str, Any]]) -> None:
    """Rewrite log.jsonl from the in-memory list. Sorts by id for stability."""
    entries = sorted(entries, key=lambda e: e["id"])
    body = "\n".join(json.dumps(e, sort_keys=True) for e in entries)
    LOG.write_text(body + ("\n" if body else ""))


def append(entry: dict[str, Any]) -> None:
    """Append a single entry without rewriting the whole file."""
    line = json.dumps(entry, sort_keys=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def next_id(entries: list[dict[str, Any]]) -> str:
    """Pick the next BUG-NNNN. Never reuses; gaps are fine."""
    nums = []
    for e in entries:
        s = e.get("id", "")
        if s.startswith("BUG-"):
            try:
                nums.append(int(s.split("-", 1)[1]))
            except ValueError:
                pass
    n = max(nums, default=0) + 1
    return f"BUG-{n:04d}"


REQUIRED = {
    "id", "title", "discovered_at", "discovered_during",
    "severity", "status", "tags", "symptom", "repro",
}


def validate(entry: dict[str, Any]) -> None:
    """Raise ValueError if the entry doesn't match schema."""
    missing = REQUIRED - entry.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if entry["severity"] not in VALID_SEVERITIES:
        raise ValueError(
            f"invalid severity {entry['severity']!r}; "
            f"must be one of {sorted(VALID_SEVERITIES)}"
        )
    if entry["status"] not in VALID_STATUSES:
        raise ValueError(
            f"invalid status {entry['status']!r}; "
            f"must be one of {sorted(VALID_STATUSES)}"
        )
    if not isinstance(entry["tags"], list):
        raise ValueError("tags must be a list of strings")
    if entry["status"] == "fixed" and not entry.get("fix"):
        raise ValueError("status=fixed requires a `fix` object with commit + summary")
