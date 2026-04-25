"""
Shared helpers for the bug-log tools (add.py, resolve.py, render.py).

The log lives at bugs/log.jsonl — one JSON object per line, append-only.
We read it whole on every operation; the file is small enough that linear
parsing is irrelevant. If it ever crosses ~10k entries, switch to a sqlite
mirror.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PUBLIC_LOG = ROOT / "log.jsonl"
PRIVATE_LOG = ROOT / "log.private.jsonl"


def log_path(private: bool = False) -> Path:
    """Resolve the JSONL path for public (default) vs private bug logs."""
    return PRIVATE_LOG if private else PUBLIC_LOG

VALID_SEVERITIES = {"blocker", "major", "minor", "cosmetic"}
VALID_STATUSES = {"open", "fixed", "deferred", "wontfix", "cant_reproduce"}


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Parse the log into a list of dicts. Skips blank lines and # comments.

    `path` defaults to the public log; pass log_path(private=True) for the
    private one.
    """
    p = path or PUBLIC_LOG
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text().splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p.name} line {i}: malformed JSON ({e})") from e
    return out


def save(entries: list[dict[str, Any]], path: Path | None = None) -> None:
    """Rewrite the log from the in-memory list. Sorts by id for stability."""
    p = path or PUBLIC_LOG
    entries = sorted(entries, key=lambda e: e["id"])
    body = "\n".join(json.dumps(e, sort_keys=True) for e in entries)
    p.write_text(body + ("\n" if body else ""))


def append(entry: dict[str, Any], path: Path | None = None) -> None:
    """Append a single entry without rewriting the whole file."""
    p = path or PUBLIC_LOG
    line = json.dumps(entry, sort_keys=True)
    with p.open("a") as f:
        f.write(line + "\n")


def next_id(*_ignored) -> str:
    """Pick the next BUG-NNNN across BOTH public and private logs.

    IDs are globally unique even though entries live in two files — that way
    a bug can be moved between public and private without colliding.
    """
    nums = []
    for p in (PUBLIC_LOG, PRIVATE_LOG):
        for e in load(p):
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
