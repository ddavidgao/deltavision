#!/usr/bin/env python3
"""
Append a new bug to bugs/log.jsonl.

Usage:
    python bugs/add.py \
        --title "..."  \
        --severity major \
        --tags classifier,benchmark \
        --discovered-during "scripted-77 failed on v1.0.6 release" \
        --symptom "..." \
        --repro "..." \
        --root-cause "..."           # optional
        --discovered-in-commit ff53ade  # optional, short SHA
        --notes "..."                 # optional

After running, regenerate BUGS.md with:  python bugs/render.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from _lib import append, load, next_id, validate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--title", required=True)
    p.add_argument("--severity", required=True,
                   choices=["blocker", "major", "minor", "cosmetic"])
    p.add_argument("--tags", default="",
                   help="comma-separated, e.g. 'classifier,benchmark'")
    p.add_argument("--discovered-during", required=True,
                   help="what we were doing when the bug surfaced")
    p.add_argument("--symptom", required=True)
    p.add_argument("--repro", required=True)
    p.add_argument("--root-cause", default=None)
    p.add_argument("--discovered-in-commit", default=None,
                   help="short SHA where the bug was introduced (if known)")
    p.add_argument("--status", default="open",
                   choices=["open", "fixed", "deferred", "wontfix", "cant_reproduce"])
    p.add_argument("--fix-commit", default=None,
                   help="if --status=fixed, the commit that fixed it")
    p.add_argument("--fix-summary", default=None,
                   help="if --status=fixed, one-line summary of the fix")
    p.add_argument("--notes", default=None)
    p.add_argument("--date", default=None,
                   help="discovery date YYYY-MM-DD; default today")
    args = p.parse_args()

    entries = load()
    bug_id = next_id(entries)

    entry = {
        "id": bug_id,
        "title": args.title,
        "discovered_at": args.date or dt.date.today().isoformat(),
        "discovered_in_commit": args.discovered_in_commit,
        "discovered_during": args.discovered_during,
        "severity": args.severity,
        "status": args.status,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "symptom": args.symptom,
        "repro": args.repro,
        "root_cause": args.root_cause,
        "fix": None,
        "notes": args.notes,
    }
    if args.status == "fixed":
        if not (args.fix_commit and args.fix_summary):
            sys.exit("ERROR: --status=fixed requires --fix-commit and --fix-summary")
        entry["fix"] = {"commit": args.fix_commit, "summary": args.fix_summary}

    try:
        validate(entry)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    append(entry)
    print(f"Logged {bug_id}: {args.title}")
    print(f"  severity={args.severity}  status={args.status}  tags={entry['tags']}")
    print("Run `python bugs/render.py` to refresh BUGS.md.")


if __name__ == "__main__":
    main()
