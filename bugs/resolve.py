#!/usr/bin/env python3
"""
Mark a bug as fixed/deferred/wontfix in bugs/log.jsonl.

Usage:
    python bugs/resolve.py BUG-0003 \
        --status fixed --commit abc1234 \
        --summary "Run merger after MAX_REGIONS truncation"

    python bugs/resolve.py BUG-0007 --status deferred \
        --notes "needs full re-tune of MIN_CONTOUR_AREA — defer to v1.1"

Rewrites the entry in place. Run `python bugs/render.py` after to refresh
BUGS.md.
"""
from __future__ import annotations

import argparse
import sys

from _lib import load, save, validate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bug_id", help="BUG-NNNN")
    p.add_argument("--status", required=True,
                   choices=["open", "fixed", "deferred", "wontfix", "cant_reproduce"])
    p.add_argument("--commit", default=None,
                   help="short SHA of the fix commit (required for --status=fixed)")
    p.add_argument("--summary", default=None,
                   help="one-line description of the fix (required for --status=fixed)")
    p.add_argument("--notes", default=None,
                   help="append/replace the notes field")
    p.add_argument("--root-cause", default=None,
                   help="set or update root_cause if you've now figured it out")
    args = p.parse_args()

    entries = load()
    matches = [i for i, e in enumerate(entries) if e.get("id") == args.bug_id]
    if not matches:
        sys.exit(f"ERROR: no entry with id={args.bug_id}")
    if len(matches) > 1:
        sys.exit(f"ERROR: duplicate id {args.bug_id} (this should never happen)")

    idx = matches[0]
    e = entries[idx]
    e["status"] = args.status
    if args.status == "fixed":
        if not (args.commit and args.summary):
            sys.exit("ERROR: --status=fixed requires --commit and --summary")
        e["fix"] = {"commit": args.commit, "summary": args.summary}
    if args.notes is not None:
        e["notes"] = args.notes
    if args.root_cause is not None:
        e["root_cause"] = args.root_cause

    try:
        validate(e)
    except ValueError as ve:
        sys.exit(f"ERROR: {ve}")

    save(entries)
    print(f"{args.bug_id} → status={args.status}")
    if args.status == "fixed":
        print(f"  fix: {args.commit} — {args.summary}")
    print("Run `python bugs/render.py` to refresh BUGS.md.")


if __name__ == "__main__":
    main()
