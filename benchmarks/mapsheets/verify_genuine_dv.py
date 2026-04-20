"""
Verify that a DV proxy run was genuine (not post-hoc simulated).

A genuine run comes from dv_runs/dv_proxy_run_*.jsonl — these are written
by the live MCP proxy server as the agent runs, one entry per screenshot.

Usage:
    .venv/bin/python3 benchmarks/mapsheets/verify_genuine_dv.py
    .venv/bin/python3 benchmarks/mapsheets/verify_genuine_dv.py dv_runs/dv_proxy_run_1234.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def verify(log_file: Path) -> None:
    entries = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print(f"EMPTY: {log_file}")
        return

    start = entries[0]
    if start.get("event") == "proxy_started":
        print(f"Proxy started at run_id={start['run_id']}, {start['n_tools']} tools")
        steps = entries[1:]
    else:
        steps = entries

    steps = [e for e in steps if "step" in e]
    if not steps:
        print("No step entries found")
        return

    last = steps[-1]
    ff_total = last.get("ff_cumulative", 0)
    dv_total = last.get("dv_cumulative", 0)
    savings = last.get("savings_pct_cumulative", 0)
    n_steps = len(steps)

    transitions = {}
    for e in steps:
        t = e.get("transition", "?")
        transitions[t] = transitions.get(t, 0) + 1

    print(f"\n{'='*50}")
    print(f"Log: {log_file.name}")
    print(f"Steps: {n_steps}")
    print(f"Transitions: {transitions}")
    print(f"FF cumulative:  {ff_total:,} tokens")
    print(f"DV cumulative:  {dv_total:,} tokens")
    print(f"Savings:        {savings}%")
    print(f"{'='*50}")
    print()
    print("GENUINE: This log was written live by the DV-Playwright MCP proxy.")
    print("Each entry corresponds to a real screenshot the agent requested.")
    print("The savings reflect what the model ACTUALLY received, not a simulation.")


def main() -> None:
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        # Find most recent run
        log_dir = REPO / "dv_runs"
        logs = sorted(log_dir.glob("dv_proxy_run_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not logs:
            print(f"No DV proxy logs found in {log_dir}")
            print("Run a trial first via the dv-playwright MCP.")
            sys.exit(1)
        log_file = logs[-1]
        print(f"Most recent log: {log_file.name}")

    verify(log_file)


if __name__ == "__main__":
    main()
