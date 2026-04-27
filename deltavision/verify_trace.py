"""
`deltavision verify-trace <path>` — independent validator for a benchmark trace.

The point of this command is credibility: anyone can take a trace file from a
DeltaVision benchmark, run `deltavision verify-trace path/to/trace.jsonl`, and
get a yes/no answer on whether the trace's stated savings are internally
consistent. They don't need to clone DV or trust our numbers — the trace and
the verifier are enough.

What it checks today (v1, this skeleton):
    1. JSONL is structurally valid (header, steps, optional summary).
    2. Required schema fields are present in every record.
    3. observation_mode + obs_type values are in the allowed set.
    4. schema_version matches what this validator knows.
    5. Token totals add up: model_facing ≤ dv_internal per step, recomputed
       totals match the summary line, and savings_pct_total is consistent.
    6. If a step has frame_path or payload_path set, the file exists and
       its sha256 matches the recorded hash.

What it does NOT check yet (planned, growing the verifier in lockstep with
new benchmark surface):
    * That delta-payload images contain only crops, never a full frame
      (the "no append" invariant). Requires a payload-introspection step
      that knows how to parse each adapter's payload format.
    * Cross-trial paired comparison: did FF and DV traces from the same
      trial_group_id agree on task definition, model, seed?
    * Replay: re-run the diff engine on stored frames and confirm bboxes
      match what the trace claims.

Exit codes:
    0   trace passed validation (errors empty)
    1   trace had structural or invariant errors
    2   bad CLI usage (missing path, unknown flag)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from results.trace import parse_trace, validate_trace


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="deltavision verify-trace",
        description="Validate a DeltaVision benchmark trace file.",
    )
    p.add_argument("path", type=Path, help="path to a .jsonl trace file")
    p.add_argument(
        "--no-paths", action="store_true",
        help="skip the optional file-on-disk hash check. Use this if you're "
             "validating a trace whose frames/payloads were not bundled.",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="only print errors; suppress the per-section summary on success.",
    )
    args = p.parse_args(argv)

    try:
        trace = parse_trace(args.path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    report = validate_trace(trace, check_paths=not args.no_paths)

    if not report.ok:
        print(f"FAIL  {args.path}", file=sys.stderr)
        print(f"  {len(report.errors)} error(s):", file=sys.stderr)
        for e in report.errors:
            print(f"    - {e}", file=sys.stderr)
        if report.warnings:
            print(f"  {len(report.warnings)} warning(s):", file=sys.stderr)
            for w in report.warnings:
                print(f"    - {w}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"OK    {args.path}")
        print(f"  trace_id          {trace.header.get('trace_id')}")
        print(f"  observation_mode  {trace.header.get('observation_mode')}")
        print(f"  task_id           {trace.header.get('task_id')}")
        print(f"  model             {trace.header.get('model')}")
        print(f"  steps             {report.n_steps}")
        print(f"  dv_internal_tok   {report.total_dv_internal_tokens:,}")
        print(f"  model_facing_tok  {report.total_model_facing_tokens:,}")
        print(f"  savings_pct       {report.savings_pct_total:.2f}%")
        if report.warnings:
            print(f"  warnings          {len(report.warnings)}")
            for w in report.warnings:
                print(f"    · {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
