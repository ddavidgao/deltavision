#!/usr/bin/env python3
"""
Head-to-head with phantom-state culling.

DV's sheet was pre-filled with stale data from a prior run; FF's sheet was empty.
DV had to dismiss a dialog + navigate around the stale data + delete it before
beginning its actual data-entry. Steps 8, 9, 10 of the DV run capture this
phantom work that FF never did. They are excluded from the comparison.

This makes the matched-trajectory comparison HONEST: same task, same starting
state.
"""
import json
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/dv_proxy_run_1777072627.jsonl"
FF_LOG = ROOT / "benchmarks/mapsheets/results/run_19_ff_parallel/dv_proxy_run_1777072642_ff.jsonl"
DV_SHOTS = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/screenshots"
FF_SHOTS = ROOT / "benchmarks/mapsheets/results/run_19_ff_parallel/screenshots"

# Phantom DV steps caused by stale sheet state (1-indexed).
# Step 8: pre-filled sheet visible
# Step 9: same data visible after dismissing the help dialog
# Step 10: namebox dropdown navigating around the old data
DV_EXCLUDE_STEPS = {8, 9, 10}


def load_steps(p):
    out = []
    for line in open(p):
        d = json.loads(line)
        if "step" in d:
            out.append(d)
    return out


def main():
    dv = load_steps(DV_LOG)
    ff = load_steps(FF_LOG)

    dv_kept = [s for s in dv if s["step"] not in DV_EXCLUDE_STEPS]
    dv_culled = [s for s in dv if s["step"] in DV_EXCLUDE_STEPS]

    ff_total = sum(s["dv_tokens"] for s in ff)  # FF mode: dv_tokens == ff_tokens
    dv_total_raw = sum(s["dv_tokens"] for s in dv)
    dv_total_culled = sum(s["dv_tokens"] for s in dv_kept)
    culled_tokens = sum(s["dv_tokens"] for s in dv_culled)

    print("=" * 76)
    print("FF vs DV — matched parallel run, phantom-state culled")
    print("=" * 76)
    print(f"\nDV phantom steps removed: {sorted(DV_EXCLUDE_STEPS)} = {culled_tokens:,} tokens")
    print("  (DV's sheet pre-filled with stale data; FF started clean)")

    print(f"\n{'':<28}{'FF':>14}{'DV (culled)':>14}")
    print(f"Total steps:               {len(ff):>14}{len(dv_kept):>14}")
    print(f"Total tokens:              {ff_total:>14,}{dv_total_culled:>14,}")
    print(f"Tokens / step:             {ff_total/len(ff):>14.0f}{dv_total_culled/len(dv_kept):>14.0f}")

    per_step = (1 - (dv_total_culled / len(dv_kept)) / (ff_total / len(ff))) * 100
    total = (1 - dv_total_culled / ff_total) * 100

    print("\n--- Per-step savings (DV avg per frame vs FF's fixed 1,365) ---")
    print(f"  {per_step:.1f}%   <-- HEADLINE: DV sends this much less per observation")

    print("\n--- Total task savings (full trajectory token cost) ---")
    print(f"  {total:.1f}%")

    print("\n--- For reference: raw (uncull'd) DV ---")
    raw_per_step = (1 - (dv_total_raw / len(dv)) / (ff_total / len(ff))) * 100
    raw_total = (1 - dv_total_raw / ff_total) * 100
    print(f"  Per-step: {raw_per_step:.1f}% | Total: {raw_total:.1f}%")
    print(f"  ({len(dv)} DV steps, {dv_total_raw:,} tokens incl. phantom-state)")

    out = {
        "headline_per_step_savings_pct": round(per_step, 1),
        "total_task_savings_pct": round(total, 1),
        "ff": {"steps": len(ff), "total_tokens": ff_total},
        "dv_culled": {
            "steps": len(dv_kept),
            "total_tokens": dv_total_culled,
            "excluded_steps": sorted(DV_EXCLUDE_STEPS),
            "excluded_tokens": culled_tokens,
        },
        "dv_raw": {"steps": len(dv), "total_tokens": dv_total_raw},
    }
    out_path = ROOT / "benchmarks/mapsheets/results/ff_vs_dv_culled.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
