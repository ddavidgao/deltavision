#!/usr/bin/env python3
"""
Head-to-head FF vs DV — v1.0.5 classifier rerun (matched parallel SF run).

Same task, same starting state, both ran simultaneously on separate sheets.
Phantom-state DV steps culled (DV's sheet was pre-filled from a prior session).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_20_dv_v105/dv_proxy_run_1777089316.jsonl"
FF_LOG = ROOT / "benchmarks/mapsheets/results/run_21_ff_v105/dv_proxy_run_1777082861_ff.jsonl"

# DV's sheet was pre-filled with stale data; steps 16-21 are phantom navigation
# around that data before reaching a clean A1. Steps 16=sheet_loaded shows the
# stale rows; 17-21 = namebox attempts + escape + grid click + clear. By 21 the
# sheet is empty. FF's sheet was already empty at step 8.
DV_EXCLUDE_STEPS = {16, 17, 18, 19, 20, 21}


def load(p):
    return [json.loads(line) for line in open(p) if "step" in line]


def main():
    dv_all = load(DV_LOG)
    ff = load(FF_LOG)

    dv_kept = [s for s in dv_all if s["step"] not in DV_EXCLUDE_STEPS]
    culled_tokens = sum(s["dv_tokens"] for s in dv_all if s["step"] in DV_EXCLUDE_STEPS)

    ff_total = sum(s["dv_tokens"] for s in ff)  # FF mode: dv_tokens == ff_tokens
    dv_total_culled = sum(s["dv_tokens"] for s in dv_kept)
    dv_total_raw = sum(s["dv_tokens"] for s in dv_all)

    print("=" * 76)
    print("FF vs DV — v1.0.5 classifier rerun, matched parallel SF run")
    print("=" * 76)
    print(f"\nDV phantom steps removed: {sorted(DV_EXCLUDE_STEPS)} = {culled_tokens:,} tokens")

    print(f"\n{'':<28}{'FF':>14}{'DV (culled)':>14}")
    print(f"Total steps:               {len(ff):>14}{len(dv_kept):>14}")
    print(f"Total tokens:              {ff_total:>14,}{dv_total_culled:>14,}")
    print(f"Tokens / step:             {ff_total/len(ff):>14.0f}{dv_total_culled/len(dv_kept):>14.0f}")

    per_step = (1 - (dv_total_culled / len(dv_kept)) / (ff_total / len(ff))) * 100
    total = (1 - dv_total_culled / ff_total) * 100

    print("\n--- Per-step savings (HEADLINE: avg DV cost per frame vs FF's 1,365) ---")
    print(f"  {per_step:.1f}%")

    print("\n--- Total task savings (full trajectory) ---")
    print(f"  {total:.1f}%")

    print("\n--- For reference: raw (uncull'd) DV ---")
    print(f"  {(1 - dv_total_raw / ff_total) * 100:.1f}% total")
    print(f"  ({len(dv_all)} DV steps, {dv_total_raw:,} tokens incl. phantom-state)")

    out = {
        "run": "v1.0.5 matched parallel SF Maps→Sheets",
        "headline_per_step_savings_pct": round(per_step, 1),
        "total_task_savings_pct": round(total, 1),
        "ff": {"steps": len(ff), "total_tokens": ff_total},
        "dv_culled": {
            "steps": len(dv_kept),
            "total_tokens": dv_total_culled,
            "excluded_steps": sorted(DV_EXCLUDE_STEPS),
            "excluded_tokens": culled_tokens,
        },
    }
    out_path = ROOT / "benchmarks/mapsheets/results/ff_vs_dv_v105.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
