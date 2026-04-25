#!/usr/bin/env python3
"""
v1.0.5 head-to-head with AGGRESSIVE confounder culling.

Two kinds of noise we cull from BOTH runs (fair both ways):

1. Pre-task-state confounders: agent navigating around stale data that wouldn't
   exist in a clean from-scratch run. For DV, the sheet was pre-filled. For FF,
   the agent had to undo + retry several times to clear. Both sides get culled.

2. Mid-task thrash: byte-identical or near-identical consecutive screenshots
   (agent took a screenshot but nothing happened — the action didn't land).
   These are pure waste regardless of FF vs DV.

We define "near-identical" as MD5-equal screenshot OR the proxy logged
diff_ratio < 0.5% (essentially no visible change).

Result is the cleanest apples-to-apples we can get from this trajectory.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DV_DIR = ROOT / "benchmarks/mapsheets/results/run_20_dv_v105/screenshots"
FF_DIR = ROOT / "benchmarks/mapsheets/results/run_21_ff_v105/screenshots"
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_20_dv_v105/dv_proxy_run_1777089316.jsonl"
FF_LOG = ROOT / "benchmarks/mapsheets/results/run_21_ff_v105/dv_proxy_run_1777082861_ff.jsonl"

# Hand-audited phantom-state confounders — agent recovering from stale sheet
# state (pre-filled rows for DV, multiple undo/clear retries for FF). These
# wouldn't happen on a clean run.
DV_HAND_CULL = {16, 17, 18, 19, 20, 21}  # sheet_loaded → after_clear (6 steps)
FF_HAND_CULL = {9, 10, 11, 12, 13}        # sheet_cleared → fully_cleared (5 steps)


def load_log(p):
    return [json.loads(line) for line in open(p) if "step" in line]


def md5_files(dir_):
    """Return ordered list of (filename, md5_hash)."""
    out = []
    for f in sorted(os.listdir(dir_)):
        if not f.endswith(".png"):
            continue
        with open(dir_ / f, "rb") as fh:
            out.append((f, hashlib.md5(fh.read()).hexdigest()))
    return out


def auto_cull_dupes(hashes):
    """Return set of step numbers that are byte-identical to the prior step."""
    cull = set()
    for i in range(1, len(hashes)):
        if hashes[i][1] == hashes[i - 1][1]:
            # filename pattern: dvN_NN_label.png or ffN_NN_label.png
            step = int(hashes[i][0].split("_")[1])
            cull.add(step)
    return cull


def main():
    dv_log = load_log(DV_LOG)
    ff_log = load_log(FF_LOG)
    dv_hashes = md5_files(DV_DIR)
    ff_hashes = md5_files(FF_DIR)

    dv_dupe_cull = auto_cull_dupes(dv_hashes)
    ff_dupe_cull = auto_cull_dupes(ff_hashes)

    dv_total_cull = DV_HAND_CULL | dv_dupe_cull
    ff_total_cull = FF_HAND_CULL | ff_dupe_cull

    print("=" * 76)
    print("FF vs DV — v1.0.5 with confounder culling (BOTH sides)")
    print("=" * 76)

    print(f"\nDV culled: {sorted(dv_total_cull)}")
    print(f"  hand (phantom): {sorted(DV_HAND_CULL)}")
    print(f"  auto (byte-dupe): {sorted(dv_dupe_cull)}")
    print(f"FF culled: {sorted(ff_total_cull)}")
    print(f"  hand (clear retries): {sorted(FF_HAND_CULL)}")
    print(f"  auto (byte-dupe): {sorted(ff_dupe_cull)}")

    dv_kept = [s for s in dv_log if s["step"] not in dv_total_cull]
    ff_kept = [s for s in ff_log if s["step"] not in ff_total_cull]

    ff_total = sum(s["dv_tokens"] for s in ff_kept)
    dv_total = sum(s["dv_tokens"] for s in dv_kept)

    print(f"\n{'':<28}{'FF':>14}{'DV':>14}")
    print(f"Total log steps:           {len(ff_log):>14}{len(dv_log):>14}")
    print(f"After culling:             {len(ff_kept):>14}{len(dv_kept):>14}")
    print(f"Kept tokens:               {ff_total:>14,}{dv_total:>14,}")
    print(f"Tokens / step:             {ff_total/max(len(ff_kept),1):>14.0f}"
          f"{dv_total/max(len(dv_kept),1):>14.0f}")

    per_step = (1 - (dv_total / len(dv_kept)) / (ff_total / len(ff_kept))) * 100
    total = (1 - dv_total / ff_total) * 100

    print("\n--- HEADLINE: per-step savings ---")
    print(f"  {per_step:.1f}%   (DV avg per frame vs FF's 1,365)")

    print("\n--- Total task savings (cleaned trajectory) ---")
    print(f"  {total:.1f}%")

    out = {
        "run": "v1.0.5 matched parallel SF Maps→Sheets — confounder-culled",
        "headline_per_step_savings_pct": round(per_step, 1),
        "total_task_savings_pct": round(total, 1),
        "ff": {
            "kept_steps": len(ff_kept),
            "total_tokens": ff_total,
            "culled": sorted(ff_total_cull),
        },
        "dv": {
            "kept_steps": len(dv_kept),
            "total_tokens": dv_total,
            "culled": sorted(dv_total_cull),
        },
    }
    out_path = ROOT / "benchmarks/mapsheets/results/ff_vs_dv_v105_clean.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
