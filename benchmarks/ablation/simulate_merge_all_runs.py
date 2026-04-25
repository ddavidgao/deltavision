#!/usr/bin/env python3
"""
Replay every saved DV run through the new greedy-merge diff engine to see how
much the merge would save across different trajectories. This is an OFFLINE
simulation — no API calls, no costs.

For each run directory under benchmarks/mapsheets/results/run_*/screenshots/:
  - Step through consecutive frames
  - Compute the diff with merge ENABLED vs DISABLED
  - Sum DV-side tokens for both
  - Compare to FF synthetic total (n_steps × 1365)
  - Print savings under each policy

Output: per-run table + grand totals.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from config import DeltaVisionConfig
from vision.diff import (
    compute_diff,
    _bbox_token_cost,
    merge_bboxes_for_min_cost,
)

FULL_FRAME_TOKENS = 1365
CROP_BASE_TOKENS = 85
RUNS_DIR = ROOT / "benchmarks" / "mapsheets" / "results"


def cost_for_diff(diff_result, *, force_full=False):
    """
    Compute what the proxy would charge for this diff. Mirrors
    dv_playwright_mcp.py's logic: if total crop cost > full frame, fall back.
    """
    if force_full or not diff_result.changed_bboxes:
        if diff_result.diff_ratio < 0.005:
            return CROP_BASE_TOKENS  # cache-pointer fast exit
        return FULL_FRAME_TOKENS
    crop_total = sum(_bbox_token_cost(w, h)
                     for _, _, w, h in diff_result.changed_bboxes)
    crop_total = max(crop_total, CROP_BASE_TOKENS)
    if crop_total > FULL_FRAME_TOKENS:
        return FULL_FRAME_TOKENS  # token-cap fallback
    return crop_total


def replay_run(shots_dir: Path, *, merge_enabled: bool):
    cfg = DeltaVisionConfig(BBOX_MERGE_ENABLED=merge_enabled)
    files = sorted(p for p in shots_dir.iterdir() if p.suffix == ".png")
    if len(files) < 2:
        return None

    dv_total = FULL_FRAME_TOKENS  # initial frame
    ff_total = FULL_FRAME_TOKENS
    n_full_frame = 1
    n_cap_fallback = 0

    for i in range(1, len(files)):
        t0 = Image.open(files[i - 1]).convert("RGB")
        t1 = Image.open(files[i]).convert("RGB")
        # Some runs have viewport-size jumps (a screenshot mid-task at a
        # different size). Skip those frames cleanly rather than crashing.
        if t0.size != t1.size:
            ff_total += FULL_FRAME_TOKENS
            dv_total += FULL_FRAME_TOKENS
            n_full_frame += 1
            continue
        try:
            diff = compute_diff(t0, t1, cfg)
        except Exception:
            ff_total += FULL_FRAME_TOKENS
            dv_total += FULL_FRAME_TOKENS
            n_full_frame += 1
            continue
        cost = cost_for_diff(diff)
        dv_total += cost
        ff_total += FULL_FRAME_TOKENS
        if cost == FULL_FRAME_TOKENS:
            n_full_frame += 1
        # Detect cap fallback: input was a delta but the cost still hit the cap
        if diff.changed_bboxes:
            naive = sum(_bbox_token_cost(w, h) for _, _, w, h in diff.changed_bboxes)
            if naive > FULL_FRAME_TOKENS and cost == FULL_FRAME_TOKENS:
                n_cap_fallback += 1

    return {
        "n_steps": len(files),
        "dv_total": dv_total,
        "ff_total": ff_total,
        "savings_pct": (1 - dv_total / ff_total) * 100,
        "n_full_frame": n_full_frame,
        "n_cap_fallback": n_cap_fallback,
    }


def main():
    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        shots = run_dir / "screenshots"
        if not shots.exists():
            continue
        n = len([p for p in shots.iterdir() if p.suffix == ".png"])
        if n < 5:
            continue
        runs.append(run_dir)

    print(f"Replaying {len(runs)} runs through both engines...\n")
    print(f"{'run':<40}{'steps':>6}{'pre $/save':>14}{'post $/save':>14}{'Δ':>10}")
    print("-" * 84)

    grand_pre_dv = 0
    grand_post_dv = 0
    grand_ff = 0

    for run in runs:
        shots = run / "screenshots"
        before = replay_run(shots, merge_enabled=False)
        after = replay_run(shots, merge_enabled=True)
        if not before:
            continue
        delta = after["dv_total"] - before["dv_total"]
        pre_label = f"{before['dv_total']:>5,}/{before['savings_pct']:>4.1f}%"
        post_label = f"{after['dv_total']:>5,}/{after['savings_pct']:>4.1f}%"
        print(f"{run.name:<40}{before['n_steps']:>6}"
              f"{pre_label:>14}{post_label:>14}{delta:>+10,}")
        grand_pre_dv += before["dv_total"]
        grand_post_dv += after["dv_total"]
        grand_ff += before["ff_total"]

    print("-" * 84)
    print(f"\nGrand totals across {len(runs)} runs:")
    print(f"  FF tokens (synthetic):    {grand_ff:>9,}")
    print(f"  DV tokens (pre-merge):    {grand_pre_dv:>9,}  → "
          f"{(1 - grand_pre_dv/grand_ff) * 100:.1f}% savings")
    print(f"  DV tokens (post-merge):   {grand_post_dv:>9,}  → "
          f"{(1 - grand_post_dv/grand_ff) * 100:.1f}% savings")
    print(f"  Net improvement: {grand_pre_dv - grand_post_dv:,} tokens "
          f"({(grand_pre_dv - grand_post_dv) / grand_pre_dv * 100:.1f}% of pre-merge DV)")


if __name__ == "__main__":
    main()
