#!/usr/bin/env python3
"""
Simulate what DV would have cost on run_20 if the diff engine used a greedy
rectangle-merge optimizer instead of "send each contour as its own crop."

Cost model (matches dv_playwright_mcp.py):
  region_cost(w, h) = 85 + ceil(w/512) * ceil(h/512) * 170
  full_frame_tokens = 1365 (1280x800 frame)

Greedy merge:
  Start with all contour bboxes. While there exist two bboxes A,B such that
  cost(A∪B) < cost(A) + cost(B), merge the pair that saves the most.
  After merging, if total cost > full_frame_tokens, fall back to full frame.

This is the practical heuristic for the rectangle-cover optimization.
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from config import DeltaVisionConfig
from vision.diff import compute_diff

DV_DIR = ROOT / "benchmarks/mapsheets/results/run_20_dv_v105/screenshots"
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_20_dv_v105/dv_proxy_run_1777089316.jsonl"

BASE = 85
TILE = 170
FULL_FRAME = 1365


def region_cost(w, h):
    tw = max(1, math.ceil(w / 512))
    th = max(1, math.ceil(h / 512))
    return BASE + tw * th * TILE


def union_box(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x = min(ax, bx)
    y = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x, y, x2 - x, y2 - y)


def greedy_merge(boxes):
    """Merge pairs while it saves tokens. O(n³) worst case but n ≤ ~10."""
    boxes = [tuple(b) for b in boxes]
    while len(boxes) > 1:
        best_save = 0
        best = None
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                u = union_box(boxes[i], boxes[j])
                save = (region_cost(boxes[i][2], boxes[i][3])
                        + region_cost(boxes[j][2], boxes[j][3])
                        - region_cost(u[2], u[3]))
                if save > best_save:
                    best_save = save
                    best = (i, j, u)
        if best is None:
            break
        i, j, u = best
        boxes = [b for k, b in enumerate(boxes) if k != i and k != j] + [u]
    total = sum(region_cost(b[2], b[3]) for b in boxes)
    if total >= FULL_FRAME:
        # Fall back to full frame if greedy still loses
        return [(0, 0, 1280, 800)], FULL_FRAME, "fallback_full_frame"
    return boxes, total, "greedy_merged"


def main():
    cfg = DeltaVisionConfig()
    log_steps = {json.loads(l)["step"]: json.loads(l)
                 for l in open(DV_LOG) if "step" in l}

    files = sorted(p for p in DV_DIR.iterdir() if p.suffix == ".png")
    print(f"{'step':>4} {'orig':>6} {'merged':>7} {'save':>6}  status              kind")
    print("-" * 80)

    orig_total = 0
    merged_total = 0
    ff_total = 0

    for i in range(1, len(files)):
        step = int(files[i].name.split("_")[1])
        log = log_steps.get(step)
        if not log:
            continue

        orig = log["dv_tokens"]
        ff = log.get("ff_tokens", 1365)

        # Replay diff
        t0 = Image.open(files[i - 1]).convert("RGB")
        t1 = Image.open(files[i]).convert("RGB")
        diff = compute_diff(t0, t1, cfg)

        if log["transition"] in ("new_page", "initial"):
            # Classifier overrides the crops decision; honor it
            merged = FULL_FRAME
            kind = log["transition"]
        elif diff.diff_ratio < 0.005:
            # Cache-pointer fast exit (matches proxy's behavior on no-change)
            merged = BASE
            kind = "cache_pointer"
        else:
            _, merged, kind = greedy_merge(diff.changed_bboxes)

        save = orig - merged
        marker = "<-- BIG WIN" if save >= 500 else ""
        print(f"{step:>4} {orig:>6} {merged:>7} {save:>+6}  {log['transition']:<10} {log.get('trigger', ''):<14} {kind} {marker}")

        orig_total += orig
        merged_total += merged
        ff_total += ff

    print("-" * 80)
    print(f"\nOriginal DV total: {orig_total:>7,} tokens")
    print(f"Greedy-merge DV:   {merged_total:>7,} tokens")
    print(f"FF (synthetic):    {ff_total:>7,} tokens")
    print()
    print(f"Original savings:    {(1 - orig_total/ff_total)*100:>5.1f}%")
    print(f"Greedy-merge savings: {(1 - merged_total/ff_total)*100:>5.1f}%")
    print(f"Improvement: {(orig_total - merged_total):,} tokens "
          f"({(orig_total - merged_total)/orig_total*100:.1f}% of original DV)")


if __name__ == "__main__":
    main()
