"""
Measure FF vs DV token cost for a screenshot sequence captured during a subagent run.

Usage:
    python benchmarks/mapsheets/measure_tokens.py results/run_01/screenshots/

For each consecutive pair (step_N, step_N+1), runs DeltaVision's classifier to
decide DELTA or NEW_PAGE. Computes:
  - FF tokens: always 1,365 per step (1280x800 @ low detail)
  - DV tokens: full frame (1,365) on NEW_PAGE, estimated crop tokens on DELTA

Output: JSON metrics + per-step log printed to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import re

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from PIL import Image

from config import DeltaVisionConfig
from vision.classifier import classify_transition, extract_anchor, TransitionType
from vision.diff import compute_diff, extract_crops

# Token cost constants (Anthropic vision, 1280x800, low detail = 1 tile = 85 + 85 = 170?
# Actually Claude image tokens: low detail = 85 tokens flat.
# High detail 1280x800: ceil(1280/512)*ceil(800/512) = 3*2 = 6 tiles * 170 + 85 = 1105
# Measured empirically in head-to-head: ~1365/step for full frame. Use that.
FULL_FRAME_TOKENS = 1365
CROP_BASE_TOKENS = 85   # minimum per crop (low-detail tile)
CROP_PER_TILE = 170     # tokens per 512x512 tile in a crop


def estimate_crop_tokens(crops: list) -> int:
    """Estimate tokens for a list of PIL crop images."""
    total = 0
    for crop in crops:
        if crop is None:
            continue
        w, h = crop.size
        tiles_w = max(1, (w + 511) // 512)
        tiles_h = max(1, (h + 511) // 512)
        total += CROP_BASE_TOKENS + tiles_w * tiles_h * CROP_PER_TILE
    return max(total, CROP_BASE_TOKENS)  # at least one tile


def measure_run(screenshot_dir: Path, config: DeltaVisionConfig | None = None) -> dict:
    if config is None:
        config = DeltaVisionConfig()

    # Load screenshots in order
    files = sorted(
        screenshot_dir.glob("step_*.png"),
        key=lambda p: int(re.search(r"step_(\d+)", p.stem).group(1))
    )
    if len(files) < 2:
        return {"error": f"need at least 2 screenshots, found {len(files)}"}

    frames = [Image.open(f).convert("RGB") for f in files]
    n_steps = len(frames) - 1  # number of transitions

    ff_tokens = 0
    dv_tokens = 0
    step_log = []

    # Step 0 always pays full frame (initial observation)
    ff_tokens += FULL_FRAME_TOKENS
    dv_tokens += FULL_FRAME_TOKENS

    t0 = frames[0]
    anchor = extract_anchor(t0, config)

    for i in range(n_steps):
        t1 = frames[i + 1]

        diff_result = compute_diff(t0, t1, config)
        result = classify_transition(
            t0=t0,
            t1=t1,
            url_before="os://browser",  # unknown — no URL metadata
            url_after="os://browser",
            anchor_template=anchor,
            config=config,
            diff_result=diff_result,
        )

        ff_step = FULL_FRAME_TOKENS
        ff_tokens += ff_step

        if result.transition == TransitionType.NEW_PAGE:
            dv_step = FULL_FRAME_TOKENS
            t0 = t1
            anchor = extract_anchor(t0, config)
        else:
            crops = extract_crops(t0, t1, diff_result.changed_bboxes, config.CROP_PADDING)
            crop_imgs = [c["crop_after"] for c in crops] if crops else []
            dv_step = estimate_crop_tokens(crop_imgs) if crop_imgs else CROP_BASE_TOKENS

        dv_tokens += dv_step

        step_log.append({
            "step": i + 1,
            "file": files[i + 1].name,
            "transition": result.transition.value,
            "trigger": result.trigger,
            "diff_ratio": round(result.diff_ratio, 3),
            "phash_distance": result.phash_distance,
            "ff_tokens": ff_step,
            "dv_tokens": dv_step,
        })

        if result.transition == TransitionType.DELTA:
            t0 = t1  # update anchor on scroll

    savings_pct = round((ff_tokens - dv_tokens) / ff_tokens * 100, 1) if ff_tokens else 0.0
    n_full = sum(1 for s in step_log if s["transition"] == "new_page")
    n_delta = sum(1 for s in step_log if s["transition"] == "delta")

    metrics = {
        "screenshot_dir": str(screenshot_dir),
        "n_screenshots": len(frames),
        "n_transitions": n_steps,
        "n_full_frames": n_full + 1,  # +1 for step 0
        "n_deltas": n_delta,
        "ff_total_tokens": ff_tokens,
        "dv_total_tokens": dv_tokens,
        "savings_pct": savings_pct,
        "step_log": step_log,
    }
    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python measure_tokens.py <screenshot_dir>")
        sys.exit(1)

    shot_dir = Path(sys.argv[1])
    if not shot_dir.exists():
        print(f"Directory not found: {shot_dir}")
        sys.exit(1)

    metrics = measure_run(shot_dir)
    print(json.dumps(metrics, indent=2))

    if "error" not in metrics:
        print("\n--- Summary ---")
        print(f"Steps: {metrics['n_transitions']}")
        print(f"Full frames: {metrics['n_full_frames']}  Deltas: {metrics['n_deltas']}")
        print(f"FF tokens:  {metrics['ff_total_tokens']:,}")
        print(f"DV tokens:  {metrics['dv_total_tokens']:,}")
        print(f"Savings:    {metrics['savings_pct']}%")


if __name__ == "__main__":
    main()
