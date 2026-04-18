"""
Stateless CLI wrapper around DeltaVision's observation pipeline.

For the V1 Sonnet-subagent ablation:
  - A Sonnet subagent drives Chrome via Claude-in-Chrome MCP tools
  - It captures screenshots (t0, t1) and passes them to this CLI
  - This CLI runs DeltaVision's classifier cascade + crop extraction
  - Returns JSON observation the subagent can read and reason about

Design principles:
  - Stateless: every call takes the full inputs it needs (no session)
  - JSON out: subagent reads stdout or --output file
  - Path-based crop output: subagent gets image paths it can load individually

Usage:
  # DeltaVision mode (delta gating on):
  python dv_observe.py \\
      --mode delta \\
      --t0 before.png \\
      --t1 after.png \\
      --url-before "https://example.com" \\
      --url-after "https://example.com" \\
      --anchor anchor.png \\
      --last-action "click(100, 200)" \\
      --no-change-count 0 \\
      --output-dir /tmp/step_03/ \\
      --step 3

  # Full-frame mode (baseline, no classifier):
  python dv_observe.py --mode full_frame --t1 after.png --step 3 --output-dir /tmp/step_03/
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DeltaVisionConfig
from vision.classifier import TransitionType, classify_transition
from vision.diff import compute_diff, extract_crops


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_delta(args) -> dict:
    config = DeltaVisionConfig()
    out_dir = _ensure_dir(Path(args.output_dir))

    t0 = Image.open(args.t0).convert("RGB")
    t1 = Image.open(args.t1).convert("RGB")
    anchor = Image.open(args.anchor).convert("RGB") if args.anchor else t0

    last_action_type = None
    if args.last_action:
        # "click(100, 200)" → "click"
        paren = args.last_action.find("(")
        last_action_type = args.last_action[:paren] if paren > 0 else args.last_action.strip()

    # 4-layer classifier cascade
    diff = compute_diff(t0, t1, config)
    cls = classify_transition(
        t0=t0,
        t1=t1,
        url_before=args.url_before or "",
        url_after=args.url_after or "",
        anchor_template=anchor,
        config=config,
        diff_result=diff,
        last_action_type=last_action_type,
    )

    # action_had_effect is already computed on the DiffResult itself
    action_had_effect = diff.action_had_effect

    result = {
        "mode": "delta",
        "step": args.step,
        "transition": cls.transition.value,
        "trigger": cls.trigger,
        "diff_ratio": round(cls.diff_ratio, 4),
        "phash_distance": int(cls.phash_distance),
        "anchor_score": round(cls.anchor_score, 4),
        "action_had_effect": bool(action_had_effect),
        "no_change_count": args.no_change_count,
        "last_action": args.last_action,
        "crops": [],
    }

    # If NEW_PAGE, subagent should receive full current frame (not crops)
    if cls.transition == TransitionType.NEW_PAGE:
        result["obs_type"] = "full_frame"
        result["current_frame_path"] = str(Path(args.t1).resolve())
        result["trigger_reason"] = cls.trigger
    else:
        result["obs_type"] = "delta"
        # Extract and save crops of changed regions (up to 3, sorted by size)
        bboxes = sorted(diff.changed_bboxes, key=lambda b: b[2] * b[3], reverse=True)[:3]
        crops = extract_crops(t0, t1, bboxes)
        for i, c in enumerate(crops):
            x, y, w, h = c["bbox"]
            before_path = out_dir / f"crop_{i}_before.png"
            after_path = out_dir / f"crop_{i}_after.png"
            c["crop_before"].save(before_path)
            c["crop_after"].save(after_path)
            result["crops"].append({
                "index": i,
                "bbox": [int(x), int(y), int(w), int(h)],
                "change_magnitude": round(float(c["change_magnitude"]), 4),
                "crop_before_path": str(before_path),
                "crop_after_path": str(after_path),
            })

        # Always provide a thumbnail of t1 for spatial context
        thumb_path = out_dir / "thumbnail.png"
        thumb = t1.resize((320, 225), Image.LANCZOS)
        thumb.save(thumb_path)
        result["thumbnail_path"] = str(thumb_path)
        result["current_frame_path"] = str(Path(args.t1).resolve())

    return result


def run_full_frame(args) -> dict:
    """FF mode: just report the full frame, no classifier. This is the paper's
    ablation baseline — every step sees the same screenshot format."""
    return {
        "mode": "full_frame",
        "obs_type": "full_frame",
        "step": args.step,
        "current_frame_path": str(Path(args.t1).resolve()),
        "last_action": args.last_action,
        "trigger_reason": "full_frame_ablation",
    }


def main():
    p = argparse.ArgumentParser(description="DeltaVision observation CLI (stateless)")
    p.add_argument("--mode", choices=["delta", "full_frame"], required=True)
    p.add_argument("--t0", help="Before screenshot (required for delta mode)")
    p.add_argument("--t1", required=True, help="After screenshot")
    p.add_argument("--url-before", default="")
    p.add_argument("--url-after", default="")
    p.add_argument("--anchor", help="Anchor template image for Layer 4 (falls back to t0)")
    p.add_argument("--last-action", default="", help="e.g. click(100, 200)")
    p.add_argument("--no-change-count", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--output-dir", default="/tmp/dv_obs", help="Where to save crops/thumbnails")
    p.add_argument("--output", help="JSON output path (default: stdout)")
    args = p.parse_args()

    if args.mode == "delta":
        if not args.t0:
            p.error("--t0 required for delta mode")
        result = run_delta(args)
    else:
        result = run_full_frame(args)

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
