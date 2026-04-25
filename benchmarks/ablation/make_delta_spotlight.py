#!/usr/bin/env python3
"""
Generate "delta spotlight" frames: full screenshot with everything dimmed/blurred
except the region that changed — showing exactly what DeltaVision feeds the model.

Usage:
  python make_delta_spotlight.py --run run_02 --out /tmp/dv_spotlight
"""
import argparse
import os
import glob
import re
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2

SCREENSHOTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "../mapsheets/results"
)

# How much to dim the non-delta area (0=black, 1=original)
DIM_ALPHA = 0.22
# Blur radius for non-delta area
BLUR_RADIUS = 6
# Padding around the detected bounding box
BBOX_PAD = 24
# Border color (cyan to match DV aesthetic)
BORDER_COLOR = (0, 229, 199)
BORDER_WIDTH = 4

# Label styling
LABEL_BG = (0, 229, 199)
LABEL_FG = (0, 0, 0)

# Min changed pixel fraction to show a spotlight (below this = no change)
MIN_DIFF_FRACTION = 0.002


def get_changed_bbox(img_before: np.ndarray, img_after: np.ndarray, pad: int = BBOX_PAD):
    """Return (x, y, w, h) bounding box of changed region, or None if no significant change."""
    diff = cv2.absdiff(img_before, img_after)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    changed_frac = mask.sum() / 255 / mask.size
    if changed_frac < MIN_DIFF_FRACTION:
        return None, changed_frac

    # Dilate to merge nearby regions
    kernel = np.ones((20, 20), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, changed_frac

    # Bounding box over all contours
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    x, y, w, h = cv2.boundingRect(all_pts)

    H, W = img_before.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + pad * 2)
    h = min(H - y, h + pad * 2)

    return (x, y, w, h), changed_frac


def make_spotlight_frame(img_pil: Image.Image, bbox, label: str, step_num: int, total_steps: int) -> Image.Image:
    """Dim everything outside bbox, keep bbox sharp, add cyan border + label."""
    img = img_pil.convert("RGB")
    W, H = img.size

    if bbox is None:
        # No change — show full dim with "NO CHANGE" label
        dimmed = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - DIM_ALPHA))))
        result = Image.alpha_composite(dimmed.convert("RGBA"), overlay).convert("RGB")
        _draw_label(result, "NO CHANGE — 85 tokens", (W // 2 - 160, 40), BORDER_COLOR)
        _draw_step_counter(result, step_num, total_steps)
        return result

    x, y, w, h = bbox

    # 1. Blurred + dimmed version of full image
    blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - DIM_ALPHA))))
    dimmed = Image.alpha_composite(blurred.convert("RGBA"), overlay).convert("RGB")

    # 2. Paste the sharp original over the bbox region
    sharp_crop = img.crop((x, y, x + w, y + h))
    dimmed.paste(sharp_crop, (x, y))

    # 3. Draw cyan border around the spotlight
    draw = ImageDraw.Draw(dimmed)
    for i in range(BORDER_WIDTH):
        draw.rectangle(
            [x - i, y - i, x + w + i, y + h + i],
            outline=BORDER_COLOR
        )

    # 4. Label above the box
    _draw_label(dimmed, label, (x, max(0, y - 34)))
    _draw_step_counter(dimmed, step_num, total_steps)

    return dimmed


def _draw_label(img: Image.Image, text: str, pos, color=BORDER_COLOR):
    draw = ImageDraw.Draw(img)
    x, y = pos
    # Simple pill background
    bbox = draw.textbbox((x + 6, y + 4), text)
    pad = 6
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=color)
    draw.text((x + 6, y + 4), text, fill=LABEL_FG)


def _draw_step_counter(img: Image.Image, step: int, total: int):
    draw = ImageDraw.Draw(img)
    W, _ = img.size
    text = f"STEP {step}/{total}  ·  DeltaVision active"
    draw.rectangle([W - 310, 14, W - 10, 44], fill=(0, 0, 0, 200))
    draw.text((W - 300, 18), text, fill=BORDER_COLOR)


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run_02", help="Run directory name")
    parser.add_argument("--out", default="/tmp/dv_spotlight", help="Output frame directory")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    parser.add_argument("--hold", type=float, default=2.5, help="Seconds to hold each frame")
    parser.add_argument("--video", default="", help="Output video path (empty = frames only)")
    args = parser.parse_args()

    screenshots_dir = os.path.join(SCREENSHOTS_DIR, args.run, "screenshots")
    if not os.path.isdir(screenshots_dir):
        print(f"ERROR: {screenshots_dir} not found")
        return

    files = sorted(
        glob.glob(os.path.join(screenshots_dir, "*.png")),
        key=lambda p: natural_sort_key(os.path.basename(p))
    )
    print(f"Found {len(files)} screenshots in {args.run}")

    os.makedirs(args.out, exist_ok=True)

    frames_out = []
    anchor_arr = None  # always diff against the first frame (anchor)

    for i, f in enumerate(files):
        img_pil = Image.open(f).convert("RGB")
        img_arr = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

        if anchor_arr is None:
            # First frame: show unmodified — it IS the anchor/full frame
            result = img_pil.convert("RGB")
            _draw_label(result, "FULL FRAME — new page", (20, 20))
            _draw_step_counter(result, 1, len(files))
            out_path = os.path.join(args.out, f"frame_{i:03d}.png")
            result.save(out_path)
            frames_out.append(out_path)
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} → FULL FRAME — new page")
            anchor_arr = img_bgr
            continue
        else:
            # Always diff against anchor (frame 0), not previous frame
            # This ensures each frame's spotlight shows what's NEW vs the baseline
            bbox, changed_frac = get_changed_bbox(anchor_arr, img_bgr)
            pct = int(changed_frac * 100)
            if bbox is None:
                label = f"NO CHANGE — 85 tokens  ({pct}% pixels)"
            else:
                x, y, w, h = bbox
                crop_px = w * h
                full_px = img_arr.shape[0] * img_arr.shape[1]
                est_tokens = max(85, int(crop_px / full_px * 1365))
                label = f"DELTA — ~{est_tokens} tokens  ({pct}% changed)"

        result = make_spotlight_frame(img_pil, bbox, label, i + 1, len(files))

        out_path = os.path.join(args.out, f"frame_{i:03d}.png")
        result.save(out_path)
        frames_out.append(out_path)
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} → {label}")

    print(f"\nSaved {len(frames_out)} spotlight frames to {args.out}")

    if args.video:
        hold_frames = max(1, int(args.fps * args.hold))
        frame_list_path = os.path.join(args.out, "frames.txt")
        with open(frame_list_path, "w") as fh:
            for p in frames_out:
                for _ in range(hold_frames):
                    fh.write(f"file '{p}'\n")
                    fh.write(f"duration {1/args.fps:.4f}\n")

        os.system(
            f'ffmpeg -y -f concat -safe 0 -i "{frame_list_path}" '
            f'-vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" '
            f'-c:v libx264 -crf 18 -pix_fmt yuv420p "{args.video}"'
        )
        print(f"Video saved to {args.video}")


if __name__ == "__main__":
    main()
