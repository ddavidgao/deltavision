#!/usr/bin/env python3
"""
Proper delta spotlight:
- Full-size screenshots, readable content
- Heavy blur + dim on everything except the actual changed region
- Cyan bounding box around what DV actually sends to the model
- Running token counter (cumulative FF vs DV, savings %)
- Pulls from chicago FF run screenshots (full-size, crisp)

Output: /tmp/proper_spotlight/proper_spotlight.mp4
"""
import argparse
import os
import glob
import re
import json
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2

SRC = "benchmarks/mapsheets/results/ff_chicago_20260420_030657/screenshots"
OUT_DIR = "/tmp/proper_spotlight"
OUT_VIDEO = os.path.join(OUT_DIR, "proper_spotlight.mp4")

# ---- Look & feel ----
DIM_ALPHA = 0.12       # non-delta area keeps only 12% of original brightness
BLUR_RADIUS = 14       # heavy blur
BBOX_PAD = 32
BORDER_COLOR = (0, 229, 199)
BORDER_WIDTH = 5
MIN_DIFF_FRACTION = 0.003

# Per-frame hold time (seconds)
HOLD_S = 2.0
FPS = 24

# Token model
FF_TOKENS_PER_FRAME = 1365
# DV tokens = base 85 + area_ratio * 1365
DV_BASE = 85


def get_bbox(img_before: np.ndarray, img_after: np.ndarray, pad: int = BBOX_PAD):
    diff = cv2.absdiff(img_before, img_after)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    changed_frac = mask.sum() / 255 / mask.size
    if changed_frac < MIN_DIFF_FRACTION:
        return None, changed_frac
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, changed_frac
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    x, y, w, h = cv2.boundingRect(all_pts)
    H, W = img_before.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + pad * 2)
    h = min(H - y, h + pad * 2)
    return (x, y, w, h), changed_frac


def font(sz):
    for path in [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def render_spotlight(img_pil, bbox, ff_cum, dv_cum, step_num, total_steps, delta_tokens):
    """Blur/dim non-bbox region, sharp bbox, overlay token panel."""
    img = img_pil.convert("RGB")
    W, H = img.size

    # If bbox covers ≥80% of screen, treat as full-frame NEW_PAGE: show unmodified
    is_full_frame = False
    if bbox is not None:
        _, _, bw, bh = bbox
        area_frac = (bw * bh) / (W * H)
        if area_frac >= 0.8:
            is_full_frame = True

    if is_full_frame:
        base = img.copy()
    else:
        # --- Base layer: heavy blur + dim ---
        blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - DIM_ALPHA))))
        base = Image.alpha_composite(blurred.convert("RGBA"), overlay).convert("RGB")

        # --- Spotlight: paste sharp bbox region over base ---
        if bbox is not None:
            x, y, w, h = bbox
            sharp = img.crop((x, y, x + w, y + h))
            base.paste(sharp, (x, y))

            # cyan border
            draw = ImageDraw.Draw(base)
            for i in range(BORDER_WIDTH):
                draw.rectangle(
                    [x - i, y - i, x + w + i, y + h + i],
                    outline=BORDER_COLOR,
                )

    # --- Top overlay: token panel (across full width) ---
    draw = ImageDraw.Draw(base)
    panel_h = 86
    draw.rectangle([0, 0, W, panel_h], fill=(6, 10, 14))

    f_big = font(32)
    f_med = font(20)
    f_sm  = font(14)

    # Left: step counter
    draw.text((24, 18), f"STEP {step_num:02d}/{total_steps:02d}", font=f_sm, fill=(140, 140, 140))
    draw.text((24, 38), "DeltaVision active", font=f_med, fill=BORDER_COLOR)

    # Center: FF vs DV running totals
    ff_txt = f"FF: {ff_cum:>6,}"
    dv_txt = f"DV: {dv_cum:>6,}"
    saved = int(round((1 - dv_cum / max(ff_cum, 1)) * 100))
    save_txt = f"SAVED {saved:>2d}%"

    cx = W // 2
    draw.text((cx - 260, 18), "FULL FRAME BASELINE", font=f_sm, fill=(220, 90, 100))
    draw.text((cx - 260, 38), ff_txt, font=f_big, fill=(220, 120, 130))
    draw.text((cx + 40, 18), "DELTAVISION", font=f_sm, fill=BORDER_COLOR)
    draw.text((cx + 40, 38), dv_txt, font=f_big, fill=BORDER_COLOR)

    # Right: savings %
    draw.rectangle([W - 200, 14, W - 20, panel_h - 14],
                   outline=BORDER_COLOR, width=2)
    draw.text((W - 188, 22), "SAVED", font=f_sm, fill=(140, 140, 140))
    draw.text((W - 188, 42), f"{saved}%", font=f_big, fill=BORDER_COLOR)

    # Bottom label: what delta was sent this step
    bot_h = 44
    draw.rectangle([0, H - bot_h, W, H], fill=(6, 10, 14))
    if bbox is None:
        msg = f"NO CHANGE  —  sent {delta_tokens} tokens (cached pointer)"
    elif is_full_frame:
        msg = f"NEW_PAGE  —  sent {delta_tokens} tokens (full frame, new context)"
    else:
        x, y, w, h = bbox
        msg = f"DELTA  —  sent {delta_tokens} tokens  ({w}x{h}px region)"
    draw.text((24, H - bot_h + 12), msg, font=f_med, fill=BORDER_COLOR)

    return base


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=SRC)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--max-frames", type=int, default=14,
                        help="Cap number of frames rendered")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.src, "*.png")),
                       key=lambda p: natural_key(os.path.basename(p)))
    print(f"Found {len(all_files)} screenshots in {args.src}")

    # Curated: first 3 frames show NEW_PAGE full-frame (Maps navigation, big bbox)
    # Then jump to frames with tight localized deltas (sidebar, dialog, data row)
    # Finally end on the sort-dialog + completed sheet (small dialog delta on existing data)
    curated_indices = [0, 1, 2, 13, 14, 15, 16, 21, 22]
    files = [all_files[i] for i in curated_indices if i < len(all_files)]
    print(f"Using {len(files)} curated frames")

    prev_arr = None
    frame_paths = []
    ff_cum = 0
    dv_cum = 0

    for i, f in enumerate(files):
        img_pil = Image.open(f).convert("RGB")
        img_arr = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

        ff_cum += FF_TOKENS_PER_FRAME

        if prev_arr is None:
            # First frame — full frame sent
            bbox = None
            delta_tokens = FF_TOKENS_PER_FRAME
            dv_cum += delta_tokens
        else:
            bbox, frac = get_bbox(prev_arr, img_bgr)
            if bbox is None:
                delta_tokens = DV_BASE
            else:
                _, _, w, h = bbox
                full_px = img_arr.shape[0] * img_arr.shape[1]
                ratio = (w * h) / full_px
                delta_tokens = max(DV_BASE, int(ratio * FF_TOKENS_PER_FRAME))
            dv_cum += delta_tokens

        result = render_spotlight(
            img_pil, bbox, ff_cum, dv_cum,
            step_num=i + 1, total_steps=len(files),
            delta_tokens=delta_tokens,
        )

        out_path = os.path.join(args.out, f"frame_{i:03d}.png")
        result.save(out_path)
        frame_paths.append(out_path)
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} "
              f"ff={ff_cum} dv={dv_cum} saved={int((1-dv_cum/max(ff_cum,1))*100)}%")

        prev_arr = img_bgr

    # Build concat list with hold
    list_path = os.path.join(args.out, "frames.txt")
    hold_frames = int(FPS * HOLD_S)
    with open(list_path, "w") as fh:
        for p in frame_paths:
            for _ in range(hold_frames):
                fh.write(f"file '{p}'\n")
                fh.write(f"duration {1/FPS:.4f}\n")

    # Render video at 1920x1080
    os.system(
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
        f'-vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps={FPS}" '
        f'-c:v libx264 -crf 18 -pix_fmt yuv420p "{OUT_VIDEO}" 2>/dev/null'
    )
    print(f"\nVideo: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
