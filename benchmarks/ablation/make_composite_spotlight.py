#!/usr/bin/env python3
"""
Proper composite frames matching the original video aesthetic:
- LEFT: full-size SF screenshot (from run_02), blurred except the delta bbox
- RIGHT: proxy log panel showing FF cumulative / DV cumulative / SAVED %
  with a step list where current step is highlighted
- TOP: "DELTAVISION LIVE PROXY · SAN FRANCISCO · dv_proxy_run_1776665530.jsonl"
- BOTTOM: "DELTA — 510 tokens (vs 1,365 FF)" tag

Uses SF DV run log (real data) + run_02 screenshots (same SF workflow).
"""
import os
import json
import re
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageFont

SCREENSHOTS_DIR = "benchmarks/mapsheets/results/run_14_sf_fixed/screenshots"
LOG_PATH = "dv_runs/dv_proxy_run_1777061077.jsonl"
OUT_DIR = "/tmp/composite_spotlight_v2"
OUT_VIDEO = os.path.join(OUT_DIR, "composite_spotlight_v2.mp4")

# Canvas dimensions
CANVAS_W = 1920
CANVAS_H = 1080
LEFT_W = 1070       # browser panel width
RIGHT_X = LEFT_W + 20

# Colors
BG = (6, 10, 14)
CYAN = (0, 229, 199)
RED = (220, 90, 100)
FG = (220, 220, 220)
DIM = (100, 100, 100)
HIGHLIGHT_BG = (12, 28, 34)

# Blur settings: defocus only, keep everything visible
BLUR_RADIUS = 6
DIM_ALPHA = 0.75  # non-delta area keeps 75% brightness — visible but defocused

FPS = 24
HOLD_S = 1.1  # ~1.1s per step, matches FF section pacing


def load_log():
    steps = []
    with open(LOG_PATH) as fh:
        for line in fh:
            d = json.loads(line)
            if "step" in d:
                steps.append(d)
    return steps


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def load_screenshots():
    files = sorted(
        [f for f in os.listdir(SCREENSHOTS_DIR) if f.endswith(".png")],
        key=natural_key,
    )
    return [os.path.join(SCREENSHOTS_DIR, f) for f in files]


def font(sz, mono=True):
    paths = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ] if mono else [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def get_bbox(img_before, img_after, pad=32, min_frac=0.003):
    """Return bbox of changed region between two BGR images."""
    diff = cv2.absdiff(img_before, img_after)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    frac = mask.sum() / 255 / mask.size
    if frac < min_frac:
        return None, frac
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frac
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    x, y, w, h = cv2.boundingRect(all_pts)
    H, W = img_before.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + pad * 2)
    h = min(H - y, h + pad * 2)
    return (x, y, w, h), frac


def render_browser_panel(screenshot_path, bbox, target_w, target_h):
    """Render the left browser panel: full screenshot fitted to target, with blur outside bbox."""
    img = Image.open(screenshot_path).convert("RGB")
    W, H = img.size

    # Scale bbox coordinates from original to resized space
    scale_w = target_w / W
    scale_h = target_h / H
    scale = min(scale_w, scale_h)  # maintain aspect ratio
    new_w = int(W * scale)
    new_h = int(H * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    if bbox is None:
        # No change — light blur uniformly to indicate "cached"
        return img_resized.filter(ImageFilter.GaussianBlur(radius=3))

    # Check if bbox covers most of the image (NEW_PAGE full frame only)
    # Only skip blur if it's a declared NEW_PAGE AND covers >=90% (i.e. genuinely a new page)
    x, y, w, h = bbox
    area_frac = (w * h) / (W * H)
    if area_frac >= 0.95:
        # Full-frame NEW_PAGE: show sharp, no blur
        return img_resized

    # Scale bbox to resized coords
    sx = int(x * scale)
    sy = int(y * scale)
    sw = int(w * scale)
    sh = int(h * scale)

    # Create blurred + dimmed version of entire image
    blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    overlay = Image.new("RGBA", (new_w, new_h), (0, 0, 0, int(255 * (1 - DIM_ALPHA))))
    base = Image.alpha_composite(blurred.convert("RGBA"), overlay).convert("RGB")

    # Paste sharp bbox region
    sharp_crop = img_resized.crop((sx, sy, sx + sw, sy + sh))
    base.paste(sharp_crop, (sx, sy))

    # Cyan border
    draw = ImageDraw.Draw(base)
    for i in range(4):
        draw.rectangle([sx - i, sy - i, sx + sw + i, sy + sh + i], outline=CYAN)

    return base


def render_composite(screenshot_path, bbox, step_data, all_steps, step_idx):
    """Build full 1920x1080 composite frame."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)

    # --- Top header bar ---
    f_hdr = font(14)
    f_hdr_bold = font(18)
    draw.text((30, 12), "DELTAVISION LIVE PROXY · SAN FRANCISCO, CA · dv_proxy_run_1777061077.jsonl",
              font=f_hdr, fill=DIM)
    draw.text((30, 32), "CV classifier intercepts every screenshot — sends only what changed",
              font=f_hdr_bold, fill=FG)

    # Step counter top right
    total = len(all_steps)
    step_num = step_data["step"]
    f_step_big = font(56)
    f_step_small = font(18)
    sx_box = CANVAS_W - 170
    draw.rectangle([sx_box, 20, CANVAS_W - 30, 90], outline=DIM, width=1)
    draw.text((sx_box + 20, 22), "STEP", font=f_step_small, fill=DIM)
    step_txt = f"{step_num:02d}"
    draw.text((sx_box + 20, 36), step_txt, font=f_step_big, fill=FG)
    w_bbox = draw.textbbox((0, 0), step_txt, font=f_step_big)
    draw.text((sx_box + 30 + (w_bbox[2] - w_bbox[0]), 70), f"/{total}",
              font=f_step_small, fill=DIM)

    # --- LEFT: browser panel ---
    panel_x = 0
    panel_y = 90
    panel_h = CANVAS_H - panel_y - 50
    browser = render_browser_panel(screenshot_path, bbox, LEFT_W, panel_h)
    bw, bh = browser.size
    canvas.paste(browser, (panel_x + (LEFT_W - bw) // 2, panel_y))

    # --- RIGHT: cumulative token panels ---
    right_w = CANVAS_W - RIGHT_X - 20
    card_h = 100
    card_gap = 14
    card_y = 100

    f_card_label = font(13)
    f_card_num = font(48)

    # FF card
    ff_card_w = (right_w - card_gap * 2) // 3
    draw.rectangle([RIGHT_X, card_y, RIGHT_X + ff_card_w, card_y + card_h],
                   fill=(30, 15, 20), outline=RED, width=1)
    draw.text((RIGHT_X + 14, card_y + 10), "FF WOULD COST", font=f_card_label, fill=RED)
    draw.text((RIGHT_X + 14, card_y + 30), f"{step_data['ff_cumulative']:,}",
              font=f_card_num, fill=RED)

    # DV card
    dv_card_x = RIGHT_X + ff_card_w + card_gap
    draw.rectangle([dv_card_x, card_y, dv_card_x + ff_card_w, card_y + card_h],
                   fill=(15, 30, 28), outline=CYAN, width=1)
    draw.text((dv_card_x + 14, card_y + 10), "DV ACTUAL", font=f_card_label, fill=CYAN)
    draw.text((dv_card_x + 14, card_y + 30), f"{step_data['dv_cumulative']:,}",
              font=f_card_num, fill=CYAN)

    # SAVED card
    saved_card_x = dv_card_x + ff_card_w + card_gap
    saved_pct = step_data.get("savings_pct_cumulative", 0)
    draw.rectangle([saved_card_x, card_y, saved_card_x + ff_card_w, card_y + card_h],
                   fill=(15, 30, 28), outline=CYAN, width=1)
    draw.text((saved_card_x + 14, card_y + 10), "SAVED", font=f_card_label, fill=CYAN)
    draw.text((saved_card_x + 14, card_y + 30),
              f"{saved_pct:.0f}%", font=f_card_num, fill=CYAN)

    # --- RIGHT: step log list ---
    log_y = card_y + card_h + 30
    f_log_head = font(11)
    draw.text((RIGHT_X + 14, log_y), "#        DECISION        PIXEL DIFF",
              font=f_log_head, fill=DIM)
    draw.text((CANVAS_W - 130, log_y), "DV TOKENS", font=f_log_head, fill=DIM)
    log_y += 22

    # Show a window of steps around current
    window = 7
    start_idx = max(0, step_idx - 3)
    end_idx = min(len(all_steps), start_idx + window)
    row_h = 40
    f_row = font(13)
    f_num = font(14)

    for i in range(start_idx, end_idx):
        s = all_steps[i]
        row_y = log_y + (i - start_idx) * row_h
        is_current = i == step_idx

        if is_current:
            draw.rectangle([RIGHT_X, row_y - 6, CANVAS_W - 20, row_y + row_h - 10],
                           fill=HIGHLIGHT_BG, outline=CYAN, width=1)

        # Step number
        draw.text((RIGHT_X + 14, row_y), f"{s['step']:02d}", font=f_row,
                  fill=FG if is_current else DIM)

        # Decision pill — treat "initial" as NEW_PAGE (full frame sent)
        trans = s["transition"]
        is_new = trans in ("new_page", "initial")
        pill_color = (80, 30, 40) if is_new else (20, 50, 46)
        pill_text_color = (220, 120, 130) if is_new else CYAN
        pill_x = RIGHT_X + 58
        pill_label = "NEW_PAGE" if is_new else "DELTA ▲"
        f_pill = font(11)
        pill_bbox = draw.textbbox((pill_x + 8, row_y + 3), pill_label, font=f_pill)
        draw.rectangle([pill_x, row_y - 2, pill_bbox[2] + 6, pill_bbox[3] + 4],
                       fill=pill_color)
        draw.text((pill_x + 8, row_y + 3), pill_label, font=f_pill, fill=pill_text_color)

        # Pixel diff bar
        diff_ratio = s.get("diff_ratio", 0)
        bar_x = pill_x + 120
        bar_w = 340
        bar_h = 8
        bar_y_c = row_y + 10
        draw.rectangle([bar_x, bar_y_c, bar_x + bar_w, bar_y_c + bar_h],
                       fill=(30, 30, 34))
        fill_w = int(bar_w * min(1.0, diff_ratio))
        fill_color = (180, 60, 80) if is_new else CYAN
        if fill_w > 0:
            draw.rectangle([bar_x, bar_y_c, bar_x + fill_w, bar_y_c + bar_h],
                           fill=fill_color)
        draw.text((bar_x + bar_w + 10, row_y + 3),
                  f"{int(diff_ratio*100)}%", font=f_pill, fill=DIM)

        # DV tokens
        dv_t = s["dv_tokens"]
        draw.text((CANVAS_W - 130, row_y + 3), f"{dv_t:,}",
                  font=f_num, fill=FG if is_current else DIM)
        if "ff_tokens" in s and s["ff_tokens"] != dv_t:
            delta = dv_t - s["ff_tokens"]
            draw.text((CANVAS_W - 80, row_y + 3), f"{delta:+,}",
                      font=font(10), fill=DIM)

    # --- BOTTOM: DELTA tag ---
    bot_y = CANVAS_H - 40
    is_delta = step_data["transition"] == "delta"
    tag_label = "DELTA ▲" if is_delta else "NEW_PAGE"
    tag_color = (20, 50, 46) if is_delta else (80, 30, 40)
    tag_text_color = CYAN if is_delta else (220, 120, 130)
    f_tag = font(14)
    tag_bbox = draw.textbbox((30, bot_y + 4), tag_label, font=f_tag)
    draw.rectangle([20, bot_y, tag_bbox[2] + 12, tag_bbox[3] + 10], fill=tag_color)
    draw.text((30, bot_y + 4), tag_label, font=f_tag, fill=tag_text_color)

    info_txt = f"{step_data['dv_tokens']:,} tokens (vs {step_data['ff_tokens']:,} FF)"
    draw.text((tag_bbox[2] + 30, bot_y + 4), info_txt, font=f_tag, fill=DIM)

    # Legend bottom right
    f_legend = font(11)
    draw.text((CANVAS_W - 380, bot_y + 6),
              "■ NEW_PAGE → full frame    ■ DELTA → cropped diff only",
              font=f_legend, fill=DIM)

    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    steps = load_log()
    shots = load_screenshots()
    # Align: log step 1 = shots[0]
    n = min(len(steps), len(shots))

    # Walk through ALL 29 steps in order — full trace like FF section
    curated = list(range(n))

    prev_bgr = None
    frame_paths = []

    for frame_idx, i in enumerate(curated):
        step = steps[i]
        shot = shots[i]
        img = cv2.imread(shot)

        # Need previous frame for bbox computation
        if i > 0:
            prev = cv2.imread(shots[i - 1])
            # Resize if different dimensions
            if prev.shape != img.shape:
                prev = cv2.resize(prev, (img.shape[1], img.shape[0]))
            if step["transition"] == "delta":
                # Use smaller dilation for tighter bbox on true deltas
                bbox, _ = get_bbox(prev, img, pad=20)
                # If bbox is absurdly large (>80% of screen) for a "delta" claimed to be small,
                # fall back to a smaller centered region around the densest change
                if bbox is not None:
                    _, _, bw, bh = bbox
                    area_frac = (bw * bh) / (img.shape[1] * img.shape[0])
                    diff_ratio = step.get("diff_ratio", 1.0)
                    if area_frac > 0.6 and diff_ratio < 0.3:
                        # Dense change is small but spread out — use diff_ratio to cap bbox
                        # Center a reasonable box around the detected region
                        cx = bbox[0] + bw // 2
                        cy = bbox[1] + bh // 2
                        target_area = diff_ratio * 2.5  # overprovision 2.5x
                        target_w = int((img.shape[1] * img.shape[0] * target_area) ** 0.5)
                        target_h = target_w
                        nx = max(0, cx - target_w // 2)
                        ny = max(0, cy - target_h // 2)
                        nw = min(img.shape[1] - nx, target_w)
                        nh = min(img.shape[0] - ny, target_h)
                        bbox = (nx, ny, nw, nh)
            else:
                # NEW_PAGE: full frame
                bbox = (0, 0, img.shape[1], img.shape[0])
        else:
            bbox = (0, 0, img.shape[1], img.shape[0])  # initial

        composite = render_composite(shot, bbox, step, steps, i)
        out_path = os.path.join(OUT_DIR, f"frame_{frame_idx:03d}.png")
        composite.save(out_path)
        frame_paths.append(out_path)
        saved = step.get("savings_pct_cumulative", 0)
        print(f"  [{frame_idx+1}/{len(curated)}] step {step['step']:02d} "
              f"{step['transition']:9} ff_cum={step['ff_cumulative']:6,} "
              f"dv_cum={step['dv_cumulative']:6,} saved={saved:.0f}%")

    # Render video
    list_path = os.path.join(OUT_DIR, "frames.txt")
    hold_frames = int(FPS * HOLD_S)
    with open(list_path, "w") as fh:
        for p in frame_paths:
            for _ in range(hold_frames):
                fh.write(f"file '{p}'\n")
                fh.write(f"duration {1/FPS:.4f}\n")

    os.system(
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
        f'-vf "fps={FPS}" -c:v libx264 -crf 18 -pix_fmt yuv420p "{OUT_VIDEO}" 2>&1 | tail -3'
    )
    print(f"\nVideo: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
