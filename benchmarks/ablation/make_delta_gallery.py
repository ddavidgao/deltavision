#!/usr/bin/env python3
"""
Static delta gallery — visual proof of what DV's classifier identifies as "the
delta" between consecutive frames, side-by-side with the unmodified pair.

For each interesting frame transition in the SF parallel run, we output a
single PNG showing:

    ┌────────────────────┬────────────────────┬────────────────────┐
    │   t0 (before)      │   t1 (after)       │   spotlight on t1  │
    │   plain frame      │   plain frame      │   blur outside     │
    │                    │                    │   bbox + cyan box  │
    └────────────────────┴────────────────────┴────────────────────┘
    step N  diff_ratio=X  trigger=...  → bbox=(x,y,w,h)  area=Y%

This makes it obvious which frames DV could spotlight (small localized change)
vs which it has to send as a full frame (big global change).

Output: /tmp/delta_gallery/*.png  + an index page
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import DeltaVisionConfig
from vision.classifier import classify_transition, extract_anchor
from vision.diff import compute_diff

DV_DIR = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/screenshots"
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/dv_proxy_run_1777072627.jsonl"
OUT_DIR = Path("/tmp/delta_gallery")
OUT_DIR.mkdir(exist_ok=True)

# Phantom DV steps to skip (same as the video script)
DV_EXCLUDE_STEPS = {8, 9, 10}

# Layout
W, H = 1920, 720
PANEL_W = 600
PANEL_GAP = 16
TOP_LABEL_H = 48
BOTTOM_LABEL_H = 80
PANEL_TOP = TOP_LABEL_H + 8

# Colors
BG = (9, 9, 9)
PANEL_BG = (14, 14, 16)
FG = (240, 240, 240)
DIM = (120, 120, 120)
CYAN = (20, 184, 166)
RED = (248, 113, 113)
YELLOW = (250, 204, 21)

# Spotlight params
BLUR_RADIUS = 8
DIM_OUTSIDE = 0.4
BBOX_PAD = 24
BORDER_WIDTH = 4

_FONT_DIR = ROOT / "benchmarks/ablation/.fonts"
INTER = {
    "regular": _FONT_DIR / "Inter-Regular.ttf",
    "medium": _FONT_DIR / "Inter-Medium.ttf",
    "semibold": _FONT_DIR / "Inter-SemiBold.ttf",
    "bold": _FONT_DIR / "Inter-Bold.ttf",
}
JBM = {"regular": _FONT_DIR / "JetBrainsMono-Regular.ttf",
       "medium": _FONT_DIR / "JetBrainsMono-Medium.ttf"}


def font(sz, mono=False, weight="regular"):
    table = JBM if mono else INTER
    if mono and weight in ("semibold", "bold"):
        weight = "medium"
    return ImageFont.truetype(str(table.get(weight, table["regular"])), sz)


def largest_contour_bbox(t0_bgr, t1_bgr, pad=BBOX_PAD):
    """Largest single changed region (matches the fixed video logic)."""
    diff = cv2.absdiff(t0_bgr, t1_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    if mask.sum() / 255 / mask.size < 0.003:
        return None
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    Hh, Ww = t1_bgr.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(Ww - x, w + pad * 2)
    h = min(Hh - y, h + pad * 2)
    return (x, y, w, h)


def apply_spotlight(pil_img, bbox, force_full=False):
    img = pil_img.convert("RGB")
    Wp, Hp = img.size
    if force_full or bbox is None:
        return img
    bx, by, bw, bh = bbox
    if (bw * bh) / (Wp * Hp) >= 0.8:
        return img
    from PIL import ImageFilter
    blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    overlay = Image.new("RGBA", (Wp, Hp), (0, 0, 0, int(255 * (1 - DIM_OUTSIDE))))
    base = Image.alpha_composite(blurred.convert("RGBA"), overlay).convert("RGB")
    sharp = img.crop((bx, by, bx + bw, by + bh))
    base.paste(sharp, (bx, by))
    d = ImageDraw.Draw(base)
    for i in range(BORDER_WIDTH):
        d.rectangle([bx - i, by - i, bx + bw + i, by + bh + i], outline=CYAN)
    return base


def scale_panel(pil_img):
    """Scale to fit PANEL_W width, preserving aspect."""
    w0, h0 = pil_img.size
    scale = PANEL_W / w0
    new_h = int(h0 * scale)
    return pil_img.resize((PANEL_W, new_h), Image.LANCZOS)


def render_card(t0_path, t1_path, log_entry, idx, total):
    """One gallery card showing t0 | t1 | spotlight."""
    t0_pil = Image.open(t0_path).convert("RGB")
    t1_pil = Image.open(t1_path).convert("RGB")
    t0_bgr = cv2.imread(str(t0_path))
    t1_bgr = cv2.imread(str(t1_path))

    # Native bbox + classifier
    bbox = largest_contour_bbox(t0_bgr, t1_bgr)
    cfg = DeltaVisionConfig()
    anchor = extract_anchor(t0_pil, cfg)
    diff = compute_diff(t0_pil, t1_pil, cfg)
    cls = classify_transition(t0_pil, t1_pil, "", "", anchor, cfg, diff_result=diff)

    # If classifier said NEW_PAGE, the spotlight is honest about it: full frame
    force_full = cls.transition.value == "new_page"
    spotlit = apply_spotlight(t1_pil, bbox, force_full=force_full)

    # Scale all three panels
    p0 = scale_panel(t0_pil)
    p1 = scale_panel(t1_pil)
    p2 = scale_panel(spotlit)
    panel_h = p0.size[1]

    canvas_w = PANEL_W * 3 + PANEL_GAP * 2 + 64
    canvas_h = TOP_LABEL_H + 8 + panel_h + BOTTOM_LABEL_H + 32

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    d = ImageDraw.Draw(canvas)

    # Top labels
    f_label = font(14, mono=True, weight="medium")
    f_step = font(20, weight="semibold")

    # Filename row at the very top — small, dim
    f_file = font(12, mono=True)
    d.text((32, 8), f"t0: {t0_path.name}", font=f_file, fill=DIM)
    d.text((32 + PANEL_W + PANEL_GAP, 8), f"t1: {t1_path.name}", font=f_file, fill=DIM)
    d.text((32 + (PANEL_W + PANEL_GAP) * 2, 8),
           f"DV spotlight (step {log_entry['step']})", font=f_file, fill=CYAN)

    # Column headers
    head_y = 28
    d.text((32, head_y), "BEFORE (t0)", font=f_label, fill=DIM)
    d.text((32 + PANEL_W + PANEL_GAP, head_y), "AFTER (t1)", font=f_label, fill=DIM)
    decision = cls.transition.value.upper()
    decision_color = RED if decision == "NEW_PAGE" else CYAN
    d.text((32 + (PANEL_W + PANEL_GAP) * 2, head_y),
           f"DV: {decision}", font=f_label, fill=decision_color)

    # Paste panels
    py = PANEL_TOP
    canvas.paste(p0, (32, py))
    canvas.paste(p1, (32 + PANEL_W + PANEL_GAP, py))
    canvas.paste(p2, (32 + (PANEL_W + PANEL_GAP) * 2, py))

    # Bottom: detail strip
    by = py + panel_h + 16
    f_meta = font(13, mono=True)
    f_meta_b = font(13, mono=True, weight="medium")

    # Build the explanation
    if force_full:
        spotlight_note = "no spotlight: classifier says NEW_PAGE → full frame sent"
        spotlight_color = RED
    elif bbox is None:
        spotlight_note = "no detectable change (idle frame)"
        spotlight_color = DIM
    else:
        bx, _, bw, bh = bbox
        Wn, Hn = t1_pil.size
        area = (bw * bh) / (Wn * Hn) * 100
        if area >= 80:
            spotlight_note = f"bbox {bw}×{bh} = {area:.0f}% of frame → too big, full frame"
            spotlight_color = YELLOW
        else:
            spotlight_note = f"spotlight: bbox {bw}×{bh} = {area:.0f}% of frame"
            spotlight_color = CYAN

    diff_pct = diff.diff_ratio * 100
    line1 = (f"step {log_entry['step']:>2}/{total}  ·  "
             f"transition={log_entry.get('transition', '?')}  ·  "
             f"trigger={log_entry.get('trigger', '?')}  ·  "
             f"diff_ratio={diff_pct:.1f}%  ·  "
             f"phash={log_entry.get('phash_distance', 0)}")

    dv_tok = log_entry.get("dv_tokens", 0)
    ff_tok = log_entry.get("ff_tokens", 1365)
    saved_tok = ff_tok - dv_tok
    line2 = (f"DV billed {dv_tok:,} tokens  ·  "
             f"FF would be {ff_tok:,}  ·  "
             f"{'saved ' + format(saved_tok, ',') if saved_tok > 0 else 'no savings'}")

    d.text((32, by), line1, font=f_meta, fill=FG)
    d.text((32, by + 22), line2, font=f_meta_b, fill=CYAN if saved_tok > 0 else DIM)
    d.text((32, by + 46), spotlight_note, font=f_meta, fill=spotlight_color)

    return canvas


def main():
    # Load log
    log_steps = {}
    for line in open(DV_LOG):
        if "step" not in line:
            continue
        rec = json.loads(line)
        log_steps[rec["step"]] = rec

    # Files in step order, excluding phantoms
    files = sorted(p for p in DV_DIR.iterdir() if p.suffix == ".png")
    files = [f for f in files if int(f.name.split("_")[1]) not in DV_EXCLUDE_STEPS]
    print(f"Found {len(files)} kept screenshots")

    # Choose interesting transitions: pick a varied selection across the run.
    # Skip the very first frame (no t0). Pick representative samples spaced
    # across the trajectory.
    indices = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 23, 26]  # 0-indexed in kept list
    indices = [i for i in indices if i < len(files)]
    print(f"Rendering {len(indices)} cards")

    for n, i in enumerate(indices):
        t0 = files[i - 1]
        t1 = files[i]
        # Find log entry — file name has step number in the prefix
        step_num = int(t1.name.split("_")[1])
        log_entry = log_steps.get(step_num, {"step": step_num})
        card = render_card(t0, t1, log_entry, i, len(files) - 1)
        out = OUT_DIR / f"card_{n:02d}_step_{step_num:02d}.png"
        card.save(out)
        print(f"  [{n+1}/{len(indices)}] {out.name}  ({card.size[0]}×{card.size[1]})")

    print(f"\nGallery: {OUT_DIR}")


if __name__ == "__main__":
    main()
