"""
DeltaVision real comparison video — synchronized step-by-step.

Two independent agents, same task in plain English:
  "Add 5 todos, check 2, click Active filter"

Left side = DeltaVision agent (3 steps total)
Right side = Full-Frame agent (6 steps total)

Both progress step-by-step at the same pace.
After DV finishes at step 2, left shows COMPLETE while FF keeps going.
Every action shown. Every observation shown.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE    = Path(__file__).parent
DV_DIR  = BASE / "runs/subagent_dv"
FF_DIR  = BASE / "runs/subagent_ff"
OUT_DIR = BASE / "video_frames"
OUT_MP4 = OUT_DIR / "real_comparison.mp4"
OUT_DIR.mkdir(exist_ok=True)

# ── Fonts ─────────────────────────────────────────────────────────────
try:
    F_BIG = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    F_LG  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    F_MD  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    F_SM  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    F_XS  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
except Exception:
    F_BIG = F_LG = F_MD = F_SM = F_XS = ImageFont.load_default()

# ── Colors ────────────────────────────────────────────────────────────
DARK   = (14,  14,  18)
DARK2  = (22,  22,  28)
DARK3  = (32,  32,  40)
BLUE   = (60,  170, 240)
GREEN  = (50,  200,  90)
ORANGE = (220, 120,  50)
GOLD   = (240, 200,  60)
TEAL   = (60,  210, 190)
WHITE  = (235, 235, 240)
GRAY   = (110, 110, 120)
RED    = (220,  60,  60)

CW, CH = 1520, 900
HALF   = CW // 2
PAD    = 16

# ── Helpers ───────────────────────────────────────────────────────────
def bx(draw, x0, y0, x1, y1, **kw): draw.rectangle([x0, y0, x1, y1], **kw)
def tx(draw, x, y, s, font, fill):  draw.text((x, y), s, font=font, fill=fill)

def load_img(p):  return Image.open(str(p)).convert("RGB")
def load_obs(p):  return json.loads(Path(p).read_text())

def fit_img(canvas, img, x, y, w, h, border=None, bw=2):
    """Paste img into box (x,y,w,h) preserving aspect ratio, centered."""
    iw, ih = img.size
    s  = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), (ox, oy))
    if border:
        ImageDraw.Draw(canvas).rectangle([ox, oy, ox+nw, oy+nh],
                                         outline=border, width=bw)
    return ox, oy, nw, nh

def draw_green_boxes(img, crops, src_w=1200, src_h=700):
    img = img.copy()
    d   = ImageDraw.Draw(img)
    tw, th = img.size
    for c in crops:
        x, y, w, h = c["bbox"]
        sx, sy = tw/src_w, th/src_h
        d.rectangle([int(x*sx), int(y*sy),
                     int((x+w)*sx), int((y+h)*sy)],
                    outline=(0, 220, 80), width=2)
    return img

# ── Load all data upfront ─────────────────────────────────────────────
dv = [load_obs(DV_DIR / f"step_{i}/obs.json") for i in range(3)]
ff = [load_obs(FF_DIR / f"step_{i}/obs.json") for i in range(6)]

dv_imgs = {
    "s0": load_img(DV_DIR / "step_0/t0.png"),
    "s1_thumb": draw_green_boxes(
        load_img(DV_DIR / "step_1/thumbnail.png"),
        dv[1].get("crops", [])
    ),
    "s1_crop0": load_img(DV_DIR / "step_1/crop_0_after.png"),
    "s1_crop1": load_img(DV_DIR / "step_1/crop_1_after.png"),
    "s2": load_img(DV_DIR / "step_2/t1.png"),
}
ff_imgs = {f"s{i}": load_img(FF_DIR / f"step_{i}/{'t0' if i==0 else 't1'}.png")
           for i in range(6)}

# Token totals
DV_FINAL_TOK = 3620
FF_FINAL_TOK = 9600
MAX_TOK = 10000

# ─────────────────────────────────────────────────────────────────────
# FRAME BUILDER
# ─────────────────────────────────────────────────────────────────────

HEAD_H    = 68     # top header bar
LABEL_H   = 52     # per-panel action label
IMG_TOP   = HEAD_H + LABEL_H
IMG_BOT   = CH - 90
IMG_H     = IMG_BOT - IMG_TOP
PW        = HALF - PAD * 2
LP_X      = PAD
RP_X      = HALF + PAD
BAR_Y     = IMG_BOT + 10
BAR_H     = 22


def make_frame(
    step_n: int,
    # Left (DV) panel
    dv_action: str,
    dv_obs_label: str,
    dv_imgs_list: list,          # [(PIL, caption, border_color), ...]
    dv_tok_step: int,
    dv_tok_cumul: int,
    dv_done: bool,
    # Right (FF) panel
    ff_action: str,
    ff_obs_label: str,
    ff_imgs_list: list,
    ff_tok_step: int,
    ff_tok_cumul: int,
    ff_done: bool = False,
) -> Image.Image:

    img  = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    # ── Top header ───────────────────────────────────────────────────
    bx(draw, 0, 0, CW, HEAD_H, fill=DARK2)
    tx(draw, PAD, 10, f"Step {step_n}", F_BIG, WHITE)
    tx(draw, PAD, 46, "DeltaVision vs Full-Frame  —  same task, independent agents", F_XS, GRAY)
    # Divider
    bx(draw, HALF, 0, HALF + 1, CH, fill=DARK3)

    # ── DV panel label ───────────────────────────────────────────────
    dv_col = BLUE if not dv_done else GREEN
    bx(draw, LP_X, HEAD_H, LP_X + PW, HEAD_H + LABEL_H, fill=DARK3)
    tx(draw, LP_X + 4, HEAD_H + 4,
       "DeltaVision" + (" — COMPLETE" if dv_done else ""), F_MD, dv_col)
    if not dv_done:
        tx(draw, LP_X + 4, HEAD_H + 24, f"Action: {dv_action}", F_SM, WHITE)
        tx(draw, LP_X + 4, HEAD_H + 40, dv_obs_label, F_XS, dv_col)
    else:
        tx(draw, LP_X + 4, HEAD_H + 24, "Finished at step 2  |  3,620 tokens total", F_SM, GREEN)

    # ── FF panel label ───────────────────────────────────────────────
    ff_col = ORANGE if not ff_done else GREEN
    bx(draw, RP_X, HEAD_H, RP_X + PW, HEAD_H + LABEL_H, fill=DARK3)
    tx(draw, RP_X + 4, HEAD_H + 4,
       "Full-Frame" + (" — COMPLETE" if ff_done else ""), F_MD, ff_col)
    if not ff_done:
        tx(draw, RP_X + 4, HEAD_H + 24, f"Action: {ff_action}", F_SM, WHITE)
        tx(draw, RP_X + 4, HEAD_H + 40, ff_obs_label, F_XS, ff_col)
    else:
        tx(draw, RP_X + 4, HEAD_H + 24, "Finished at step 5  |  9,600 tokens total", F_SM, GREEN)

    # ── DV image area ────────────────────────────────────────────────
    if dv_done:
        # Big COMPLETE card
        cx, cy = LP_X + PW // 2, IMG_TOP + IMG_H // 2
        bx(draw, LP_X + 40, IMG_TOP + 60, LP_X + PW - 40, IMG_TOP + IMG_H - 60, fill=DARK2)
        bx(draw, LP_X + 40, IMG_TOP + 60, LP_X + PW - 40, IMG_TOP + IMG_H - 60,
           outline=GREEN, width=2)
        tx(draw, LP_X + 60, IMG_TOP + 90,  "Task complete", F_LG, GREEN)
        tx(draw, LP_X + 60, IMG_TOP + 130, "3 steps", F_BIG, WHITE)
        tx(draw, LP_X + 60, IMG_TOP + 178, "3,620 tokens", F_BIG, BLUE)
        tx(draw, LP_X + 60, IMG_TOP + 240, f"FF has used {ff_tok_cumul:,} tokens so far", F_SM, GRAY)
        tx(draw, LP_X + 60, IMG_TOP + 260, f"DV saved {ff_tok_cumul - dv_tok_cumul:,} tokens already", F_SM, TEAL)
        pct = int((1 - dv_tok_cumul / ff_tok_cumul) * 100) if ff_tok_cumul else 0
        tx(draw, LP_X + 60, IMG_TOP + 295, f"{pct}% fewer tokens", F_LG, GOLD)
    else:
        n = len(dv_imgs_list)
        each_h = IMG_H // max(n, 1)
        iy = IMG_TOP
        for (pil, cap, col) in dv_imgs_list:
            if pil:
                fit_img(img, pil, LP_X, iy, PW, each_h - 18, border=col)
            if cap:
                tx(draw, LP_X + 2, iy + each_h - 16, cap, F_XS, col or GRAY)
            iy += each_h

    # ── FF image area ────────────────────────────────────────────────
    n = len(ff_imgs_list)
    each_h = IMG_H // max(n, 1)
    iy = IMG_TOP
    for (pil, cap, col) in ff_imgs_list:
        if pil:
            fit_img(img, pil, RP_X, iy, PW, each_h - 18, border=col)
        if cap:
            tx(draw, RP_X + 2, iy + each_h - 16, cap, F_XS, col or GRAY)
        iy += each_h

    # ── Token bars ───────────────────────────────────────────────────
    hw = PW
    dv_frac = min(dv_tok_cumul / MAX_TOK, 1.0)
    ff_frac = min(ff_tok_cumul / MAX_TOK, 1.0)

    tx(draw, LP_X,     BAR_Y - 14, f"DV: {dv_tok_cumul:,} tokens cumulative"
                                    + (f"  (+{dv_tok_step:,} this step)" if not dv_done else "  [DONE]"),
       F_SM, BLUE if not dv_done else GREEN)
    bx(draw, LP_X, BAR_Y, LP_X + hw, BAR_Y + BAR_H, fill=DARK3)
    if dv_frac > 0:
        bx(draw, LP_X, BAR_Y, LP_X + int(hw * dv_frac), BAR_Y + BAR_H,
           fill=BLUE if not dv_done else GREEN)

    tx(draw, RP_X,     BAR_Y - 14, f"FF: {ff_tok_cumul:,} tokens cumulative"
                                    + (f"  (+{ff_tok_step:,} this step)" if not ff_done else "  [DONE]"),
       F_SM, ORANGE if not ff_done else GREEN)
    bx(draw, RP_X, BAR_Y, RP_X + hw, BAR_Y + BAR_H, fill=DARK3)
    if ff_frac > 0:
        bx(draw, RP_X, BAR_Y, RP_X + int(hw * ff_frac), BAR_Y + BAR_H,
           fill=ORANGE if not ff_done else GREEN)

    # Savings callout when DV is ahead
    if ff_tok_cumul > dv_tok_cumul:
        pct = int((1 - dv_tok_cumul / ff_tok_cumul) * 100)
        tx(draw, CW - 200, BAR_Y - 16, f"DV: {pct}% less", F_LG, GOLD)

    return img


# ── Intro card ────────────────────────────────────────────────────────
def intro_card():
    img  = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)
    bx(draw, 0, 0, CW, CH, fill=DARK)

    tx(draw, PAD, 60,  "DeltaVision vs Full-Frame", F_BIG, WHITE)
    tx(draw, PAD, 108, "Two independent agents, same task in plain English, own browser session each.", F_MD, GRAY)

    bx(draw, PAD, 148, CW - PAD, 149, fill=DARK3)

    tx(draw, PAD, 166, "Task:", F_SM, GRAY)
    tx(draw, 120, 166, "Add write report, review PR, send invoice, update docs, schedule meeting.", F_SM, WHITE)
    tx(draw, 120, 184, "Check write report and review PR as done. Click the Active filter.", F_SM, WHITE)
    tx(draw, PAD, 210, "Site:", F_SM, GRAY)
    tx(draw, 120, 210, "https://todomvc.com/examples/react/dist/", F_SM, GRAY)

    bx(draw, PAD, 240, CW - PAD, 241, fill=DARK3)

    # Two column result summary
    tx(draw, PAD,      270, "DeltaVision mode", F_LG, BLUE)
    tx(draw, PAD,      302, "Sees thumbnail + crops on SPA delta steps", F_SM, GRAY)
    tx(draw, PAD,      322, "Full frame only on URL change (new page)", F_SM, GRAY)

    tx(draw, HALF+PAD, 270, "Full-Frame baseline", F_LG, ORANGE)
    tx(draw, HALF+PAD, 302, "Full 1200x700 screenshot every single step", F_SM, GRAY)
    tx(draw, HALF+PAD, 322, "No filtering, no delta gating", F_SM, GRAY)

    bx(draw, PAD, 360, CW - PAD, 361, fill=DARK3)

    # The punch
    tx(draw, PAD,       390, "Steps to complete:", F_MD, GRAY)
    tx(draw, PAD,       420, "3", F_BIG, BLUE)
    tx(draw, HALF+PAD,  390, "Steps to complete:", F_MD, GRAY)
    tx(draw, HALF+PAD,  420, "6", F_BIG, ORANGE)

    tx(draw, PAD,       490, "Total tokens:", F_MD, GRAY)
    tx(draw, PAD,       520, "3,620", F_BIG, BLUE)
    tx(draw, HALF+PAD,  490, "Total tokens:", F_MD, GRAY)
    tx(draw, HALF+PAD,  520, "9,600", F_BIG, ORANGE)

    tx(draw, CW//2 - 200, 610, "62% fewer tokens", F_BIG, GOLD)
    tx(draw, CW//2 - 140, 660, "genuine independent runs — paths differed", F_SM, GRAY)

    tx(draw, PAD, CH - 40, "Watch both agents step by step ->", F_SM, GRAY)
    return img


# ── Summary card ──────────────────────────────────────────────────────
def summary_card():
    img  = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    tx(draw, PAD, 30, "Final Comparison", F_BIG, WHITE)
    tx(draw, PAD, 76, "Same task  |  Independent agents  |  Genuine runs", F_SM, GRAY)

    # Table
    COLS = [PAD, 90, 260, 450, 720, 1000, 1200, 1420]
    def row(y, vals, colors, bg):
        bx(draw, PAD, y, CW-PAD, y+28, fill=bg)
        for x, v, c in zip(COLS, vals, colors, strict=False):
            tx(draw, x+4, y+7, str(v), F_SM, c)

    y = 114
    bx(draw, PAD, y, CW-PAD, y+28, fill=DARK3)
    headers = ["#", "Agent", "Obs type", "Action", "What model saw", "DV tok", "FF tok", "Cumul"]
    for x, h in zip(COLS, headers, strict=False):
        tx(draw, x+4, y+8, h, F_SM, GRAY)
    y += 30

    dv_rows = [
        (0, "DV", "full_frame", "navigate",
         "full 1200x700 screenshot", 1600, 1600, "1,600"),
        (1, "DV", "DELTA",     "type all 5 todos + Enter x5",
         "320x225 thumb + 2 crops",   420, 1600, "2,020"),
        (2, "DV", "full_frame", "check x2 + click Active",
         "full frame (URL changed)",  1600, 1600, "3,620"),
    ]
    ff_rows = [
        (0, "FF", "full_frame", "navigate",
         "full 1200x700",  1600, 1600, "1,600"),
        (1, "FF", "full_frame", "type('write report') + Enter",
         "full 1200x700",  1600, 1600, "3,200"),
        (2, "FF", "full_frame", "type remaining 4 + Enter x4",
         "full 1200x700",  1600, 1600, "4,800"),
        (3, "FF", "full_frame", "check 'write report'",
         "full 1200x700",  1600, 1600, "6,400"),
        (4, "FF", "full_frame", "check 'review PR'",
         "full 1200x700",  1600, 1600, "8,000"),
        (5, "FF", "full_frame", "click Active filter",
         "full 1200x700",  1600, 1600, "9,600"),
    ]

    for r in dv_rows:
        n, ag, obs, act, what, dv_t, ff_t, cumul = r
        c_obs   = TEAL if obs == "DELTA" else WHITE
        c_what  = TEAL if obs == "DELTA" else GRAY
        colors  = [WHITE, BLUE, c_obs, WHITE, c_what, BLUE, GRAY, BLUE]
        row(y, [n, ag, obs, act, what, dv_t, ff_t, cumul], colors,
            DARK2 if n % 2 == 0 else DARK)
        y += 28

    bx(draw, PAD, y, CW-PAD, y+1, fill=DARK3)
    y += 6

    for r in ff_rows:
        n, ag, obs, act, what, dv_t, ff_t, cumul = r
        colors = [WHITE, ORANGE, WHITE, WHITE, GRAY, ORANGE, ORANGE, ORANGE]
        row(y, [n, ag, obs, act, what, dv_t, ff_t, cumul], colors,
            DARK2 if n % 2 == 0 else DARK)
        y += 28

    y += 10
    bx(draw, PAD, y, CW-PAD, y+34, fill=DARK3)
    tx(draw, COLS[0]+4, y+9, "TOTAL", F_MD, WHITE)
    tx(draw, COLS[1]+4, y+9, "DV: 3 steps", F_MD, BLUE)
    tx(draw, COLS[2]+60, y+9, "FF: 6 steps", F_MD, ORANGE)
    tx(draw, COLS[5]+4, y+9, "3,620", F_MD, BLUE)
    tx(draw, COLS[6]+4, y+9, "9,600", F_MD, ORANGE)
    y += 50

    tx(draw, PAD, y,    "DeltaVision:  3,620 tokens  (3 steps)", F_LG, BLUE)
    tx(draw, PAD, y+32, "Full-Frame:   9,600 tokens  (6 steps)", F_LG, ORANGE)
    tx(draw, PAD, y+72, "62% fewer tokens", F_BIG, GOLD)
    tx(draw, PAD, y+116,"Both agents completed the task correctly. Paths genuinely differed.", F_SM, GRAY)
    tx(draw, PAD, y+134,"DV got focused crops -> batched more decisively. FF re-verified each sub-action.", F_SM, GRAY)

    return img


# ─────────────────────────────────────────────────────────────────────
# DEFINE ALL FRAMES
# ─────────────────────────────────────────────────────────────────────
STEP_HOLD = 5.0   # seconds per step — same for all
INTRO_HOLD = 4.0
SUMM_HOLD  = 7.0

FRAMES = []  # (PIL, hold_s)
FRAMES.append((intro_card(), INTRO_HOLD))

# ── Step 0 ────────────────────────────────────────────────────────────
FRAMES.append((make_frame(
    step_n=0,
    dv_action="navigate to TodoMVC",
    dv_obs_label="full_frame (initial load) | 1,600 tokens",
    dv_imgs_list=[(dv_imgs["s0"], "Full page — both modes identical at step 0", BLUE)],
    dv_tok_step=1600, dv_tok_cumul=1600, dv_done=False,
    ff_action="navigate to TodoMVC",
    ff_obs_label="full_frame (initial load) | 1,600 tokens",
    ff_imgs_list=[(ff_imgs["s0"], "Full page — both modes identical at step 0", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=1600,
), STEP_HOLD))

# ── Step 1 ────────────────────────────────────────────────────────────
# DV: typed all 5, saw delta (thumb + crops)
# FF: typed only 'write report', saw full frame
FRAMES.append((make_frame(
    step_n=1,
    dv_action="type all 5 todos + Enter x5  (batched)",
    dv_obs_label="DELTA | diff 6.2% | thumbnail (60 tok) + 2 crops (360 tok) = 420 tokens",
    dv_imgs_list=[
        (dv_imgs["s1_thumb"],
         "(1) 320x225 thumbnail — green boxes show WHERE changed", TEAL),
        (dv_imgs["s1_crop0"],
         "(2) crop_0 — full todo list region zoomed in, all 5 items visible", GREEN),
    ],
    dv_tok_step=420, dv_tok_cumul=2020, dv_done=False,
    ff_action="type('write report') + Enter",
    ff_obs_label="full_frame | 1,600 tokens — full 1200x700 for a single new row",
    ff_imgs_list=[(ff_imgs["s1"],
                   "Entire page re-sent: sidebar, header, footer — 1 todo row added", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=3200,
), STEP_HOLD))

# ── Step 2 ────────────────────────────────────────────────────────────
# DV: check 2 + click Active -> URL change -> full_frame -> DONE
# FF: typed remaining 4 todos
FRAMES.append((make_frame(
    step_n=2,
    dv_action="check 'write report', check 'review PR', click Active",
    dv_obs_label="full_frame (URL changed: #/active = NEW_PAGE) | 1,600 tokens — DV done",
    dv_imgs_list=[(dv_imgs["s2"],
                   "Final state: Active filter, 3 items left. Task complete.", BLUE)],
    dv_tok_step=1600, dv_tok_cumul=3620, dv_done=False,
    ff_action="type remaining 4 todos + Enter x4",
    ff_obs_label="full_frame | 1,600 tokens — full page for 4 more rows",
    ff_imgs_list=[(ff_imgs["s2"],
                   "All 5 todos visible — FF took 2 steps just to add them", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=4800,
), STEP_HOLD))

# ── Step 3  (DV DONE, FF keeps going) ────────────────────────────────
FRAMES.append((make_frame(
    step_n=3,
    dv_action="", dv_obs_label="",
    dv_imgs_list=[], dv_tok_step=0, dv_tok_cumul=3620, dv_done=True,
    ff_action="click checkbox: 'write report'",
    ff_obs_label="full_frame | 1,600 tokens — full page to confirm one checkbox",
    ff_imgs_list=[(ff_imgs["s3"],
                   "'write report' checked (green + strikethrough). 4 items left.", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=6400,
), STEP_HOLD))

# ── Step 4 ────────────────────────────────────────────────────────────
FRAMES.append((make_frame(
    step_n=4,
    dv_action="", dv_obs_label="",
    dv_imgs_list=[], dv_tok_step=0, dv_tok_cumul=3620, dv_done=True,
    ff_action="click checkbox: 'review PR'",
    ff_obs_label="full_frame | 1,600 tokens — full page to confirm second checkbox",
    ff_imgs_list=[(ff_imgs["s4"],
                   "Both 'write report' and 'review PR' checked. 3 items left.", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=8000,
), STEP_HOLD))

# ── Step 5 ────────────────────────────────────────────────────────────
FRAMES.append((make_frame(
    step_n=5,
    dv_action="", dv_obs_label="",
    dv_imgs_list=[], dv_tok_step=0, dv_tok_cumul=3620, dv_done=True,
    ff_action="click 'Active' filter link",
    ff_obs_label="full_frame | 1,600 tokens — FF finally done",
    ff_imgs_list=[(ff_imgs["s5"],
                   "Active filter selected. 3 items left. Task complete.", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=9600, ff_done=True,
), STEP_HOLD))

FRAMES.append((summary_card(), SUMM_HOLD))

# ─────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────
FPS    = 30
FADE_F = int(FPS * 0.35)

all_np   = []
rendered = [f for (f, _) in FRAMES]

for i, (frame, hold_s) in enumerate(FRAMES):
    if i > 0:
        prev = rendered[i - 1]
        for f in range(FADE_F):
            a = f / FADE_F
            all_np.append(np.array(Image.blend(prev, frame, a)))
    hold_n = int(FPS * hold_s)
    for _ in range(hold_n):
        all_np.append(np.array(frame))

# Fade to black
fade_out = int(FPS * 0.6)
blank    = Image.new("RGB", (CW, CH), DARK)
for f in range(fade_out):
    a = 1.0 - f / fade_out
    all_np.append(np.array(Image.blend(blank, rendered[-1], a)))

dur = len(all_np) / FPS
print(f"Frames: {len(all_np)}  |  Duration: {dur:.1f}s  |  Steps: {len(FRAMES)-2}")

from moviepy import ImageSequenceClip

clip = ImageSequenceClip(all_np, fps=FPS)
clip.write_videofile(
    str(OUT_MP4), fps=FPS, codec="libx264", audio=False, logger=None,
    ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"]
)
sz = OUT_MP4.stat().st_size / 1024
print(f"\n✅  {OUT_MP4.name}  ({sz:.0f} KB, {dur:.1f}s)")
print("   Left: DV — 3 steps, done at step 2, COMPLETE card for steps 3-5")
print("   Right: FF — 6 steps, every action shown")
print("   62% fewer tokens. Both completed correctly.")
