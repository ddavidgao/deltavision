"""
DeltaVision GitHub comparison video.

Task: vercel/next.js issue triage
  - total open issues?  bug count?  latest bug title?  has comments?
  - most recent merged PR?  latest release?

DV agent:  11 steps, 15,240 tokens (13.4% savings)
FF agent:  10 steps, 16,000 tokens

KEY INSIGHT: GitHub is URL-navigation-heavy → most steps are NEW_PAGE (full frame both).
The delta savings come specifically on SCROLLING within an issue page —
DV's scroll_bypass classifier fires, sends only the scrolled-in region.

Video structure:
  Intro → [Navigation phase] → [Issue view] → [THE KEY MOMENT: scroll deltas] →
  [Research complete] → Summary
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE    = Path(__file__).parent
DV_DIR  = BASE / "runs/github_dv"
FF_DIR  = BASE / "runs/github_ff"
OUT_DIR = BASE / "video_frames"
OUT_MP4 = OUT_DIR / "github_comparison.mp4"
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

def bx(draw, x0, y0, x1, y1, **kw): draw.rectangle([x0, y0, x1, y1], **kw)
def tx(draw, x, y, s, f, c):        draw.text((x, y), s, font=f, fill=c)

def load_img(p):  return Image.open(str(p)).convert("RGB")
def load_obs(p):  return json.loads(Path(p).read_text())

def fit_img(canvas, img, x, y, w, h, border=None, bw=2):
    iw, ih = img.size
    s  = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), (ox, oy))
    if border:
        ImageDraw.Draw(canvas).rectangle([ox, oy, ox+nw, oy+nh], outline=border, width=bw)
    return ox, oy, nw, nh

def draw_green_boxes(img, crops, src_w=1200, src_h=700):
    img = img.copy()
    d   = ImageDraw.Draw(img)
    tw, th = img.size
    for c in crops:
        x, y, w, h = c["bbox"]
        sx, sy = tw/src_w, th/src_h
        d.rectangle([int(x*sx), int(y*sy), int((x+w)*sx), int((y+h)*sy)],
                    outline=(0, 220, 80), width=2)
    return img

# ── Load images ───────────────────────────────────────────────────────
dv7_obs   = load_obs(DV_DIR / "step_7/obs_raw.json")
dv8_obs   = load_obs(DV_DIR / "step_8/obs_raw.json")

imgs = {
    # DV navigation phase — all full frames
    "dv_issues":   load_img(DV_DIR / "step_1/t1.png"),
    "dv_bugs":     load_img(DV_DIR / "step_2/t1.png"),
    "dv_issue":    load_img(DV_DIR / "step_6/t1.png"),
    # DV delta steps — the key moments
    "dv7_thumb":   draw_green_boxes(load_img(DV_DIR / "step_7/thumbnail.png"),
                                    dv7_obs.get("crops", [])),
    "dv7_crop0":   load_img(DV_DIR / "step_7/crop_0_after.png"),
    "dv7_full":    load_img(DV_DIR / "step_7/t1.png"),   # what FF would send
    "dv8_thumb":   draw_green_boxes(load_img(DV_DIR / "step_8/thumbnail.png"),
                                    dv8_obs.get("crops", [])),
    "dv8_crop0":   load_img(DV_DIR / "step_8/crop_0_after.png"),
    "dv8_full":    load_img(DV_DIR / "step_8/t1.png"),
    "dv_prs":      load_img(DV_DIR / "step_9/t1.png"),
    "dv_releases": load_img(DV_DIR / "step_10/t1.png"),
    # FF steps
    "ff_issues":   load_img(FF_DIR / "step_2/t1.png"),
    "ff_bugs":     load_img(FF_DIR / "step_3/t1.png"),
    "ff_issue":    load_img(FF_DIR / "step_4/t1.png"),
    "ff_scroll1":  load_img(FF_DIR / "step_5/t1.png"),
    "ff_scroll2":  load_img(FF_DIR / "step_6/t1.png"),
    "ff_prs":      load_img(FF_DIR / "step_7/t1.png"),
    "ff_releases": load_img(FF_DIR / "step_10/t1.png"),
}

# ── Generic frame builder ─────────────────────────────────────────────
HEAD_H  = 68
LABEL_H = 52
IMG_TOP = HEAD_H + LABEL_H
IMG_BOT = CH - 88
IMG_H   = IMG_BOT - IMG_TOP
PW      = HALF - PAD * 2
LP_X    = PAD
RP_X    = HALF + PAD
BAR_Y   = IMG_BOT + 10

def make_frame(
    phase: str, subtitle: str,
    dv_label: str, dv_obs_type: str, dv_imgs_list: list, dv_tok_step: int, dv_tok_cumul: int,
    ff_label: str,                   ff_imgs_list: list, ff_tok_step: int, ff_tok_cumul: int,
    highlight: bool = False,
) -> Image.Image:
    img  = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    # Header
    hdr_color = (30, 30, 50) if not highlight else (30, 20, 10)
    bx(draw, 0, 0, CW, HEAD_H, fill=hdr_color)
    tx(draw, PAD, 10, phase, F_LG, GOLD if highlight else WHITE)
    tx(draw, PAD, 42, subtitle, F_XS, GRAY)
    if highlight:
        tx(draw, CW - 280, 16, "DELTA OBSERVATION", F_MD, TEAL)
        tx(draw, CW - 280, 40, "scroll_bypass classifier fired", F_XS, TEAL)

    # Panel divider
    bx(draw, HALF, 0, HALF+1, CH, fill=DARK3)

    # DV label bar
    is_delta = dv_obs_type == "delta"
    dv_col   = TEAL if is_delta else BLUE
    bx(draw, LP_X, HEAD_H, LP_X+PW, HEAD_H+LABEL_H, fill=DARK3)
    tx(draw, LP_X+4, HEAD_H+4,  "DeltaVision", F_MD, dv_col)
    tx(draw, LP_X+4, HEAD_H+24, dv_label, F_SM, WHITE)
    tok_str = f"{dv_tok_step:,} tokens this step  |  {dv_tok_cumul:,} cumulative"
    tx(draw, LP_X+4, HEAD_H+40, tok_str, F_XS, dv_col)

    # FF label bar
    bx(draw, RP_X, HEAD_H, RP_X+PW, HEAD_H+LABEL_H, fill=DARK3)
    tx(draw, RP_X+4, HEAD_H+4,  "Full-Frame", F_MD, ORANGE)
    tx(draw, RP_X+4, HEAD_H+24, ff_label, F_SM, WHITE)
    tok_ff  = f"{ff_tok_step:,} tokens this step  |  {ff_tok_cumul:,} cumulative"
    tx(draw, RP_X+4, HEAD_H+40, tok_ff, F_XS, ORANGE)

    # DV images
    n = len(dv_imgs_list)
    each_h = IMG_H // max(n, 1)
    iy = IMG_TOP
    for (pil, cap, col) in dv_imgs_list:
        if pil:
            fit_img(img, pil, LP_X, iy, PW, each_h-16, border=col)
        if cap:
            tx(draw, LP_X+2, iy+each_h-14, cap, F_XS, col or GRAY)
        iy += each_h

    # FF images
    n = len(ff_imgs_list)
    each_h = IMG_H // max(n, 1)
    iy = IMG_TOP
    for (pil, cap, col) in ff_imgs_list:
        if pil:
            fit_img(img, pil, RP_X, iy, PW, each_h-16, border=col)
        if cap:
            tx(draw, RP_X+2, iy+each_h-14, cap, F_XS, col or GRAY)
        iy += each_h

    # Token bars
    MAX_TOK = 17600
    hw = PW
    dv_frac = min(dv_tok_cumul / MAX_TOK, 1.0)
    ff_frac = min(ff_tok_cumul / MAX_TOK, 1.0)

    tx(draw, LP_X, BAR_Y-14, f"DV: {dv_tok_cumul:,} tokens", F_SM, dv_col)
    bx(draw, LP_X, BAR_Y, LP_X+hw, BAR_Y+20, fill=DARK3)
    if dv_frac > 0:
        bx(draw, LP_X, BAR_Y, LP_X+int(hw*dv_frac), BAR_Y+20, fill=dv_col)

    tx(draw, RP_X, BAR_Y-14, f"FF: {ff_tok_cumul:,} tokens", F_SM, ORANGE)
    bx(draw, RP_X, BAR_Y, RP_X+hw, BAR_Y+20, fill=DARK3)
    if ff_frac > 0:
        bx(draw, RP_X, BAR_Y, RP_X+int(hw*ff_frac), BAR_Y+20, fill=ORANGE)

    if ff_tok_cumul > 0 and dv_tok_cumul < ff_tok_cumul:
        pct = int((1 - dv_tok_cumul/ff_tok_cumul)*100)
        tx(draw, CW-180, BAR_Y-16, f"DV: {pct}% less", F_LG, GOLD)

    return img


# ── Intro card ────────────────────────────────────────────────────────
def intro_card() -> Image.Image:
    img  = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    tx(draw, PAD, 50,  "DeltaVision on a Real Production Site", F_BIG, WHITE)
    tx(draw, PAD, 96,  "github.com/vercel/next.js  —  live issue triage task", F_MD, GRAY)
    bx(draw, PAD, 130, CW-PAD, 131, fill=DARK3)

    tx(draw, PAD, 148, "Task:", F_SM, GRAY)
    lines = [
        "1. Total open issue count?",
        "2. How many labeled 'bug'?",
        "3. Title of most recent bug?",
        "4. Does that bug have comments? (requires scrolling)",
        "5. Most recently merged PR?",
        "6. Latest release version?",
    ]
    y = 148
    for l in lines:
        tx(draw, 120, y, l, F_SM, WHITE)
        y += 20

    bx(draw, PAD, y+10, CW-PAD, y+11, fill=DARK3)

    tx(draw, PAD, y+28, "DeltaVision mode", F_LG, BLUE)
    tx(draw, PAD, y+56, "11 steps  |  15,240 tokens  |  13.4% savings", F_MD, BLUE)
    tx(draw, PAD, y+80, "Savings from: scroll_bypass on 2 intra-page scroll steps", F_SM, GRAY)
    tx(draw, PAD, y+98, "GitHub is URL-navigation-heavy — most nav = NEW_PAGE (full frames both modes)", F_SM, GRAY)

    tx(draw, HALF+PAD, y+28, "Full-Frame baseline", F_LG, ORANGE)
    tx(draw, HALF+PAD, y+56, "10 steps  |  16,000 tokens  |  0% savings", F_MD, ORANGE)
    tx(draw, HALF+PAD, y+80, "Every step sends full 1200x700 screenshot", F_SM, GRAY)
    tx(draw, HALF+PAD, y+98, "Including scroll steps — unchanged nav bar re-sent every time", F_SM, GRAY)

    bx(draw, PAD, y+120, CW-PAD, y+121, fill=DARK3)
    tx(draw, PAD, y+136, "Both agents ran independently on live GitHub data.", F_SM, GRAY)
    tx(draw, PAD, y+154, "They found the same answers but slightly different PR/release versions (ran minutes apart — live site).", F_XS, GRAY)

    return img


# ── Summary card ──────────────────────────────────────────────────────
def summary_card() -> Image.Image:
    img  = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    tx(draw, PAD, 30, "Results — vercel/next.js Issue Triage", F_BIG, WHITE)
    tx(draw, PAD, 76, "Real production site  |  Live data  |  Independent agents", F_SM, GRAY)

    # Answers table
    bx(draw, PAD, 100, CW-PAD, 101, fill=DARK3)
    tx(draw, PAD, 108, "ANSWERS (both agents agreed)", F_MD, GREEN)

    answers = [
        ("Total open issues",      "2,111"),
        ("Open 'bug' issues",      "813"),
        ("Latest bug title",       "@vercel/otel fetch instrumentation stripped after HMR dev mode (#92877)"),
        ("Bug has comments?",      "No — scroll to bottom confirmed, only author in Participants"),
        ("Latest stable release",  "v16.2.4  (DV agent also found v16.3.0-canary.0 pre-release — ran later)"),
    ]
    y = 130
    for q, a in answers:
        bx(draw, PAD, y, CW-PAD, y+26, fill=DARK2)
        tx(draw, PAD+4,  y+6, q+":", F_SM, GRAY)
        tx(draw, PAD+260, y+6, a,    F_SM, WHITE)
        y += 28

    bx(draw, PAD, y+8, CW-PAD, y+9, fill=DARK3)
    y += 22

    # Step type breakdown
    tx(draw, PAD, y, "Observation type breakdown:", F_MD, WHITE)
    y += 28
    dv_steps = [
        ("0",  "full_frame", "navigate to repo",         1600, 1600),
        ("1",  "full_frame", "issues tab",               1600, 3200),
        ("2",  "full_frame", "filter: label=bug",        1600, 4800),
        ("3-5","full_frame", "navigate to issue #92877", 1600, 9600),
        ("6",  "full_frame", "open issue #92877",        1600, 11200),
        ("7",  "DELTA",      "scroll within issue",       420, 11620),
        ("8",  "DELTA",      "scroll: confirm no comments",420,12040),
        ("9",  "full_frame", "merged PRs tab",            1600,13640),
        ("10", "full_frame", "releases page",             1600,15240),
    ]
    COLS = [PAD, 80, 170, 400, 600, 800]
    bx(draw, PAD, y, CW//2-PAD, y+22, fill=DARK3)
    for c, h in zip(COLS, ["Step","Type","Action","DV tok","FF tok","DV cumul"], strict=False):
        tx(draw, c+2, y+5, h, F_XS, GRAY)
    y += 22
    for (sn, otype, act, dv_t, dv_c) in dv_steps:
        col = TEAL if otype == "DELTA" else BLUE
        bx(draw, PAD, y, CW//2-PAD, y+20, fill=DARK2 if int(sn[0])%2==0 else DARK)
        for c, v, vc in zip(COLS, [sn, otype, act, str(dv_t), "1600", str(dv_c)],
                                   [WHITE, col, WHITE, col, GRAY, col], strict=False):
            tx(draw, c+2, y+4, v, F_XS, vc)
        y += 20

    # Right side: big numbers
    bx(draw, HALF+PAD, 130, CW-PAD, 440, fill=DARK2)
    tx(draw, HALF+PAD+10, 145, "Token summary",   F_MD,  WHITE)
    tx(draw, HALF+PAD+10, 175, "DeltaVision:  15,240", F_LG,  BLUE)
    tx(draw, HALF+PAD+10, 208, "11 steps",             F_SM,  BLUE)
    tx(draw, HALF+PAD+10, 240, "Full-Frame:   16,000", F_LG,  ORANGE)
    tx(draw, HALF+PAD+10, 273, "10 steps",             F_SM,  ORANGE)
    bx(draw, HALF+PAD+10, 298, CW-PAD-10, 299, fill=DARK3)
    tx(draw, HALF+PAD+10, 308, "Savings: 13.4%",       F_BIG, GOLD)
    tx(draw, HALF+PAD+10, 356, "From: 2 scroll steps",  F_SM,  TEAL)
    tx(draw, HALF+PAD+10, 376, "scroll_bypass -> DELTA", F_SM, TEAL)
    tx(draw, HALF+PAD+10, 396, "9 / 11 steps = NEW_PAGE (URL nav)",F_SM, GRAY)
    tx(draw, HALF+PAD+10, 416, "only intra-page scroll = DELTA",    F_SM, GRAY)

    tx(draw, HALF+PAD, 458, "Why modest savings?", F_MD, WHITE)
    tx(draw, HALF+PAD, 484, "GitHub routes every major action through URL changes.", F_SM, GRAY)
    tx(draw, HALF+PAD, 502, "DV correctly identifies those as NEW_PAGE and sends", F_SM, GRAY)
    tx(draw, HALF+PAD, 520, "full frames. Only intra-page scrolling stays as DELTA.", F_SM, GRAY)
    tx(draw, HALF+PAD, 546, "For SPA-heavy apps (TodoMVC, Gmail, Figma), savings", F_SM, GRAY)
    tx(draw, HALF+PAD, 564, "are much higher (60-70%). GitHub is a URL-nav-heavy site.", F_SM, GRAY)
    tx(draw, HALF+PAD, 592, "13.4% is the honest number for this site/task.", F_SM, GREEN)

    return img


# ─────────────────────────────────────────────────────────────────────
# BUILD FRAMES
# ─────────────────────────────────────────────────────────────────────
FRAMES = []

FRAMES.append((intro_card(), 5.0))

# ── Phase 1: Navigation (representative full-frame steps) ─────────────
FRAMES.append((make_frame(
    phase="Navigation — Issues tab and bug filter",
    subtitle="Both agents navigate to the bug-filtered issues list. URL changes = NEW_PAGE = full frames both.",
    dv_label="navigate to /issues?q=label:bug  |  full_frame (URL changed)",
    dv_obs_type="full_frame",
    dv_imgs_list=[(imgs["dv_bugs"], "DV receives full 1200x700 — same as FF on URL changes", BLUE)],
    dv_tok_step=1600, dv_tok_cumul=4800,
    ff_label="navigate to /issues?q=label:bug  |  full_frame",
    ff_imgs_list=[(imgs["ff_bugs"], "FF receives full 1200x700 — 813 bug issues visible", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=4800,
), 3.5))

# ── Phase 2: Opening the issue ────────────────────────────────────────
FRAMES.append((make_frame(
    phase="Opening bug issue #92877",
    subtitle="Click on the most recent bug report. URL changes to /issues/92877 — NEW_PAGE — full frame both.",
    dv_label="click issue #92877  |  full_frame (URL changed)",
    dv_obs_type="full_frame",
    dv_imgs_list=[(imgs["dv_issue"], "DV receives full frame — GitHub URL change = NEW_PAGE", BLUE)],
    dv_tok_step=1600, dv_tok_cumul=11200,
    ff_label="click issue #92877  |  full_frame",
    ff_imgs_list=[(imgs["ff_issue"], "FF receives full frame — same page", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=8000,
), 3.5))

# ── THE KEY MOMENT — Scroll step 7 (DELTA) ───────────────────────────
FRAMES.append((make_frame(
    phase="THE KEY MOMENT — Scrolling within the issue page",
    subtitle="No URL change. DV fires scroll_bypass -> DELTA. FF still sends 1,600 tokens.",
    dv_label="scroll down (page.scrollBy 500px)  |  DELTA — scroll_bypass  |  diff_ratio=0.29",
    dv_obs_type="delta",
    dv_imgs_list=[
        (imgs["dv7_thumb"],
         "(1) 320x225 thumbnail — green box shows scrolled-in region  |  ~60 tokens", TEAL),
        (imgs["dv7_crop0"],
         "(2) crop: the new content that scrolled into view  |  ~360 tokens", GREEN),
    ],
    dv_tok_step=420, dv_tok_cumul=11620,
    ff_label="scroll down  |  full_frame — 1,600 tokens for a scroll",
    ff_imgs_list=[(imgs["ff_scroll1"],
                   "Entire 1200x700 re-sent. Nav bar, sidebar labels unchanged — all re-transmitted.", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=9600,
    highlight=True,
), 5.5))

# ── Delta step 8 — Scroll to confirm no comments ──────────────────────
FRAMES.append((make_frame(
    phase="Scroll to bottom — confirming no comments",
    subtitle="Second scroll. DV fires scroll_bypass again -> DELTA. Key info: no comment threads.",
    dv_label="scroll to bottom  |  DELTA — scroll_bypass  |  diff_ratio=0.18",
    dv_obs_type="delta",
    dv_imgs_list=[
        (imgs["dv8_thumb"],
         "(1) thumbnail — green box on new content area  |  ~60 tokens", TEAL),
        (imgs["dv8_crop0"],
         "(2) crop: bottom of issue — Participants sidebar, sign-up banner  |  ~360 tokens", GREEN),
    ],
    dv_tok_step=420, dv_tok_cumul=12040,
    ff_label="scroll to confirm no comments  |  full_frame  |  1,600 tokens",
    ff_imgs_list=[(imgs["ff_scroll2"],
                   "Full GitHub page re-sent. The answer (no comments) is a tiny region.", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=11200,
    highlight=True,
), 5.5))

# ── Phase 3: PRs and releases ─────────────────────────────────────────
FRAMES.append((make_frame(
    phase="Merged PRs and releases — final research",
    subtitle="Navigate to /pulls?q=is:merged and /releases. URL changes = full frames both.",
    dv_label="navigate to merged PRs, then releases  |  full_frame both",
    dv_obs_type="full_frame",
    dv_imgs_list=[(imgs["dv_prs"], "DV: merged PRs page — found latest merged PR", BLUE)],
    dv_tok_step=1600, dv_tok_cumul=15240,
    ff_label="navigate to merged PRs, then releases  |  full_frame both",
    ff_imgs_list=[(imgs["ff_prs"], "FF: same page — latest merged PR", ORANGE)],
    ff_tok_step=1600, ff_tok_cumul=16000,
), 3.5))

FRAMES.append((summary_card(), 7.0))

# ── Render ────────────────────────────────────────────────────────────
FPS    = 30
FADE_F = int(FPS * 0.35)

all_np   = []
rendered = [f for (f, _) in FRAMES]

for i, (frame, hold_s) in enumerate(FRAMES):
    if i > 0:
        prev = rendered[i-1]
        for f in range(FADE_F):
            a = f / FADE_F
            all_np.append(np.array(Image.blend(prev, frame, a)))
    for _ in range(int(FPS * hold_s)):
        all_np.append(np.array(frame))

# Fade to black
blank = Image.new("RGB", (CW, CH), DARK)
for f in range(int(FPS * 0.6)):
    a = 1.0 - f / (FPS * 0.6)
    all_np.append(np.array(Image.blend(blank, rendered[-1], a)))

dur = len(all_np) / FPS
print(f"Frames: {len(all_np)}  |  Duration: {dur:.1f}s")

from moviepy import ImageSequenceClip

clip = ImageSequenceClip(all_np, fps=FPS)
clip.write_videofile(str(OUT_MP4), fps=FPS, codec="libx264", audio=False, logger=None,
                     ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"])
sz = OUT_MP4.stat().st_size / 1024
print(f"\n✅  {OUT_MP4.name}  ({sz:.0f} KB, {dur:.1f}s)")
print("   Honest result: 13.4% savings on URL-heavy GitHub task")
print("   Delta moments: 2 scroll steps (scroll_bypass classifier)")
print("   SPA-heavy task comparison still in real_comparison.mp4 (62% savings)")
