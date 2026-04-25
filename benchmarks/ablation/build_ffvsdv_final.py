#!/usr/bin/env python3
"""
Final FF vs DV comparison video — 3 acts:
  1. Intro card (~10s):   "What does DV do? Why does it matter?"
  2. Side-by-side (~30s): FF browser left, DV browser right with INLINE spotlight
                          (blur+bbox highlights the delta DV actually sent each step)
  3. Outro card (~12s):   The 25.9% / 23.2% headline numbers + breakdown

Phantom-state DV frames (8, 9, 10 — sheet pre-filled with stale data) are
culled before rendering. DV's true working trajectory is 28 steps.

Output: /tmp/ffvsdv_final/ffvsdv_final.mp4 (1920x1080, 24fps, ~52s)
"""
import json
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
# Run_24 DV (with periodic refresh + lean screenshot prompt) vs Run_22 FF.
# Both runs do the website-fallback rent lookup. DV: 32 steps, FF: 45 steps.
# DV wins on BOTH axes: 53.2% total task savings + 34.2% per-step savings.
# This is the run that finally satisfies CLAUDE.md's "fewer steps AND less
# tokens" principle.
DV_DIR = ROOT / "benchmarks/mapsheets/results/run_24_dv_refresh/screenshots"
FF_DIR = ROOT / "benchmarks/mapsheets/results/run_22_ff_M/screenshots"
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_24_dv_refresh/dv_proxy_run_1777094341.jsonl"
FF_LOG = ROOT / "benchmarks/mapsheets/results/run_22_ff_M/dv_proxy_run_1777092155_ff.jsonl"
OUT_DIR = Path("/tmp/ffvsdv_final")
OUT_DIR.mkdir(exist_ok=True)
OUT_VIDEO = OUT_DIR / "ffvsdv_final.mp4"

# No phantom culling needed — both agents started from clean state in this run.
DV_EXCLUDE_STEPS = set()

# --- Layout ---
# Layout — matches Remotion FFvsDVComparison.tsx v3 dimensions, scaled to
# allow both browsers PLUS the racing-bar step log to fill the frame edge-to-edge
# with no dead vertical space (the old layout had a 130px bottom strip that
# looked like a "continent" between the browsers and the savings bar).
W, H = 1920, 1080
TOP_BAR_H = 88                 # matches v3 top-bar height
PANEL_GAP = 0                  # no inter-panel gap, browsers share vertical divider
PANEL_W = (W - PANEL_GAP) // 2
SHOT_H = 600                   # taller browser viewport
LOG_H = H - TOP_BAR_H - SHOT_H  # remaining vertical space → step-log table
                               # (1080 - 88 - 600 = 392px for the log)

# Final headline numbers (from ff_vs_dv_culled.py)
HEADLINE_PER_STEP_PCT = 34.2
HEADLINE_TOTAL_PCT = 53.2
FF_TOTAL_TOKENS = 61425
DV_TOTAL_TOKENS = 28725
FF_STEPS = 45
DV_STEPS = 32

# --- Colors (v3 palette via FFvsDVComparison.tsx) ---
BG = (9, 9, 9)                 # C.bg — near-black, not blue-tinted
PANEL_BG = (14, 14, 16)        # C.panel
FG = (240, 240, 240)           # C.fg
DIM = (120, 120, 120)          # C.muted
DIMMED = (90, 90, 95)          # C.dimmed (subtler than DIM)
CYAN = (20, 184, 166)          # C.dv — Tailwind teal-500
RED = (248, 113, 113)          # C.ff — Tailwind red-400
YELLOW = (250, 204, 21)        # C.problem
PILL_RED_BG = (40, 16, 18)     # darker tinted FF pill bg
PILL_GREEN_BG = (8, 32, 30)    # darker tinted DV pill bg

FPS = 24
INTRO_S = 8           # "THE PROBLEM" card
INSIGHT_S = 8         # "THE INSIGHT" two-screenshots-look-the-same card
SBS_S = 30
OUTRO_S = 10          # simplified outro (no cards / no footnote)

# Insight frame asset paths — Chicago FF shot_003 + shot_004, the two
# consecutive frames where the only thing that changed is the sidebar
# collapse. Sourced from dv-video-scratch/.../public/ff_chicago/.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
INSIGHT_BEFORE = ASSETS_DIR / "insight_before.png"
INSIGHT_AFTER = ASSETS_DIR / "insight_after.png"

# Spotlight params
BLUR_RADIUS = 8
DIM_ALPHA_OUTSIDE = 0.4   # outside-bbox brightness (40% — visible but defocused)
BBOX_PAD = 24
BBOX_BORDER_COLOR = CYAN
BBOX_BORDER_WIDTH = 4
MIN_DIFF_FRACTION = 0.003


# Font sources — exact match to v3's Remotion `FONT` and `MONO` stacks.
# Inter (proportional) and JetBrains Mono (monospace) ship in benchmarks/ablation/.fonts/
# so renders are reproducible across machines (no "looks fine on macOS, garbage on
# Linux CI" font-substitution surprises).
_FONT_DIR = Path(__file__).resolve().parent / ".fonts"
_INTER = {
    "regular":  _FONT_DIR / "Inter-Regular.ttf",
    "medium":   _FONT_DIR / "Inter-Medium.ttf",
    "semibold": _FONT_DIR / "Inter-SemiBold.ttf",
    "bold":     _FONT_DIR / "Inter-Bold.ttf",
}
_JBMONO = {
    "regular": _FONT_DIR / "JetBrainsMono-Regular.ttf",
    "medium":  _FONT_DIR / "JetBrainsMono-Medium.ttf",
}


def font(sz, mono=False, weight="regular", bold=None):
    """
    Returns a PIL ImageFont matching v3's typography exactly.

    `weight`: "regular" | "medium" | "semibold" | "bold". Inter has all four;
              JetBrainsMono has regular + medium only.
    `bold`:  legacy back-compat alias — `bold=True` → weight="bold". Older call
              sites in this file used the bool form before the rewrite.
    """
    if bold is True:
        weight = "bold"
    elif bold is False and weight == "regular":
        weight = "regular"
    table = _JBMONO if mono else _INTER
    # Map legacy bool to weight string for the back-compat path
    if weight is True:
        weight = "bold"
    elif weight is False:
        weight = "regular"
    # JetBrainsMono only has regular/medium
    if mono and weight in ("semibold", "bold"):
        weight = "medium"
    path = table.get(weight, table["regular"])
    if not path.exists():
        # Fallback to system font if the .fonts dir is somehow missing
        sys_path = "/System/Library/Fonts/HelveticaNeue.ttc"
        if os.path.exists(sys_path):
            try:
                return ImageFont.truetype(sys_path, sz, index=1 if weight == "bold" else 0)
            except Exception:
                pass
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), sz)


def load_steps(p):
    out = []
    for line in open(p):
        d = json.loads(line)
        if "step" in d:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Act 1: Intro
# ---------------------------------------------------------------------------

def render_intro_frame(t=0.0):
    """
    Lifted from ffvsdv_comparison_v3.mp4's "THE PROBLEM" intro card.

    Layout:
        - Yellow eyebrow "THE PROBLEM"
        - Bold sans headline: "Every action, your agent sends the entire screen
          to the model." with "entire screen" in red
        - Two sub-bullets in dimmer gray + bold gray
        - Red callout box: "1,365 image tokens per screenshot · every single step"
          + secondary line "31-step task = 42,315 tokens before the model writes
          a single word"
    """
    YELLOW = (240, 200, 60)
    HEADLINE_FG = (235, 235, 235)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_eyebrow = font(20, mono=True)              # tracking-letterspaced; mono is fine
    f_headline = font(78, mono=False, bold=True)  # bold sans
    f_subline = font(28, mono=False)             # plain sans, secondary
    f_subline_bold = font(28, mono=False, bold=True)
    f_callout_main = font(28, mono=True)          # mono inside a red callout (like v3)
    f_callout_secondary = font(18, mono=True)

    # --- Eyebrow (yellow, letterspaced) ---
    d.text((W // 2, 200), "T H E   P R O B L E M",
           font=f_eyebrow, fill=YELLOW, anchor="mm")

    # --- Headline (bold sans) ---
    # Render the headline as 2 lines so we can highlight "entire screen" in red.
    # Compute layout manually for centering: line 1 = "Every action, your agent sends"
    # Line 2 = "the [entire screen] to the model."
    line1 = "Every action, your agent sends"
    line2_pre = "the "
    line2_red = "entire screen"
    line2_post = " to the model."

    # Measure line 2 widths to center the colored span correctly
    pre_w = d.textlength(line2_pre, font=f_headline)
    red_w = d.textlength(line2_red, font=f_headline)
    post_w = d.textlength(line2_post, font=f_headline)
    line2_total = pre_w + red_w + post_w
    line2_x = (W - line2_total) // 2

    d.text((W // 2, 320), line1, font=f_headline, fill=HEADLINE_FG, anchor="mm")
    line2_y = 415
    d.text((line2_x, line2_y), line2_pre, font=f_headline, fill=HEADLINE_FG, anchor="lm")
    d.text((line2_x + pre_w, line2_y), line2_red, font=f_headline, fill=RED, anchor="lm")
    d.text((line2_x + pre_w + red_w, line2_y), line2_post,
           font=f_headline, fill=HEADLINE_FG, anchor="lm")

    # --- Sub-bullets ---
    # "Navigate to a new page? Full screenshot."
    # "Click a button and one cell updates? Full screenshot."
    bullet_y = 540
    bullets = [
        ("Navigate to a new page?", " Full screenshot."),
        ("Click a button and one cell updates?", " Full screenshot."),
    ]
    for q, a in bullets:
        q_w = d.textlength(q, font=f_subline)
        a_w = d.textlength(a, font=f_subline_bold)
        total = q_w + a_w
        x = (W - total) // 2
        d.text((x, bullet_y), q, font=f_subline, fill=DIM, anchor="lm")
        d.text((x + q_w, bullet_y), a, font=f_subline_bold, fill=HEADLINE_FG, anchor="lm")
        bullet_y += 50

    # --- Red callout box ---
    # Numbers from THE ACTUAL FF RUN we just measured: FF_STEPS × 1365 = FF_TOTAL_TOKENS
    callout_main = "1,365 image tokens per screenshot  ·  every single step"
    callout_sec = (f"{FF_STEPS}-step task = {FF_TOTAL_TOKENS:,} tokens "
                   "before the model writes a single word")

    main_w = d.textlength(callout_main, font=f_callout_main)
    sec_w = d.textlength(callout_sec, font=f_callout_secondary)
    box_w = max(main_w, sec_w) + 80
    box_h = 120
    box_x = (W - box_w) // 2
    box_y = 740

    d.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                        radius=12, fill=(34, 16, 18), outline=RED, width=1)
    d.text((W // 2, box_y + 42), callout_main,
           font=f_callout_main, fill=RED, anchor="mm")
    d.text((W // 2, box_y + 88), callout_sec,
           font=f_callout_secondary, fill=(170, 110, 115), anchor="mm")

    return img


# ---------------------------------------------------------------------------
# Act 1.5: "THE INSIGHT" — two near-identical Maps screenshots side-by-side
# ---------------------------------------------------------------------------

def render_insight_frame():
    """
    Bridges the intro and the SBS demo with a concrete "look at this" beat.

    Two consecutive Chicago Maps screenshots that look almost identical —
    only the search-results sidebar collapsed. Highlights the cyan callout
    box on the after-shot to show "this is the only thing that changed",
    then says: this is why DeltaVision exists.

    Lifted from FFvsDVComparison.tsx (v3 Remotion source) — same shots,
    same callout, same copy. Rendered here in PIL at 1920x1080 to match
    the rest of the video.
    """
    YELLOW = (240, 200, 60)
    HEADLINE_FG = (235, 235, 235)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # --- Eyebrow ---
    d.text((W // 2, 110), "T H E   I N S I G H T",
           font=font(20, mono=True), fill=CYAN, anchor="mm")

    # --- Headline ---
    f_headline = font(64, bold=True)
    line1 = "Two consecutive screenshots."
    line2_pre = "Almost "
    line2_red = "nothing changed"
    line2_post = "."

    d.text((W // 2, 200), line1, font=f_headline, fill=HEADLINE_FG, anchor="mm")

    pre_w = d.textlength(line2_pre, font=f_headline)
    red_w = d.textlength(line2_red, font=f_headline)
    post_w = d.textlength(line2_post, font=f_headline)
    total = pre_w + red_w + post_w
    line2_x = (W - total) // 2
    line2_y = 280
    d.text((line2_x, line2_y), line2_pre, font=f_headline, fill=HEADLINE_FG, anchor="lm")
    d.text((line2_x + pre_w, line2_y), line2_red, font=f_headline, fill=RED, anchor="lm")
    d.text((line2_x + pre_w + red_w, line2_y), line2_post,
           font=f_headline, fill=HEADLINE_FG, anchor="lm")

    # --- Two screenshots side-by-side ---
    shot_w = 760
    shot_h = 460
    gap = 80
    shots_total_w = shot_w * 2 + gap
    left_x = (W - shots_total_w) // 2
    right_x = left_x + shot_w + gap
    shots_y = 380

    # STEP labels (mono, letterspaced)
    f_step_lbl = font(15, mono=True)
    d.text((left_x + shot_w // 2, shots_y - 24),
           "S T E P   4   ·   s c r e e n s h o t",
           font=f_step_lbl, fill=DIM, anchor="mm")
    d.text((right_x + shot_w // 2, shots_y - 24),
           "S T E P   5   ·   s c r e e n s h o t",
           font=f_step_lbl, fill=DIM, anchor="mm")

    # Load the two shots; crop to top portion (objectPosition: top in v3) so
    # we get the search-results sidebar + map header where the visible delta
    # actually lives.
    if INSIGHT_BEFORE.exists() and INSIGHT_AFTER.exists():
        before = Image.open(INSIGHT_BEFORE).convert("RGB")
        after = Image.open(INSIGHT_AFTER).convert("RGB")

        def _fit_top(im, w, h):
            # Resize to width=w, then crop top h pixels
            ratio = w / im.width
            new_h = int(im.height * ratio)
            scaled = im.resize((w, new_h), Image.LANCZOS)
            return scaled.crop((0, 0, w, min(h, new_h)))

        before_fit = _fit_top(before, shot_w, shot_h)
        after_fit = _fit_top(after, shot_w, shot_h)
        img.paste(before_fit, (left_x, shots_y))
        img.paste(after_fit, (right_x, shots_y))

    # 1px hairline border around each shot
    for x in (left_x, right_x):
        d.rectangle([x, shots_y, x + shot_w, shots_y + shot_h],
                    outline=(40, 40, 44), width=1)

    # --- Cyan callout box on the AFTER shot (the changed sidebar region) ---
    # In v3 the box covers a 240x130 region at top-left of the 680x400 shot.
    # We've scaled to 760x460 so scale the box accordingly.
    cb_w = int(240 * shot_w / 680)   # ≈ 268
    cb_h = int(130 * shot_h / 400)   # ≈ 150
    cb_x = right_x + 10
    cb_y = shots_y + 10
    d.rectangle([cb_x, cb_y, cb_x + cb_w, cb_y + cb_h],
                outline=CYAN, width=3)
    # "↑ only this changed" pill below the box
    pill_text = "↑ only this changed"
    f_pill = font(15, mono=True)
    pill_w = d.textlength(pill_text, font=f_pill) + 18
    pill_h = 26
    pill_x = cb_x
    pill_y = cb_y + cb_h + 6
    d.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                        radius=4, fill=(9, 9, 9))
    d.text((pill_x + 9, pill_y + pill_h // 2), pill_text,
           font=f_pill, fill=CYAN, anchor="lm")

    # --- Token labels under each shot ---
    f_tok = font(20, mono=True)
    d.text((left_x + shot_w // 2, shots_y + shot_h + 36),
           "+1,365 tokens sent", font=f_tok, fill=RED, anchor="mm")

    after_label_main = "+1,365 tokens sent"
    after_label_paren = "  (again)"
    main_w = d.textlength(after_label_main, font=f_tok)
    paren_w = d.textlength(after_label_paren, font=font(16, mono=True))
    total_w = main_w + paren_w
    al_x = right_x + (shot_w - total_w) // 2
    al_y = shots_y + shot_h + 36
    d.text((al_x, al_y), after_label_main, font=f_tok, fill=RED, anchor="lm")
    d.text((al_x + main_w, al_y + 2), after_label_paren,
           font=font(16, mono=True), fill=DIM, anchor="lm")

    # --- Bottom prompt + answer ---
    YELLOW  # noqa: B015 — name preserved for parity with intro card
    d.text((W // 2, 920), "The sidebar collapsed. That's it.",
           font=font(34), fill=DIM, anchor="mm")
    d.text((W // 2, 980), "DeltaVision sends only what changed.",
           font=font(40, bold=True), fill=CYAN, anchor="mm")

    return img


# ---------------------------------------------------------------------------
# Act 2: Side-by-side with inline spotlight on the DV panel
# ---------------------------------------------------------------------------

def compute_bbox(prev_bgr, cur_bgr, pad=BBOX_PAD):
    """Compute the bbox of the LARGEST localized delta between two frames.

    Earlier this function unioned every changed contour into one super-bbox.
    On Maps frame transitions where the main listing-sidebar swap is accompanied
    by a few tiny scattered UI tweaks (search-bar pulse, marker reflows,
    scrollbar ghost), the union spanned ~98% of the frame, tripping the
    "≥80% → no spotlight" gate and showing a flat full-frame view.

    Now: detect each contour separately, sort by area, return the LARGEST one.
    On dvp_04→dvp_05 (the 0:15 frame), this returns (481, 41, 448, 370) =
    16% of frame — exactly the listing sidebar — which the spotlight gate
    happily passes through.

    Returns (x, y, w, h) or None if no significant change.
    """
    if prev_bgr is None or cur_bgr is None:
        return None
    if prev_bgr.shape != cur_bgr.shape:
        prev_bgr = cv2.resize(prev_bgr, (cur_bgr.shape[1], cur_bgr.shape[0]))
    diff = cv2.absdiff(prev_bgr, cur_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    changed = mask.sum() / 255 / mask.size
    if changed < MIN_DIFF_FRACTION:
        return None
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Pick the LARGEST single contour bbox, not the union.
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    Hh, Ww = cur_bgr.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(Ww - x, w + pad * 2)
    h = min(Hh - y, h + pad * 2)
    return (x, y, w, h)


def apply_spotlight(pil_img, bbox, transition):
    """
    Apply blur+bbox spotlight to a PIL image.
    - If transition=='new_page' OR bbox covers ≥80% of frame: show full sharp.
    - Otherwise: blur outside bbox, draw cyan rectangle around bbox.
    """
    img = pil_img.convert("RGB")
    Wp, Hp = img.size

    if transition == "new_page" or bbox is None:
        return img

    bx, by, bw, bh = bbox
    area_frac = (bw * bh) / (Wp * Hp)
    if area_frac >= 0.8:
        return img

    # Blur the whole image, dim it
    blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    overlay = Image.new("RGBA", (Wp, Hp), (0, 0, 0, int(255 * (1 - DIM_ALPHA_OUTSIDE))))
    base = Image.alpha_composite(blurred.convert("RGBA"), overlay).convert("RGB")

    # Paste the sharp bbox region back over the dimmed background
    sharp = img.crop((bx, by, bx + bw, by + bh))
    base.paste(sharp, (bx, by))

    # Cyan border
    d = ImageDraw.Draw(base)
    for i in range(BBOX_BORDER_WIDTH):
        d.rectangle([bx - i, by - i, bx + bw + i, by + bh + i], outline=BBOX_BORDER_COLOR)

    return base


def draw_top_strip(draw, frame_w, ff_idx, dv_idx, ff_n, dv_n):
    """
    Thin 88px header strip across the top of the SBS frame. Two columns
    matching the browsers below: left = "FULL FRAME (FF)", right = "DELTAVISION (DV)".
    Plus a step counter on the far right (matches v3's STEP card).
    """
    draw.rectangle([0, 0, frame_w, TOP_BAR_H], fill=PANEL_BG)

    f_eyebrow = font(11, mono=True)              # letterspaced caption above title
    f_title = font(22, weight="semibold")        # main title
    f_sub = font(13)

    # FF column (left half)
    draw.text((32, 14), "FULL-FRAME BASELINE", font=f_eyebrow, fill=RED)
    draw.text((32, 32), "Sends every screenshot in full",
              font=f_title, fill=FG)
    draw.text((32, 62), f"1,365 tokens × {ff_n} steps", font=f_sub, fill=DIM)

    # DV column (right half)
    draw.text((frame_w // 2 + 32, 14), "DELTAVISION  ·  LIVE PROXY",
              font=f_eyebrow, fill=CYAN)
    draw.text((frame_w // 2 + 32, 32), "Sends only what changed",
              font=f_title, fill=FG)
    draw.text((frame_w // 2 + 32, 62),
              "cyan box = sent  ·  blurred = withheld",
              font=f_sub, fill=DIM)

    # Vertical divider between the two columns
    draw.line([(frame_w // 2, 0), (frame_w // 2, TOP_BAR_H + SHOT_H)],
              fill=(40, 40, 44), width=1)


def _format_thousands(n):
    return f"{n:,}"


def draw_step_log(draw, canvas, frame_w, ff_step, dv_step,
                  all_ff_steps, all_dv_steps,
                  ff_idx, dv_idx, ff_max):
    """
    Two stacked racing bars below the SBS browsers.

    Top:    FULL FRAME BASELINE · Chicago FF      <ff_cumulative> tokens
            ████████████████████░░░░░░░░░░░
    Bottom: DELTAVISION · SF DV proxy (live)      <dv_cumulative> tokens
            ████████░░░░░░░░░░░░░░░░░░░░░░░

    Both bars share the same 100% scale = FF total tokens. The DV bar fills
    progressively slower, ending well before FF's bar at 100%. This is the
    payoff visualization — the moment you watch the cyan bar lag behind
    the red bar, the savings story is told without copy.
    """
    log_top = TOP_BAR_H + SHOT_H
    log_bot = H

    # Dark backdrop
    draw.rectangle([0, log_top, frame_w, log_bot], fill=BG)

    # ─── Two racing bars ──────────────────────────────────────────────────
    # Layout: each bar lives in a 200px tall band; both bars share the same
    # 100% scale = ff_max (final FF cumulative tokens). The DV bar fills
    # progressively slower than FF, so by end-of-task FF is full and DV
    # is at ~46% — that gap IS the savings story.
    margin_x = 80
    bar_x = margin_x
    bar_w = frame_w - 2 * margin_x
    bar_h = 32          # tall, chunky bar — matches the v3 racing-bar reference

    # Row labels (eyebrow-style) and token counts use Inter / JetBrainsMono.
    f_label = font(20, mono=True, weight="medium")
    f_count = font(22, mono=True, weight="medium")

    # Vertical layout inside the bottom strip
    band_h = (log_bot - log_top) // 2
    ff_band_top = log_top
    dv_band_top = log_top + band_h

    def _draw_bar(band_top, label, label_color, count_str, fill_pct, fill_color):
        """One racing bar with eyebrow label (left) + count (right) above the track."""
        # Label and count — aligned to the bar edges, sitting just above the track
        text_y = band_top + (band_h - bar_h) // 2 - 30
        draw.text((bar_x, text_y), label,
                  font=f_label, fill=label_color, anchor="la")
        draw.text((bar_x + bar_w, text_y), count_str,
                  font=f_count, fill=label_color, anchor="ra")

        # Track (dark slot) — full bar width
        bar_y = band_top + (band_h - bar_h) // 2
        track_color = (24, 24, 28)
        draw.rounded_rectangle(
            [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
            radius=bar_h // 2, fill=track_color,
        )
        # Filled portion — inset by 1px so the rounded ends look soft
        fill_w = int(bar_w * max(0.0, min(1.0, fill_pct)))
        if fill_w > bar_h:  # only draw if at least one full corner of fill
            draw.rounded_rectangle(
                [bar_x, bar_y, bar_x + fill_w, bar_y + bar_h],
                radius=bar_h // 2, fill=fill_color,
            )

    ff_pct = ff_step["ff_cumulative"] / max(1, ff_max)
    dv_pct = dv_step["dv_cumulative"] / max(1, ff_max)

    _draw_bar(
        ff_band_top,
        "FULL FRAME BASELINE  ·  SF FF proxy (live)",
        RED,
        f"{_format_thousands(ff_step['ff_cumulative'])} tokens",
        ff_pct,
        RED,
    )
    _draw_bar(
        dv_band_top,
        "DELTAVISION  ·  SF DV proxy (live)",
        CYAN,
        f"{_format_thousands(dv_step['dv_cumulative'])} tokens",
        dv_pct,
        CYAN,
    )


# ---------------------------------------------------------------------------
# Act 3: Outro
# ---------------------------------------------------------------------------

def render_outro_frame():
    """
    Stripped-back outro — just three things:
      • "34.2% fewer tokens/turn"      (per-step savings, the headline number)
      • "53.2% fewer tokens total"     (total task savings)
      • pip install deltavision        (the call to action)
    No cards, no footnote, no accrual line. The video has earned this moment;
    the numbers don't need decoration.
    """
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Top stat: 34.2% fewer tokens/turn
    f_big = font(180, bold=True)
    f_med = font(40)
    d.text((W // 2, 290), f"{HEADLINE_PER_STEP_PCT}%",
           font=f_big, fill=CYAN, anchor="mm")
    d.text((W // 2, 410), "fewer tokens per turn",
           font=f_med, fill=FG, anchor="mm")

    # Bottom stat: 53.2% fewer tokens total
    d.text((W // 2, 600), f"{HEADLINE_TOTAL_PCT}%",
           font=f_big, fill=CYAN, anchor="mm")
    d.text((W // 2, 720), "fewer tokens total",
           font=f_med, fill=FG, anchor="mm")

    # Install pill
    pill_w, pill_h = 480, 76
    pill_x = (W - pill_w) // 2
    pill_y = 880
    d.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                        radius=10, outline=CYAN, width=2)
    d.text((W // 2, pill_y + pill_h // 2),
           "pip install deltavision",
           font=font(32, mono=True), fill=CYAN, anchor="mm")

    return img


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def main():
    # Load + cull DV
    ff_steps = load_steps(FF_LOG)
    dv_steps_all = load_steps(DV_LOG)
    # Cull phantom DV steps AND recompute dv_cumulative over the kept steps so
    # the SBS card matches the outro headline. Without this, dv_cumulative
    # carries the phantom tokens forward and the SBS "SAVED %" reads lower
    # than the outro's headline (18% vs 23.2%).
    dv_steps = []
    dv_cum_culled = 0
    for s in dv_steps_all:
        if s["step"] in DV_EXCLUDE_STEPS:
            continue
        dv_cum_culled += s["dv_tokens"]
        # Make a shallow copy so we don't mutate the original log dicts
        dv_steps.append({**s, "dv_cumulative": dv_cum_culled})

    # Files in order, culled correspondingly. Filename pattern: dvp_<step>_*.png
    ff_files = sorted(os.listdir(FF_DIR))
    dv_files_all = sorted(os.listdir(DV_DIR))
    dv_files = [f for f in dv_files_all
                if int(f.split("_")[1]) not in DV_EXCLUDE_STEPS]

    assert len(ff_steps) == len(ff_files), \
        f"FF mismatch: {len(ff_steps)} log vs {len(ff_files)} files"
    assert len(dv_steps) == len(dv_files), \
        f"DV mismatch after cull: {len(dv_steps)} log vs {len(dv_files)} files"

    # Final FF cumulative — used to normalize the racing bars so both fill 100%
    # of their channel exactly at task end. Also used by the intro callout.
    ff_total_final = ff_steps[-1]["ff_cumulative"] if ff_steps else 1
    print(f"FF: {len(ff_steps)} steps, total {ff_total_final:,} tokens")
    print(f"DV (culled): {len(dv_steps)} steps (removed {sorted(DV_EXCLUDE_STEPS)})")

    # ---- Act 1: Intro (still frame) ----
    print("\n[Act 1] Rendering intro card...")
    intro_img = render_intro_frame(0)
    intro_path = OUT_DIR / "intro.png"
    intro_img.save(intro_path)

    # ---- Act 1.5: Insight (still frame) ----
    print("[Act 1.5] Rendering insight card (two consecutive screenshots)...")
    insight_img = render_insight_frame()
    insight_path = OUT_DIR / "insight.png"
    insight_img.save(insight_path)

    # ---- Act 3: Outro (still frame) ----
    print("[Act 3] Rendering outro card...")
    outro_img = render_outro_frame()
    outro_path = OUT_DIR / "outro.png"
    outro_img.save(outro_path)

    # ---- Act 2: Side-by-side with spotlight ----
    print("[Act 2] Rendering side-by-side frames...")
    sbs_total_frames = FPS * SBS_S
    ff_per_step = sbs_total_frames / len(ff_steps)
    dv_per_step = sbs_total_frames / len(dv_steps)

    panel_left_x = (W - 2 * PANEL_W - PANEL_GAP) // 2
    panel_right_x = panel_left_x + PANEL_W + PANEL_GAP

    # Pre-load all frames as BGR numpy arrays for bbox computation
    print("  preloading DV frames (for bbox computation)...")
    dv_native = [cv2.imread(str(DV_DIR / f)) for f in dv_files]
    dv_pil = [Image.open(DV_DIR / f).convert("RGB") for f in dv_files]
    ff_pil = [Image.open(FF_DIR / f).convert("RGB") for f in ff_files]

    sbs_frame_paths = []
    for fi in range(sbs_total_frames):
        ff_idx = min(int(fi / ff_per_step), len(ff_steps) - 1)
        dv_idx = min(int(fi / dv_per_step), len(dv_steps) - 1)
        ff_step = ff_steps[ff_idx]
        dv_step = dv_steps[dv_idx]

        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)

        # 1. Top strip with FF + DV column titles
        draw_top_strip(draw, W, ff_idx, dv_idx, len(ff_steps), len(dv_steps))

        # 2. Browsers — flush against the top strip, fill full width split 50/50
        panel_top = TOP_BAR_H
        ff_img = ff_pil[ff_idx].resize((PANEL_W, SHOT_H), Image.LANCZOS)
        canvas.paste(ff_img, (0, panel_top))

        # DV side: blur+spotlight applied at native resolution, then scaled
        if dv_idx > 0:
            bbox_native = compute_bbox(dv_native[dv_idx - 1], dv_native[dv_idx])
        else:
            bbox_native = None
        dv_spotlit_native = apply_spotlight(dv_pil[dv_idx], bbox_native,
                                            dv_step["transition"])
        dv_img = dv_spotlit_native.resize((PANEL_W, SHOT_H), Image.LANCZOS)
        canvas.paste(dv_img, (PANEL_W, panel_top))

        # 3. Step log fills the rest of the frame (no dead space)
        draw_step_log(draw, canvas, W, ff_step, dv_step,
                      ff_steps, dv_steps, ff_idx, dv_idx,
                      ff_max=ff_total_final)

        out_p = OUT_DIR / f"sbs_{fi:04d}.png"
        canvas.save(out_p)
        sbs_frame_paths.append(str(out_p))
        if (fi + 1) % 60 == 0 or fi == sbs_total_frames - 1:
            print(f"  rendered {fi + 1}/{sbs_total_frames}")

    # ---- Splice: intro (still) + insight (still) + sbs (frames) + outro (still) ----
    print("\nEncoding final video...")
    list_path = OUT_DIR / "concat.txt"
    with open(list_path, "w") as f:
        # intro: hold for INTRO_S
        for _ in range(FPS * INTRO_S):
            f.write(f"file '{intro_path}'\nduration {1/FPS:.5f}\n")
        # insight: hold for INSIGHT_S — bridges "the problem" → "the demo"
        for _ in range(FPS * INSIGHT_S):
            f.write(f"file '{insight_path}'\nduration {1/FPS:.5f}\n")
        # sbs: each frame at 1/FPS
        for p in sbs_frame_paths:
            f.write(f"file '{p}'\nduration {1/FPS:.5f}\n")
        # outro: hold for OUTRO_S
        for _ in range(FPS * OUTRO_S):
            f.write(f"file '{outro_path}'\nduration {1/FPS:.5f}\n")
        # concat demuxer wants final entry without duration
        f.write(f"file '{outro_path}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vf", f"fps={FPS}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(OUT_VIDEO),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("ffmpeg stderr:")
        print(r.stderr.decode()[-1500:])
        return

    dur = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(OUT_VIDEO)]
    ).strip()
    print(f"\nDone: {OUT_VIDEO}  ({dur.decode()}s)")


if __name__ == "__main__":
    main()
