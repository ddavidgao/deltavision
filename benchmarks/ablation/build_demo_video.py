"""
DeltaVision demo video — showing what each mode SENDS TO THE MODEL.

Layout per frame:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Title: action + transition type                                    │
  ├──────────────────────────────┬──────────────────────────────────────┤
  │  FULL-FRAME                  │  DELTAVISION                         │
  │  "Model receives this →"     │  "Model receives this →"             │
  │                              │                                      │
  │  [full 1200×700 screenshot]  │  [crop zoomed in to fill panel]      │
  │  with RED box on changed      │  + small inset: WHERE on the page   │
  │  region (1-5% of area)       │                                      │
  │  "1,600 tokens / step"       │  "~400 tokens / step"                │
  ├──────────────────────────────┴──────────────────────────────────────┤
  │  [====token bar FF====]      [==DV bar==     ]  "60% less"         │
  └─────────────────────────────────────────────────────────────────────┘

Crucially: the SAME changed region occupies ~5% of the FF panel (tiny, buried in
noise) but 100% of the DV panel (zoomed in, readable). That's the story.
"""

import sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parents[2]))

BASE = Path(__file__).parent
DV_RUN  = BASE / "runs/todomvc_dv"
OUT_DIR = BASE / "video_frames"
OUT_MP4 = OUT_DIR / "deltavision_demo_v2.mp4"

# ── Step definitions ──────────────────────────────────────────────────
# bbox: [x, y, w, h] in the 1200×700 screenshot space
STEPS = [
    {
        "action":      "Initial page load",
        "transition":  "full_frame  (baseline)",
        "t1":          DV_RUN / "step_0/t0.png",
        "crop_after":  None,          # no crop — both panels show full frame
        "bbox":        None,
        "ff_tok":      1600,
        "dv_tok":      1600,
        "hold_s":      2.5,
        "obs_label":   "Initial observation — both modes send full frame",
    },
    {
        "action":      "type('buy milk') + Enter",
        "transition":  "delta  (diff_ratio=4.9%,  2 crops)",
        "t1":          DV_RUN / "step_1/t1.png",
        "crop_after":  DV_RUN / "step_1/crop_0_after.png",
        "bbox":        [447, 140, 593, 197],   # x,y,w,h
        "ff_tok":      3200,
        "dv_tok":      2000,
        "hold_s":      3.2,
        "obs_label":   "DELTA — only the new todo row changed",
    },
    {
        "action":      "type('feed cat') + Enter",
        "transition":  "delta  (diff_ratio=4.5%,  1 crop)",
        "t1":          DV_RUN / "step_2/t1.png",
        "crop_after":  DV_RUN / "step_2/crop_0_after.png",
        "bbox":        [447, 244, 593, 251],
        "ff_tok":      4800,
        "dv_tok":      2400,
        "hold_s":      3.2,
        "obs_label":   "DELTA — only the next row appeared",
    },
    {
        "action":      "type('pay rent') + Enter",
        "transition":  "delta  (diff_ratio=4.9%,  1 crop)",
        "t1":          DV_RUN / "step_3/t1.png",
        "crop_after":  DV_RUN / "step_3/crop_0_after.png",
        "bbox":        [447, 303, 593, 252],
        "ff_tok":      6400,
        "dv_tok":      2800,
        "hold_s":      3.2,
        "obs_label":   "DELTA — only the third row appeared",
    },
    {
        "action":      "click_toggle('feed cat')",
        "transition":  "delta  (diff_ratio=1.5%,  3 crops)",
        "t1":          DV_RUN / "step_4/t1_final.png",
        "crop_after":  DV_RUN / "step_4/crop_1_after.png",
        "bbox":        [446, 234, 595, 104],   # just the feed cat row
        "ff_tok":      8000,
        "dv_tok":      3200,
        "hold_s":      4.0,
        "obs_label":   "DELTA — only the checkbox + strikethrough changed  (1.5% of pixels!)",
    },
]
MAX_TOK = 8500

# ── Canvas layout constants ───────────────────────────────────────────
CW, CH    = 1440, 830        # total canvas
TITLE_H   = 52
PANEL_H   = 580
PANEL_W   = (CW - 60) // 2  # ~690 each
PAD       = 20
BAR_H     = CH - TITLE_H - PANEL_H - 20   # ~178px
INSET_W, INSET_H = 180, 100  # thumbnail inset in DV panel

DARK   = (18,  18,  22)
DARK2  = (26,  26,  32)
ORANGE = (220, 120,  50)
BLUE   = (60,  170, 240)
GREEN  = (60,  200,  90)
RED    = (220,  60,  60)
WHITE  = (240, 240, 245)
GRAY   = (110, 110, 120)
GOLD   = (240, 200,  60)

try:
    F_LG  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    F_MD  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    F_SM  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    F_XS  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    F_BIG = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
except Exception:
    F_LG = F_MD = F_SM = F_XS = F_BIG = ImageFont.load_default()


def draw_panel_border(draw, x, y, w, h, color, width=2):
    draw.rectangle([x, y, x+w, y+h], outline=color, width=width)


def render_step(step: dict) -> Image.Image:
    img = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    # ── Title bar ───────────────────────────────────────────────────
    draw.rectangle([0, 0, CW, TITLE_H], fill=DARK2)
    draw.text((PAD, 8),  f"Action:  {step['action']}", font=F_LG, fill=WHITE)
    draw.text((PAD, 32), f"DV classifier → {step['transition']}", font=F_SM, fill=GRAY)

    LP_X  = PAD
    RP_X  = CW // 2 + PAD // 2
    PY    = TITLE_H + PAD

    # ── Panel headers ───────────────────────────────────────────────
    draw.text((LP_X,      PY),     "FULL-FRAME", font=F_MD, fill=ORANGE)
    draw.text((LP_X,      PY+22),  "Model receives: full 1200×700 screenshot", font=F_SM, fill=GRAY)
    draw.text((LP_X,      PY+38),  f"{step['ff_tok']:,} tokens cumulative", font=F_SM, fill=ORANGE)

    draw.text((RP_X,      PY),     "DELTAVISION", font=F_MD, fill=BLUE)
    draw.text((RP_X,      PY+22),  step['obs_label'], font=F_SM, fill=GRAY)
    draw.text((RP_X,      PY+38),  f"{step['dv_tok']:,} tokens cumulative", font=F_SM, fill=BLUE)

    IMG_Y = PY + 60
    IMG_H = PANEL_H - 60

    # ── LEFT PANEL: full screenshot, bbox highlighted ───────────────
    if step["t1"].exists():
        full = Image.open(step["t1"]).convert("RGB")
        fw, fh = full.size
        scale = min(PANEL_W / fw, IMG_H / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        full_s = full.resize((disp_w, disp_h), Image.LANCZOS)
        img.paste(full_s, (LP_X, IMG_Y))
        draw_panel_border(draw, LP_X, IMG_Y, disp_w, disp_h, ORANGE)

        # Draw red bbox of changed region (if any)
        if step["bbox"]:
            x, y, w, h = step["bbox"]
            bx = LP_X + int(x * scale)
            by = IMG_Y + int(y * scale)
            bw = int(w * scale)
            bh = int(h * scale)
            # Highlight the tiny changed region with a bright red box
            draw.rectangle([bx, by, bx+bw, by+bh], outline=RED, width=3)
            # Add a small label
            draw.rectangle([bx, by-16, bx+bw, by], fill=RED)
            draw.text((bx+2, by-15), "CHANGED", font=F_XS, fill=WHITE)

            # Arrow pointing to the tiny box: "model must find this"
            pct = (w * h) / (fw * fh) * 100
            draw.text((LP_X, IMG_Y + disp_h + 4),
                      f"Changed region: {pct:.1f}% of total pixels  →  model must find the signal in all this noise",
                      font=F_XS, fill=RED)

    # ── RIGHT PANEL: crop zoomed in ─────────────────────────────────
    rp_img_x = RP_X
    if step["bbox"] is None or step["crop_after"] is None:
        # Step 0: DV also sends full frame
        if step["t1"].exists():
            full2 = Image.open(step["t1"]).convert("RGB")
            fw, fh = full2.size
            scale2 = min(PANEL_W / fw, IMG_H / fh)
            disp_w2 = int(fw * scale2)
            disp_h2 = int(fh * scale2)
            full_s2 = full2.resize((disp_w2, disp_h2), Image.LANCZOS)
            img.paste(full_s2, (rp_img_x, IMG_Y))
            draw_panel_border(draw, rp_img_x, IMG_Y, disp_w2, disp_h2, BLUE)
            draw.text((rp_img_x, IMG_Y + disp_h2 + 4),
                      "Step 0: DV also sends full frame (anchor reset)", font=F_XS, fill=BLUE)
    else:
        # Show zoomed crop filling the panel
        if Path(step["crop_after"]).exists():
            crop = Image.open(step["crop_after"]).convert("RGB")
            cw, ch = crop.size
            # Zoom to fill panel width, maintain aspect
            zoom_scale = min(PANEL_W / cw, IMG_H / ch)
            zoom_w = int(cw * zoom_scale)
            zoom_h = int(ch * zoom_scale)
            cropped_big = crop.resize((zoom_w, zoom_h), Image.LANCZOS)

            # Center in panel
            cx = rp_img_x + (PANEL_W - zoom_w) // 2
            img.paste(cropped_big, (cx, IMG_Y))
            draw_panel_border(draw, cx, IMG_Y, zoom_w, zoom_h, GREEN)

            # "AFTER" label
            draw.rectangle([cx, IMG_Y, cx+60, IMG_Y+18], fill=GREEN)
            draw.text((cx+2, IMG_Y+2), "AFTER", font=F_XS, fill=DARK)

            # Small inset: WHERE on the page this crop comes from
            if step["t1"].exists():
                full3 = Image.open(step["t1"]).convert("RGB")
                fw3, fh3 = full3.size
                inset = full3.resize((INSET_W, INSET_H), Image.LANCZOS)
                # draw the crop bbox on the inset
                inset_draw = ImageDraw.Draw(inset)
                x, y, w, h = step["bbox"]
                ix = int(x / fw3 * INSET_W)
                iy = int(y / fh3 * INSET_H)
                iw = int(w / fw3 * INSET_W)
                ih = int(h / fh3 * INSET_H)
                inset_draw.rectangle([ix, iy, ix+iw, iy+ih], outline=GREEN, width=2)
                inset_x = rp_img_x + PANEL_W - INSET_W - 4
                inset_y = IMG_Y + IMG_H - INSET_H - 4
                img.paste(inset, (inset_x, inset_y))
                draw.rectangle([inset_x-1, inset_y-1, inset_x+INSET_W+1, inset_y+INSET_H+1],
                               outline=BLUE, width=1)
                draw.text((inset_x, inset_y - 14), "location on page ↓", font=F_XS, fill=BLUE)

            draw.text((rp_img_x, IMG_Y + IMG_H + 4),
                      f"DV zeroes in: model only sees the changed region, zoomed in — no noise",
                      font=F_XS, fill=GREEN)

    # ── Token counter bar ──────────────────────────────────────────
    bar_y = TITLE_H + PANEL_H + 30
    half  = CW // 2

    # FF bar
    ff_frac = min(step["ff_tok"] / MAX_TOK, 1.0)
    ff_bar_w = int((half - 80) * ff_frac)
    draw.text((PAD, bar_y),      f"FF cumulative:  {step['ff_tok']:,} tokens", font=F_MD, fill=ORANGE)
    draw.rectangle([PAD, bar_y+24, PAD + half-80, bar_y+44], outline=GRAY, width=1)
    draw.rectangle([PAD, bar_y+24, PAD + ff_bar_w, bar_y+44], fill=ORANGE)

    # DV bar
    dv_frac = min(step["dv_tok"] / MAX_TOK, 1.0)
    dv_bar_w = int((half - 80) * dv_frac)
    savings = int((1 - step["dv_tok"] / step["ff_tok"]) * 100) if step["ff_tok"] else 0
    draw.text((half, bar_y),     f"DV cumulative:  {step['dv_tok']:,} tokens  ({savings}% less)", font=F_MD, fill=BLUE)
    draw.rectangle([half, bar_y+24, half + half-80, bar_y+44], outline=GRAY, width=1)
    draw.rectangle([half, bar_y+24, half + dv_bar_w, bar_y+44], fill=BLUE)

    # Final frame savings callout
    if savings >= 50:
        draw.text((CW - 180, bar_y + 5), f"↑ {savings}% less", font=F_BIG, fill=GOLD)

    return img


# ── Build animated frames ─────────────────────────────────────────────
FPS = 30
FADE_FRAMES = int(FPS * 0.5)

all_np = []
rendered = [render_step(s) for s in STEPS]

for i, step in enumerate(STEPS):
    hold_n = int(FPS * step["hold_s"])
    # Fade in from prev
    if i > 0:
        for f in range(FADE_FRAMES):
            alpha = f / FADE_FRAMES
            blended = Image.blend(rendered[i-1], rendered[i], alpha)
            all_np.append(np.array(blended))
    for _ in range(hold_n):
        all_np.append(np.array(rendered[i]))

# Fade to black
for f in range(FADE_FRAMES * 3):
    alpha = 1.0 - f / (FADE_FRAMES * 3)
    blank = Image.new("RGB", (CW, CH), DARK)
    blended = Image.blend(blank, rendered[-1], alpha)
    all_np.append(np.array(blended))

print(f"Total frames: {len(all_np)}  |  Duration: {len(all_np)/FPS:.1f}s at {FPS}fps")

# ── Export ────────────────────────────────────────────────────────────
from moviepy import ImageSequenceClip
clip = ImageSequenceClip(all_np, fps=FPS)
clip.write_videofile(str(OUT_MP4), fps=FPS, codec="libx264",
                     audio=False, logger=None,
                     ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"])

sz = OUT_MP4.stat().st_size / 1024
print(f"\n✅  {OUT_MP4.name}  —  {sz:.0f} KB  |  {len(all_np)/FPS:.1f}s")
print("   → Left panel:  full screenshot with red bbox on changed region")
print("   → Right panel: crop zoomed in to fill panel + inset showing location")
print("   → Token bars:  cumulative comparison, savings callout")
