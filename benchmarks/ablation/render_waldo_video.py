"""
Render the "Where's Waldo" demo video.

Side-by-side comparison:
  LEFT   — Full-frame mode ("find the diff")
  RIGHT  — DeltaVision mode (green box + zoomed crops = "here's Waldo")

Reads artifacts from benchmarks/ablation/waldo_demo/step_NN/*.
Output: benchmarks/ablation/video_frames/waldo_comparison.mp4
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from moviepy import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

# ============================================================= layout

W, H = 1680, 1000
FPS = 30
HOLD_PER_STEP = 5.0
FADE_FRAMES = 10
INTRO_HOLD = 6.0
OUTRO_HOLD = 8.0

HEADER = (0, 0, W, 90)
LEFT_PANEL = (20, 110, 810, 780)
RIGHT_PANEL = (830, 110, 1660, 780)
FOOTER = (20, 800, 1660, 980)

# ============================================================= colors

BG = (10, 10, 15)
PANEL_BG = (22, 22, 32)
HEADER_BG = (30, 60, 120)
BORDER = (55, 55, 72)

WHITE = (240, 240, 245)
GRAY = (160, 160, 175)
DIM = (105, 105, 120)
GREEN = (70, 220, 125)
YELLOW = (240, 210, 90)
RED = (240, 90, 90)
BLUE = (90, 160, 240)
CYAN = (110, 200, 220)
ORANGE = (240, 160, 80)

# ============================================================= fonts

_FONT_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(sz):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default()


F_HUGE = _font(44)
F_BIG = _font(26)
F_MED = _font(20)
F_SM = _font(16)
F_TINY = _font(13)


# ============================================================= helpers

def paste_fit(canvas, img, bbox, pad_top=34):
    x1, y1, x2, y2 = bbox
    iw = (x2 - x1) - 16
    ih = (y2 - y1) - pad_top - 8
    r = min(iw / img.width, ih / img.height)
    nw = max(1, int(img.width * r))
    nh = max(1, int(img.height * r))
    scaled = img.resize((nw, nh), Image.LANCZOS)
    ox = x1 + 8 + (iw - nw) // 2
    oy = y1 + pad_top + (ih - nh) // 2
    canvas.paste(scaled, (ox, oy))


def box(draw, bbox, fill=PANEL_BG, outline=BORDER, width=1):
    draw.rectangle(bbox, fill=fill, outline=outline, width=width)


# ============================================================= frame rendering

def render_step(meta: dict, ff_img: Image.Image, dv_thumb: Image.Image,
                dv_crops: list, cum_ff: int, cum_dv: int, total_steps: int) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # ---- HEADER ----
    box(draw, HEADER, fill=HEADER_BG, outline=HEADER_BG)
    draw.text((30, 14), "Where's Waldo?", font=F_HUGE, fill=WHITE)
    draw.text((30, 62), "Can you spot what changed?", font=F_MED, fill=(210, 220, 240))

    step_n = meta["step"]
    action = meta["action_label"]
    info_txt = f"Step {step_n}/{total_steps}   Action: {action}"
    draw.text((W - 30, 30), info_txt, font=F_MED, fill=WHITE, anchor="ra")
    tag = "NEW_PAGE" if meta.get("transition") == "new_page" else "DELTA"
    tag_color = YELLOW if tag == "NEW_PAGE" else GREEN
    draw.text((W - 30, 62), f"classifier: {tag}  (trigger={meta.get('trigger', '-')})",
              font=F_SM, fill=tag_color, anchor="ra")

    # ---- LEFT: Full-frame ----
    box(draw, LEFT_PANEL)
    lx1, ly1, lx2, ly2 = LEFT_PANEL
    draw.text((lx1 + 14, ly1 + 8), "Full-frame model's view",
              font=F_BIG, fill=RED)
    draw.text((lx1 + 14, ly1 + 42), f"cost: {meta['ff_tokens']:,} tokens",
              font=F_SM, fill=(220, 120, 120))
    paste_fit(canvas, ff_img, LEFT_PANEL, pad_top=74)

    # Hint caption at bottom of left panel
    cap = "The whole page. Find the diff on your own."
    draw.text((lx1 + 14, ly2 - 24), cap, font=F_SM, fill=DIM)

    # ---- RIGHT: DeltaVision ----
    box(draw, RIGHT_PANEL)
    rx1, ry1, rx2, ry2 = RIGHT_PANEL
    draw.text((rx1 + 14, ry1 + 8), "DeltaVision's view",
              font=F_BIG, fill=GREEN)
    draw.text((rx1 + 14, ry1 + 42), f"cost: {meta['dv_tokens']:,} tokens",
              font=F_SM, fill=(130, 220, 150))

    if meta.get("transition") == "new_page":
        # No crops — DV sent full frame
        paste_fit(canvas, dv_thumb, RIGHT_PANEL, pad_top=74)
        cap = "NEW_PAGE: DV escalates to full-frame too. This is the one step where both pay the same."
        draw.text((rx1 + 14, ry2 - 24), cap, font=F_SM, fill=ORANGE)
    else:
        # Layout: thumbnail on top, crops below side by side
        # Thumbnail occupies top 60% of right panel
        THUMB_BBOX = (rx1, ry1, rx2, ry1 + 430)
        paste_fit(canvas, dv_thumb, THUMB_BBOX, pad_top=74)

        # Crops occupy bottom half
        crop_y1 = ry1 + 440
        crop_y2 = ry2 - 30
        # Two columns for crops
        if len(dv_crops) >= 1:
            CROP1 = (rx1 + 20, crop_y1, (rx1 + rx2) // 2 - 10, crop_y2)
            box(draw, CROP1, fill=(16, 16, 24), outline=BORDER)
            draw.text((CROP1[0] + 10, CROP1[1] + 5), "Crop 1 (detail)",
                      font=F_TINY, fill=CYAN)
            paste_fit(canvas, dv_crops[0], CROP1, pad_top=26)
        if len(dv_crops) >= 2:
            CROP2 = ((rx1 + rx2) // 2 + 10, crop_y1, rx2 - 20, crop_y2)
            box(draw, CROP2, fill=(16, 16, 24), outline=BORDER)
            draw.text((CROP2[0] + 10, CROP2[1] + 5), "Crop 2 (detail)",
                      font=F_TINY, fill=CYAN)
            paste_fit(canvas, dv_crops[1], CROP2, pad_top=26)

        cap = "Green box = what moved. Crops = high-def close-up. Waldo handed over."
        draw.text((rx1 + 14, ry2 - 24), cap, font=F_SM, fill=GREEN)

    # ---- FOOTER: running totals + savings bar ----
    fx1, fy1, fx2, fy2 = FOOTER
    box(draw, FOOTER, fill=(18, 18, 26))

    # Left: labels
    draw.text((fx1 + 20, fy1 + 14), "Cumulative image tokens",
              font=F_MED, fill=WHITE)
    draw.text((fx1 + 20, fy1 + 50), f"Full-frame:   {cum_ff:>6,}",
              font=F_MED, fill=RED)
    draw.text((fx1 + 20, fy1 + 82), f"DeltaVision:  {cum_dv:>6,}",
              font=F_MED, fill=GREEN)
    saved = cum_ff - cum_dv
    pct = (saved / cum_ff * 100) if cum_ff else 0
    draw.text((fx1 + 20, fy1 + 118), f"Saved:        {saved:>6,}  ({pct:.0f}%)",
              font=F_BIG, fill=YELLOW)

    # Right: visual bar
    bar_x1 = fx1 + 400
    bar_x2 = fx2 - 20
    bar_y = fy1 + 80
    bar_h = 44
    bar_width = bar_x2 - bar_x1

    # Red background = FF total (full width)
    box(draw, (bar_x1, bar_y, bar_x2, bar_y + bar_h), fill=(60, 25, 25), outline=RED)
    # Green fill = DV proportion
    dv_width = int(bar_width * (cum_dv / cum_ff)) if cum_ff else 0
    box(draw, (bar_x1, bar_y, bar_x1 + dv_width, bar_y + bar_h), fill=(30, 90, 50), outline=GREEN)

    draw.text(((bar_x1 + bar_x2) // 2, bar_y - 22),
              f"DeltaVision uses {pct if saved < 0 else 100 - pct:.0f}% of full-frame cost",
              font=F_SM, fill=WHITE, anchor="ma")

    draw.text((bar_x2, bar_y + bar_h + 10), f"across {meta['step']} steps so far",
              font=F_TINY, fill=DIM, anchor="ra")

    return canvas


def render_intro() -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((W // 2, 160), "Where's Waldo?", font=F_HUGE, fill=WHITE, anchor="mm")
    draw.text((W // 2, 220), "the GUI-agent observation problem",
              font=F_BIG, fill=CYAN, anchor="mm")

    lines = [
        "",
        "Standard computer-use agents get a full screenshot each step.",
        "~1600 tokens. Needle in a haystack.",
        "",
        "DeltaVision hands over the needle.",
        "~400 tokens. Pre-highlighted. Zoomed in.",
        "",
        "This is the same task, observed two ways, on the same page.",
        "",
        "Task: add 3 todos, check 1, filter Active on TodoMVC.",
        "8 steps. The browser runs headless.",
    ]
    y = 320
    for line in lines:
        if line.startswith("~1600"):
            draw.text((W // 2, y), line, font=F_MED, fill=RED, anchor="mm")
        elif line.startswith("~400"):
            draw.text((W // 2, y), line, font=F_MED, fill=GREEN, anchor="mm")
        elif line.startswith("Task:") or line.startswith("8 steps"):
            draw.text((W // 2, y), line, font=F_MED, fill=YELLOW, anchor="mm")
        else:
            draw.text((W // 2, y), line, font=F_MED, fill=WHITE, anchor="mm")
        y += 34

    return canvas


def render_outro(cum_ff: int, cum_dv: int, delta_steps: int, new_page_steps: int) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    saved = cum_ff - cum_dv
    pct = (saved / cum_ff * 100) if cum_ff else 0

    draw.text((W // 2, 140), "Totals", font=F_HUGE, fill=WHITE, anchor="mm")

    y = 250
    draw.text((W // 2, y), f"Full-frame baseline:    {cum_ff:,} tokens",
              font=F_BIG, fill=RED, anchor="mm")
    y += 60
    draw.text((W // 2, y), f"DeltaVision actual:     {cum_dv:,} tokens",
              font=F_BIG, fill=GREEN, anchor="mm")
    y += 80
    draw.text((W // 2, y), f"->  {pct:.0f}% fewer tokens  ({saved:,} saved)",
              font=F_HUGE, fill=YELLOW, anchor="mm")
    y += 100

    notes = [
        f"{delta_steps} DELTA steps (pre-highlighted changes, ~420-600 tokens each)",
        f"{new_page_steps} NEW_PAGE step (full-frame — DV and FF pay equal cost)",
        "",
        "The only step where DV and FF cost the same is when it genuinely matters:",
        "the page structure changed enough to warrant re-grounding.",
    ]
    for line in notes:
        color = WHITE if line else WHITE
        if line.startswith(("DELTA", "NEW")) or "tokens each" in line:
            color = GREEN if "DELTA" in line else ORANGE
        draw.text((W // 2, y), line, font=F_MED, fill=color, anchor="mm")
        y += 32

    return canvas


# ============================================================= main

def main():
    demo_dir = Path("benchmarks/ablation/waldo_demo")
    steps = sorted(demo_dir.glob("step_*"))
    if not steps:
        print(f"No step directories in {demo_dir}/ — run record_waldo_demo.py first.")
        return

    frames = []
    intro_np = np.array(render_intro())
    for _ in range(int(INTRO_HOLD * FPS)):
        frames.append(intro_np)

    cum_ff = 0
    cum_dv = 0
    delta_steps = 0
    new_page_steps = 0
    rendered_frames = []

    total_steps = len(steps) - 1  # exclude step 0

    for step_dir in steps:
        meta = json.loads((step_dir / "meta.json").read_text())
        cum_ff += meta["ff_tokens"]
        cum_dv += meta["dv_tokens"]
        if meta.get("transition") == "new_page" or meta["step"] == 0:
            if meta["step"] > 0:
                new_page_steps += 1
        else:
            delta_steps += 1

        ff_img = Image.open(step_dir / "ff_fullpage.png")
        dv_thumb = Image.open(step_dir / "dv_thumb.png")
        dv_crops = [
            Image.open(step_dir / f"dv_crop_{i}.png")
            for i in range(2)
            if (step_dir / f"dv_crop_{i}.png").exists()
        ]

        img = render_step(meta, ff_img, dv_thumb, dv_crops,
                          cum_ff=cum_ff, cum_dv=cum_dv, total_steps=total_steps)
        rendered_frames.append(img)

    # Hold each step; fade between
    for i, img in enumerate(rendered_frames):
        img_np = np.array(img)
        for _ in range(int(HOLD_PER_STEP * FPS)):
            frames.append(img_np)
        if i < len(rendered_frames) - 1:
            nxt_np = np.array(rendered_frames[i + 1])
            for f in range(FADE_FRAMES):
                t = (f + 1) / (FADE_FRAMES + 1)
                blended = (img_np * (1 - t) + nxt_np * t).astype(np.uint8)
                frames.append(blended)

    outro_np = np.array(render_outro(cum_ff, cum_dv, delta_steps, new_page_steps))
    for _ in range(int(OUTRO_HOLD * FPS)):
        frames.append(outro_np)

    out_path = Path("benchmarks/ablation/video_frames/waldo_comparison.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(frames)} frames at {FPS}fps...")
    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(
        str(out_path),
        codec="libx264",
        audio=False,
        ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
    )
    print(f"\nVideo saved to {out_path}")
    print(f"Duration: {len(frames) / FPS:.1f}s")
    print(f"Totals: FF={cum_ff:,} tokens, DV={cum_dv:,} tokens, savings={(cum_ff-cum_dv)/cum_ff*100:.0f}%")


if __name__ == "__main__":
    main()
