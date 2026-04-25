#!/usr/bin/env python3
"""
Side-by-side FF vs DV comparison video from parallel matched runs.

Layout (per frame):
  Top bar: "FF (Full Frame)  ·  DV (DeltaVision, residual-first + nudge)"
  Left:    FF screenshot (1280x800) + cumulative token counter underneath
  Right:   DV screenshot (1280x800) + cumulative token counter underneath
  Bottom:  Live savings bar with FF-DV gap visualization

Both runs are stretched to the same wall-clock duration so the video shows them
"racing" — the eye sees DV's running cost stay below FF's at all times.

Output: /tmp/sbs_v1/ffvsdv_sbs.mp4 (1920x1080, 24fps)
"""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
DV_DIR = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/screenshots"
FF_DIR = ROOT / "benchmarks/mapsheets/results/run_19_ff_parallel/screenshots"
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/dv_proxy_run_1777072627.jsonl"
FF_LOG = ROOT / "benchmarks/mapsheets/results/run_19_ff_parallel/dv_proxy_run_1777072642_ff.jsonl"
OUT_DIR = Path("/tmp/sbs_v1")
OUT_DIR.mkdir(exist_ok=True)
OUT_VIDEO = OUT_DIR / "ffvsdv_sbs.mp4"

# Layout
W, H = 1920, 1080
TOP_BAR_H = 60
PANEL_W = 920          # each side
PANEL_GAP = 20         # gap between panels
SHOT_H = 575           # each screenshot scaled to this height (1280x800 → 920x575)
COUNTER_H = 60         # token counter under each shot
BOTTOM_H = 80          # savings bar at the bottom

# Colors
BG = (8, 12, 16)
PANEL_BG = (12, 16, 20)
FG = (220, 220, 220)
DIM = (120, 120, 120)
CYAN = (0, 229, 199)
RED = (220, 90, 100)
PILL_GREEN = (20, 50, 46)
PILL_RED = (60, 28, 32)

FPS = 24
TOTAL_S = 30  # total side-by-side wall-clock duration


def font(sz, mono=True):
    paths = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
    ] if mono else [
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_steps(p):
    out = []
    for line in open(p):
        d = json.loads(line)
        if "step" in d:
            out.append(d)
    return out


def render_panel(draw_target, x_offset, screenshot_path, label, color, cum_tokens, step_num, total_steps):
    """Render one side's panel: label + screenshot + token counter."""
    # Panel background
    panel_x = x_offset
    panel_top = TOP_BAR_H + 10
    panel_bottom = panel_top + SHOT_H + COUNTER_H
    draw_target.rectangle([panel_x, panel_top, panel_x + PANEL_W, panel_bottom], fill=PANEL_BG)

    # Load + scale screenshot to PANEL_W x SHOT_H
    if screenshot_path and screenshot_path.exists():
        img = Image.open(screenshot_path).convert("RGB")
        # Scale to fit panel
        img = img.resize((PANEL_W, SHOT_H), Image.LANCZOS)
        return img, panel_x, panel_top
    return None, None, None


def draw_top_bar(draw, frame_w):
    """The thin top label bar."""
    draw.rectangle([0, 0, frame_w, TOP_BAR_H], fill=PANEL_BG)
    f_label = font(20)
    f_sub = font(13)
    # Two side-by-side titles
    left_x = 30
    draw.text((left_x, 12), "FULL FRAME (FF)", font=f_label, fill=RED)
    draw.text((left_x, 36), "1,365 tokens per screenshot · always", font=f_sub, fill=DIM)

    right_x = frame_w // 2 + 30
    draw.text((right_x, 12), "DELTAVISION (DV)", font=f_label, fill=CYAN)
    draw.text((right_x, 36),
              "delta-first · residual-first warp · token-capped · no-change nudge",
              font=f_sub, fill=DIM)

    # Vertical divider
    draw.line([(frame_w // 2, 0), (frame_w // 2, H)], fill=DIM, width=1)


def draw_counter(draw, panel_x, panel_top, label, value_str, color, step_str):
    """Token counter strip below each screenshot."""
    counter_y = panel_top + SHOT_H
    draw.rectangle([panel_x, counter_y, panel_x + PANEL_W, counter_y + COUNTER_H], fill=PANEL_BG)
    f_label = font(13)
    f_value = font(28)
    f_step = font(12)
    draw.text((panel_x + 20, counter_y + 8), label, font=f_label, fill=DIM)
    draw.text((panel_x + 20, counter_y + 24), value_str, font=f_value, fill=color)
    # Step counter on the right
    draw.text((panel_x + PANEL_W - 130, counter_y + 8), "STEP", font=f_step, fill=DIM)
    draw.text((panel_x + PANEL_W - 130, counter_y + 22), step_str, font=f_value, fill=color)


def draw_bottom_bar(draw, ff_cum, dv_cum, frame_w):
    """The savings bar at the bottom."""
    bar_y = H - BOTTOM_H
    draw.rectangle([0, bar_y, frame_w, H], fill=PANEL_BG)
    f_label = font(13)
    f_big = font(40)

    # Left side: total savings %
    saving_pct = (1 - dv_cum / max(ff_cum, 1)) * 100
    draw.text((30, bar_y + 8), "LIVE SAVINGS", font=f_label, fill=DIM)
    draw.text((30, bar_y + 22), f"{saving_pct:>4.1f}%", font=f_big, fill=CYAN)

    # Center: bar visualization
    bar_x = 280
    bar_w = frame_w - 280 - 280
    bar_h = 24
    bar_top_y = bar_y + 28
    # FF (red) full width
    draw.rectangle([bar_x, bar_top_y, bar_x + bar_w, bar_top_y + bar_h], fill=PILL_RED)
    draw.text((bar_x, bar_top_y - 16), "FF cost", font=f_label, fill=RED)
    # DV (cyan) proportional
    dv_w = int(bar_w * (dv_cum / max(ff_cum, 1)))
    draw.rectangle([bar_x, bar_top_y, bar_x + dv_w, bar_top_y + bar_h], fill=PILL_GREEN)
    draw.text((bar_x + dv_w + 8, bar_top_y + 4),
              "DV", font=f_label, fill=CYAN)

    # Right side: tokens saved
    saved = ff_cum - dv_cum
    draw.text((frame_w - 270, bar_y + 8), "TOKENS SAVED", font=f_label, fill=DIM)
    draw.text((frame_w - 270, bar_y + 22), f"{saved:,}", font=f_big, fill=CYAN)


def main():
    ff_steps = load_steps(FF_LOG)
    dv_steps = load_steps(DV_LOG)
    ff_files = sorted(os.listdir(FF_DIR))
    dv_files = sorted(os.listdir(DV_DIR))

    # Both should be same length on each side
    assert len(ff_steps) == len(ff_files), f"{len(ff_steps)} ff steps vs {len(ff_files)} files"
    assert len(dv_steps) == len(dv_files), f"{len(dv_steps)} dv steps vs {len(dv_files)} files"

    # Total frames at 24fps × TOTAL_S
    total_frames = FPS * TOTAL_S

    # Both runs map onto the same timeline — each side stretches to fill it
    ff_per_step = total_frames / len(ff_steps)
    dv_per_step = total_frames / len(dv_steps)

    print(f"FF: {len(ff_steps)} steps over {total_frames} frames ({ff_per_step:.2f} frames/step)")
    print(f"DV: {len(dv_steps)} steps over {total_frames} frames ({dv_per_step:.2f} frames/step)")

    panel_left_x = (W - 2 * PANEL_W - PANEL_GAP) // 2
    panel_right_x = panel_left_x + PANEL_W + PANEL_GAP

    frame_paths = []
    for fi in range(total_frames):
        # Which step is each side currently displaying?
        ff_idx = min(int(fi / ff_per_step), len(ff_steps) - 1)
        dv_idx = min(int(fi / dv_per_step), len(dv_steps) - 1)

        ff_step = ff_steps[ff_idx]
        dv_step = dv_steps[dv_idx]
        ff_file = FF_DIR / ff_files[ff_idx]
        dv_file = DV_DIR / dv_files[dv_idx]

        # Build canvas
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)

        # Top bar
        draw_top_bar(draw, W)

        # FF panel (left)
        ff_img, ff_x, ff_top = render_panel(draw, panel_left_x, ff_file, "FF", RED,
                                             ff_step["ff_cumulative"], ff_idx + 1, len(ff_steps))
        if ff_img:
            canvas.paste(ff_img, (ff_x, ff_top))
            draw_counter(draw, ff_x, ff_top, "FF CUMULATIVE TOKENS",
                         f'{ff_step["ff_cumulative"]:>7,}', RED,
                         f"{ff_idx + 1:>2}/{len(ff_steps):<2}")

        # DV panel (right)
        dv_img, dv_x, dv_top = render_panel(draw, panel_right_x, dv_file, "DV", CYAN,
                                             dv_step["dv_cumulative"], dv_idx + 1, len(dv_steps))
        if dv_img:
            canvas.paste(dv_img, (dv_x, dv_top))
            draw_counter(draw, dv_x, dv_top, "DV CUMULATIVE TOKENS",
                         f'{dv_step["dv_cumulative"]:>7,}', CYAN,
                         f"{dv_idx + 1:>2}/{len(dv_steps):<2}")

        # Bottom: live savings (use both sides' current cumulative for the bar)
        # Use FF's actual cumulative as the "what FF paid up to this point" reference
        # Use DV's actual cumulative as the "what DV paid"
        # Both are scaled to what the agent has consumed so far on that side
        # For an honest visual, normalize both to the same "fraction-through-task" point
        # so we're comparing tokens-spent-at-same-progress
        ff_progress = (ff_idx + 1) / len(ff_steps)
        dv_progress = (dv_idx + 1) / len(dv_steps)
        # Use the lesser progress so the comparison is fair at any point
        use_progress = min(ff_progress, dv_progress)
        ff_at_progress = ff_steps[min(int(use_progress * len(ff_steps)), len(ff_steps) - 1)]["ff_cumulative"]
        dv_at_progress = dv_steps[min(int(use_progress * len(dv_steps)), len(dv_steps) - 1)]["dv_cumulative"]
        draw_bottom_bar(draw, ff_at_progress, dv_at_progress, W)

        out_path = OUT_DIR / f"frame_{fi:04d}.png"
        canvas.save(out_path)
        frame_paths.append(str(out_path))
        if (fi + 1) % 60 == 0 or fi == total_frames - 1:
            print(f"  rendered {fi + 1}/{total_frames}")

    # Build video via concat demuxer
    list_path = OUT_DIR / "frames.txt"
    with open(list_path, "w") as f:
        for p in frame_paths:
            f.write(f"file '{p}'\nduration {1/FPS:.5f}\n")
        # ffmpeg concat demuxer needs final entry without duration
        f.write(f"file '{frame_paths[-1]}'\n")

    print("Encoding video...")
    cmd = (
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
        f'-vf "fps={FPS}" -c:v libx264 -crf 18 -pix_fmt yuv420p "{OUT_VIDEO}" 2>&1'
    )
    ret = os.system(cmd + " | tail -3")
    if ret == 0:
        print(f"\nVideo: {OUT_VIDEO}")
    else:
        print(f"ffmpeg returned non-zero: {ret}")


if __name__ == "__main__":
    main()
