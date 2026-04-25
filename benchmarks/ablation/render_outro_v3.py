#!/usr/bin/env python3
"""
Render a single outro frame matching the v1 aesthetic but with run_14's numbers.
40.5% savings · 62,790 FF tokens (46 steps) · 37,370 DV tokens (11 full + 35 deltas)

Output: /tmp/outro_v3/outro.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = "/tmp/outro_v3/outro.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

W, H = 1920, 1080
BG = (0, 0, 0)
CYAN = (0, 229, 199)
RED = (220, 90, 100)
FG = (220, 220, 220)
DIM = (100, 100, 100)
CARD_GREEN_BG = (8, 22, 20)
CARD_RED_BG = (22, 10, 12)


def font(sz, bold=False):
    paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz, index=1 if bold else 0)
            except Exception:
                try:
                    return ImageFont.truetype(p, sz)
                except Exception:
                    pass
    return ImageFont.load_default()


def mono(sz):
    for p in ["/System/Library/Fonts/Menlo.ttc",
              "/System/Library/Fonts/Supplemental/Menlo.ttc"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # RESULT label
    f_label = mono(16)
    d.text((W // 2, 175), "RESULT", font=f_label, fill=CYAN, anchor="mm")

    # Big 40.5%
    f_hero = font(200, bold=True)
    d.text((W // 2, 320), "40.5%", font=f_hero, fill=CYAN, anchor="mm")

    # Subtitle
    f_sub = font(42)
    d.text((W // 2, 440), "fewer image tokens", font=f_sub, fill=FG, anchor="mm")

    # Two cards: FULL FRAME (red) and DELTAVISION (green)
    card_w, card_h = 420, 170
    card_y = 560
    arrow_gap = 120  # space between cards for the arrow
    cx_ff = W // 2 - arrow_gap // 2 - card_w
    cx_dv = W // 2 + arrow_gap // 2

    # FF card
    d.rounded_rectangle([cx_ff, card_y, cx_ff + card_w, card_y + card_h],
                        radius=10, fill=CARD_RED_BG, outline=RED, width=2)
    d.text((cx_ff + card_w // 2, card_y + 28),
           "FULL FRAME", font=mono(16), fill=RED, anchor="mm")
    d.text((cx_ff + card_w // 2, card_y + 80),
           "62,790", font=font(70, bold=True), fill=RED, anchor="mm")
    d.text((cx_ff + card_w // 2, card_y + 140),
           "tokens · 46 steps · SF", font=mono(16), fill=(180, 90, 100), anchor="mm")

    # Arrow
    arrow_y = card_y + card_h // 2
    arrow_x_start = cx_ff + card_w + 16
    arrow_x_end = cx_dv - 16
    d.line([(arrow_x_start, arrow_y), (arrow_x_end - 10, arrow_y)], fill=FG, width=2)
    d.polygon([(arrow_x_end - 10, arrow_y - 8), (arrow_x_end - 10, arrow_y + 8),
               (arrow_x_end + 2, arrow_y)], fill=FG)

    # DV card
    d.rounded_rectangle([cx_dv, card_y, cx_dv + card_w, card_y + card_h],
                        radius=10, fill=CARD_GREEN_BG, outline=CYAN, width=2)
    d.text((cx_dv + card_w // 2, card_y + 28),
           "DELTAVISION", font=mono(16), fill=CYAN, anchor="mm")
    d.text((cx_dv + card_w // 2, card_y + 80),
           "37,370", font=font(70, bold=True), fill=CYAN, anchor="mm")
    d.text((cx_dv + card_w // 2, card_y + 140),
           "tokens · 11 full + 35 deltas", font=mono(16), fill=(100, 190, 180), anchor="mm")

    # Pipeline line
    pipeline_y = card_y + card_h + 60
    d.text((W // 2, pipeline_y),
           "Browser → DeltaVision CV pipeline → delta observation → model → action",
           font=mono(20), fill=FG, anchor="mm")
    d.text((W // 2, pipeline_y + 30),
           "Zero LLM calls in the classifier · works with Claude, GPT-4o, Qwen, Ollama",
           font=mono(14), fill=DIM, anchor="mm")

    # Install pill
    pill_y = pipeline_y + 90
    pill_w = 420
    pill_h = 60
    pill_x = W // 2 - pill_w // 2
    d.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                        radius=8, outline=CYAN, width=2)
    d.text((W // 2, pill_y + pill_h // 2),
           "pip install deltavision", font=mono(26), fill=CYAN, anchor="mm")
    d.text((W // 2, pill_y + pill_h + 24),
           "github.com/ddavidgao/deltavision", font=mono(14), fill=DIM, anchor="mm")

    # Footer baseline info
    footer = ("FF baseline: same SF task · 46 screenshots · 62,790 tokens · "
              "DV run: SF · live MCP proxy · 40.5% savings (residual-first classifier, 2026-04-24)")
    d.text((W // 2, H - 28), footer, font=mono(12), fill=DIM, anchor="mm")

    img.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
