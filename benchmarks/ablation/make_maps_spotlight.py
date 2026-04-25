#!/usr/bin/env python3
"""
Generate spotlight frames for Maps screenshots using a fixed sidebar region.
The Maps sidebar is always the left ~30% — spotlight that, dim the rest.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

BORDER_COLOR = (0, 229, 199)
BORDER_WIDTH = 4
DIM_ALPHA = 0.18
BLUR_RADIUS = 5
LABEL_FG = (0, 0, 0)

# Maps sidebar is roughly left 30% of screen
SIDEBAR_FRAC = 0.30

SCREENSHOTS = [
    ("step_00.png", "FULL FRAME  —  1,365 tokens", False),
    ("step_02.png", "DELTA: sidebar changed  —  ~400 tokens", True),
    ("step_04.png", "DELTA: new listing  —  ~400 tokens", True),
    ("step_06.png", "DELTA: sidebar updated  —  ~400 tokens", True),
    ("step_08.png", "DELTA: new listing  —  ~400 tokens", True),
    ("step_10.png", "DELTA: sidebar updated  —  ~400 tokens", True),
]

def spotlight(img_pil, use_sidebar, label):
    img = img_pil.convert("RGB")
    W, H = img.size

    if not use_sidebar:
        draw = ImageDraw.Draw(img)
        _draw_label(img, label, (20, 20))
        return img

    sw = int(W * SIDEBAR_FRAC)
    x, y, w, h = 0, 0, sw, H

    blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - DIM_ALPHA))))
    dimmed = Image.alpha_composite(blurred.convert("RGBA"), overlay).convert("RGB")

    sharp_crop = img.crop((x, y, x + w, y + h))
    dimmed.paste(sharp_crop, (x, y))

    draw = ImageDraw.Draw(dimmed)
    for i in range(BORDER_WIDTH):
        draw.rectangle([x - i, y - i, x + w + i, y + h + i], outline=BORDER_COLOR)

    _draw_label(dimmed, label, (8, 8))
    return dimmed

def _draw_label(img, text, pos):
    draw = ImageDraw.Draw(img)
    x, y = pos
    bbox = draw.textbbox((x + 6, y + 4), text)
    pad = 6
    draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=BORDER_COLOR)
    draw.text((x + 6, y + 4), text, fill=LABEL_FG)

def main():
    run_dir = "benchmarks/mapsheets/results/run_02/screenshots"
    out_dir = "/tmp/dv_maps_spotlight"
    os.makedirs(out_dir, exist_ok=True)

    out_paths = []
    for i, (fname, label, use_sidebar) in enumerate(SCREENSHOTS):
        src = os.path.join(run_dir, fname)
        img = Image.open(src)
        result = spotlight(img, use_sidebar, label)
        out = os.path.join(out_dir, f"frame_{i:02d}.png")
        result.save(out)
        out_paths.append(out)
        print(f"  {fname} → {out}")

    # Build video: 1.8s per frame
    fps = 24
    hold = int(1.8 * fps)
    list_path = os.path.join(out_dir, "frames.txt")
    with open(list_path, "w") as fh:
        for p in out_paths:
            for _ in range(hold):
                fh.write(f"file '{p}'\n")
                fh.write(f"duration {1/fps:.4f}\n")

    video_out = os.path.join(out_dir, "maps_spotlight.mp4")
    os.system(
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
        f'-vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" '
        f'-c:v libx264 -crf 18 -pix_fmt yuv420p "{video_out}"'
    )
    print(f"\nVideo: {video_out}")
    return video_out

if __name__ == "__main__":
    main()
