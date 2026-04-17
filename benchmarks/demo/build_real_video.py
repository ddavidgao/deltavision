"""Build demo video from REAL model run artifacts."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (10, 10, 22)
GREEN = (75, 210, 75)
RED = (220, 70, 70)
YELLOW = (255, 210, 55)
WHITE = (255, 255, 255)
LIGHT = (200, 200, 215)
GRAY = (130, 130, 145)
DIM = (80, 80, 95)
CALLOUT_BG = (22, 28, 50)


# Cross-platform font lookup: macOS Helvetica, Windows Segoe UI, Linux DejaVu.
# Previously shipped Segoe TTFs inline (1.9MB). Now use system fonts with
# graceful fallback chain so the repo stays small.
_BOLD_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",     # macOS
    "/System/Library/Fonts/Helvetica.ttc",         # macOS fallback
    "C:/Windows/Fonts/segoeuib.ttf",               # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
]
_REG_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _b(sz):
    for p in _BOLD_PATHS:
        try: return ImageFont.truetype(p, sz)
        except: pass
    return ImageFont.load_default()


def _r(sz):
    for p in _REG_PATHS:
        try: return ImageFont.truetype(p, sz)
        except: pass
    return ImageFont.load_default()


def wrap(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) > max_w:
            if cur: lines.append(cur)
            cur = w
        else:
            cur = t
    if cur: lines.append(cur)
    return lines


def render(mode_label, mc, step, total, images, prompt, callout):
    frame = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(frame)

    # Top bar
    draw.rectangle([(0, 0), (W, 50)], fill=(6, 6, 14))
    draw.text((30, 11), mode_label, font=_b(26), fill=mc)
    draw.text((W // 2 - 50, 13), f"Step {step}", font=_b(22), fill=WHITE)
    draw.text((W - 350, 13), "Exact images sent to the model", font=_r(16), fill=DIM)

    if len(images) == 1:
        # Single image — show big
        img = images[0]
        max_w, max_h = 1400, 720
        ratio = min(max_w / img.width, max_h / img.height)
        rw, rh = int(img.width * ratio), int(img.height * ratio)
        resized = img.resize((rw, rh), Image.LANCZOS)
        ix = (W - rw) // 2
        frame.paste(resized, (ix, 60))
        draw.rectangle([(ix - 2, 58), (ix + rw + 1, 60 + rh + 1)], outline=mc, width=2)
        label = "Full screenshot" if "FULL" in prompt else "Image sent"
        draw.text((ix, 60 + rh + 5), label, font=_r(13), fill=DIM)

    elif len(images) >= 2:
        # Thumbnail + crop(s) — show side by side
        thumb = images[0]
        crop = images[1]

        # Thumbnail on left (scale up a bit for visibility)
        tw, th = 700, int(700 * thumb.height / thumb.width)
        thumb_r = thumb.resize((tw, th), Image.LANCZOS)
        frame.paste(thumb_r, (40, 65))
        draw.rectangle([(38, 63), (42 + tw, 67 + th)], outline=GREEN, width=2)
        draw.text((40, 67 + th + 5), "Low-res page overview (thumbnail)", font=_b(15), fill=GREEN)

        # Crop on right
        max_cw = W - tw - 120
        max_ch = 700
        ratio = min(max_cw / crop.width, max_ch / crop.height)
        cw, ch = int(crop.width * ratio), int(crop.height * ratio)
        crop_r = crop.resize((cw, ch), Image.LANCZOS)
        cx = tw + 80
        frame.paste(crop_r, (cx, 65))
        draw.rectangle([(cx - 2, 63), (cx + cw + 1, 67 + ch)], outline=YELLOW, width=2)
        draw.text((cx, 67 + ch + 5), "High-res crop of what changed", font=_b(15), fill=YELLOW)

        # Arrow between them
        ax = tw + 55
        ay = 65 + max(th, ch) // 2
        draw.polygon([(ax - 5, ay - 8), (ax + 7, ay), (ax - 5, ay + 8)], fill=WHITE)

    # Bottom: prompt + callout
    by = 810
    draw.rectangle([(0, by), (W, H)], fill=(15, 15, 30))

    # Prompt lines (left)
    plines = prompt.strip().split("\n")[:5]
    for j, pl in enumerate(plines):
        draw.text((30, by + 10 + j * 18), pl[:100], font=_r(14), fill=GRAY)

    # Callout (right)
    cx = W // 2 + 30
    cw = W - cx - 20
    ch = H - by - 10
    draw.rectangle([(cx, by + 5), (cx + cw, by + ch)], fill=CALLOUT_BG, outline=(45, 50, 75))
    draw.text((cx + 12, by + 12), "What's happening:", font=_b(16), fill=YELLOW)
    for j, line in enumerate(wrap(draw, callout, _r(16), cw - 24)[:5]):
        draw.text((cx + 12, by + 35 + j * 21), line, font=_r(16), fill=LIGHT)

    # Progress
    draw.rectangle([(0, H - 4), (W, H)], fill=(25, 25, 45))
    if total > 0:
        draw.rectangle([(0, H - 4), (int(W * (step + 1) / total), H)], fill=mc)

    return frame


DV_CALLOUTS = [
    "The model sees the Wikipedia homepage. Full screenshot sent — this is the first step, so the model needs full context to understand the page layout.",
    "The model typed 'Alan Turing'. DeltaVision sends a LOW-RES THUMBNAIL of the page (left) showing where the change happened, plus a HIGH-RES CROP (right) of the autocomplete dropdown. The model gets spatial context AND detail, at a fraction of the token cost.",
    "The model clicked the first search suggestion. The Alan Turing article loaded — the entire page changed. DeltaVision detects this as a new page and sends a full screenshot.",
    "The model found the Bletchley Park link in the article and clicked it. Task complete in 4 steps.",
]

FF_CALLOUTS = [
    "Same starting point — full Wikipedia homepage screenshot.",
    "The model clicked the search bar but NOTHING HAPPENED. The screenshot looks identical to before. The model has no way to know its click didn't work — so it tries the same thing again. Notice: the search bar is still empty.",
    "After seeing the same page twice, the model finally tries typing instead. This wasted step is the difference — DeltaVision told the model 'your action had effect' on the first try.",
    "The model sees the autocomplete and clicks Alan Turing. Full screenshot — the model can compare this to the previous screenshot to see things changed.",
    "The model found the Bletchley Park link and clicked it. Task complete — but in 5 steps instead of 4, because of the wasted retry.",
]


def main():
    dv_dir = Path("benchmarks/demo/real_dv_final")
    ff_dir = Path("benchmarks/demo/real_ff_final")
    out = Path("benchmarks/demo/real_frames")
    (out / "dv").mkdir(parents=True, exist_ok=True)
    (out / "ff").mkdir(parents=True, exist_ok=True)

    # Clean
    for f in (out / "dv").glob("*.png"): f.unlink()
    for f in (out / "ff").glob("*.png"): f.unlink()

    # DeltaVision frames
    dv_total = 4
    for i in range(dv_total):
        images = []
        main_img = dv_dir / f"step_{i:03d}_sent_to_model.png"
        if main_img.exists():
            images.append(Image.open(main_img))
        crop_img = dv_dir / f"step_{i:03d}_sent_to_model_1.png"
        if crop_img.exists():
            images.append(Image.open(crop_img))

        prompt = ""
        prompt_f = dv_dir / f"step_{i:03d}_prompt.txt"
        if prompt_f.exists():
            prompt = prompt_f.read_text()

        callout = DV_CALLOUTS[i] if i < len(DV_CALLOUTS) else ""
        f = render("DeltaVision", GREEN, i, dv_total, images, prompt, callout)
        f.save(out / "dv" / f"step_{i:03d}.png")
        print(f"DV step {i}: {len(images)} images")

    # Full-frame frames
    ff_total = 5
    for i in range(ff_total):
        images = []
        main_img = ff_dir / f"step_{i:03d}_sent_to_model.png"
        if main_img.exists():
            images.append(Image.open(main_img))

        prompt = ""
        prompt_f = ff_dir / f"step_{i:03d}_prompt.txt"
        if prompt_f.exists():
            prompt = prompt_f.read_text()

        callout = FF_CALLOUTS[i] if i < len(FF_CALLOUTS) else ""
        f = render("Full-Frame Baseline", RED, i, ff_total, images, prompt, callout)
        f.save(out / "ff" / f"step_{i:03d}.png")
        print(f"FF step {i}: {len(images)} images")


if __name__ == "__main__":
    main()
