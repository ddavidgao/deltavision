"""
Scripted pipeline visualization demo.

Drives the browser through a Wikipedia search while DeltaVision
captures and classifies every transition. Each step shows the browser,
what the model receives, and a plain English explanation.

Output: 1920x1080 step PNGs, stitched into video with ffmpeg.
"""

import asyncio
import time
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DeltaVisionConfig
from vision.capture import capture_screenshot
from vision.diff import compute_diff, extract_crops
from vision.classifier import classify_transition, extract_anchor, TransitionType
from vision.phash import compute_phash, hamming_distance

OUT_DIR = Path(__file__).parent / "pipeline_viz"
W, H = 1920, 1080
# Exact token costs measured from Claude Sonnet 4.6 API
FULL_FRAME_TOKENS = 2042  # measured: initial full frame
DELTA_TOKENS_AVG = 2075   # measured: delta step with annotated screenshot

BG = (10, 10, 22)
PANEL_BG = (18, 18, 35)
GREEN = (80, 215, 80)
RED = (225, 75, 75)
YELLOW = (255, 210, 60)
WHITE = (255, 255, 255)
LIGHT = (210, 210, 220)
GRAY = (140, 140, 155)
DIM = (85, 85, 100)
CALLOUT_BG = (25, 30, 55)


def _b(sz):
    """Bold font."""
    for p in ["benchmarks/demo/segoeuib.ttf", "benchmarks/demo/calibrib.ttf",
              "benchmarks/demo/arialbd.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _r(sz):
    """Regular font."""
    for p in ["benchmarks/demo/segoeui.ttf", "benchmarks/demo/calibri.ttf",
              "benchmarks/demo/arial.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def build_delta_panel(diff_result, crops, w=460, h=580):
    canvas = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 12), "What the model receives:", font=_r(14), fill=GRAY)
    draw.text((20, 32), "CHANGED REGIONS ONLY", font=_b(24), fill=GREEN)

    y = 75

    if diff_result and diff_result.diff_image:
        draw.text((20, y), "Highlighted pixels = what changed", font=_r(13), fill=GRAY)
        y += 18
        dw = w - 40
        dh = int(dw * diff_result.diff_image.height / max(diff_result.diff_image.width, 1))
        dh = min(dh, 200)
        thumb = diff_result.diff_image.resize((dw, dh), Image.LANCZOS)
        canvas.paste(thumb, (20, y))
        draw.rectangle([(19, y - 1), (21 + dw, y + dh)], outline=GREEN, width=2)
        y += dh + 15

    if crops:
        draw.text((20, y), f"{len(crops)} cropped region(s):", font=_r(13), fill=GRAY)
        y += 20
        for i, crop in enumerate(crops[:3]):
            if y > h - 70:
                break
            cw = (w - 60) // 2
            ch = min(80, int(cw * crop["crop_before"].height / max(crop["crop_before"].width, 1)))
            ch = max(ch, 25)

            draw.text((20, y), "BEFORE", font=_b(10), fill=DIM)
            draw.text((20 + cw + 20, y), "AFTER", font=_b(10), fill=YELLOW)
            y += 14
            before = crop["crop_before"].resize((cw, ch), Image.LANCZOS)
            after = crop["crop_after"].resize((cw, ch), Image.LANCZOS)
            canvas.paste(before, (20, y))
            canvas.paste(after, (20 + cw + 20, y))
            draw.rectangle([(19, y - 1), (21 + cw, y + ch)], outline=DIM, width=1)
            draw.rectangle([(19 + cw + 20, y - 1), (21 + cw * 2 + 20, y + ch)], outline=YELLOW, width=1)
            y += ch + 12

    draw.text((20, h - 35), f"{DELTA_TOKENS_AVG:,} tokens this step", font=_b(17), fill=GREEN)

    return canvas


def build_fullframe_panel(screenshot, w=460, h=580, is_baseline=False):
    canvas = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 12), "What the model receives:", font=_r(14), fill=GRAY)
    if is_baseline:
        draw.text((20, 32), "ENTIRE PAGE (every step)", font=_b(24), fill=RED)
    else:
        draw.text((20, 32), "FULL PAGE (new context)", font=_b(24), fill=RED)

    sw = w - 40
    sh = int(sw * screenshot.height / max(screenshot.width, 1))
    sh = min(sh, h - 140)
    thumb = screenshot.resize((sw, sh), Image.LANCZOS)
    canvas.paste(thumb, (20, 80))
    draw.rectangle([(19, 79), (21 + sw, 81 + sh)], outline=RED, width=2)

    draw.text((20, h - 35), f"{FULL_FRAME_TOKENS:,} tokens this step", font=_b(17), fill=RED)

    return canvas


def render_step(
    step_num, total_steps,
    browser_ss, model_panel,
    action_desc,       # what the agent did
    whats_happening,   # plain English callout for confused viewers
    token_count, is_deltavision,
    diff_ratio, phash_dist,
):
    frame = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(frame)

    # ── Top bar ──
    draw.rectangle([(0, 0), (W, 48)], fill=(6, 6, 14))
    mode_label = "DeltaVision" if is_deltavision else "Full-Frame Baseline"
    mode_color = GREEN if is_deltavision else RED
    draw.text((30, 10), mode_label, font=_b(26), fill=mode_color)
    draw.text((W // 2 - 50, 12), f"Step {step_num} of {total_steps}", font=_b(22), fill=WHITE)
    draw.text((W - 300, 12), f"Total tokens: {token_count:,}", font=_b(22), fill=YELLOW)

    # ── Browser panel (left) ──
    bx, by = 20, 60
    bw, bh = 1050, 580
    browser_resized = browser_ss.resize((bw, bh), Image.LANCZOS)
    frame.paste(browser_resized, (bx, by))
    draw.rectangle([(bx - 1, by - 1), (bx + bw, by + bh)], outline=(40, 40, 55), width=1)
    draw.text((bx + 5, by + 2), "BROWSER", font=_b(11), fill=DIM)

    # ── Arrow ──
    ax = bx + bw + 12
    ay = by + bh // 2
    for dy in range(-12, 13):
        dx = 12 - abs(dy) * 12 // 13
        draw.point((ax + dx, ay + dy), fill=GRAY)
    draw.polygon([(ax + 2, ay - 12), (ax + 14, ay), (ax + 2, ay + 12)], fill=GRAY)

    # ── Model panel (right) ──
    mx = ax + 24
    mp_resized = model_panel.resize((460, 580), Image.LANCZOS)
    frame.paste(mp_resized, (mx, by))
    draw.rectangle([(mx - 1, by - 1), (mx + 460, by + bh)], outline=(40, 40, 55), width=1)

    # ── Bottom section: action + "what's happening" callout ──
    bottom_y = by + bh + 15

    # Action description (left side)
    draw.text((30, bottom_y), action_desc, font=_r(22), fill=LIGHT)

    # Small metrics
    draw.text((30, bottom_y + 32), f"Pixel change: {diff_ratio:.1%}     Perceptual shift: {phash_dist}/64",
              font=_r(13), fill=DIM)

    # "What's happening" callout box (right side, stands out)
    cx = 750
    cw = W - cx - 30
    ch = 100
    # Rounded-ish box
    draw.rectangle([(cx, bottom_y - 5), (cx + cw, bottom_y + ch)], fill=CALLOUT_BG, outline=(50, 55, 85), width=1)

    draw.text((cx + 15, bottom_y + 5), "What's going on:", font=_b(16), fill=YELLOW)
    # Word wrap the callout text
    words = whats_happening.split()
    lines = []
    current = ""
    for word in words:
        test = current + " " + word if current else word
        bbox = draw.textbbox((0, 0), test, font=_r(17))
        if bbox[2] > cw - 40:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    for j, line in enumerate(lines[:3]):
        draw.text((cx + 15, bottom_y + 28 + j * 22), line, font=_r(17), fill=WHITE)

    # ── Progress bar ──
    draw.rectangle([(0, H - 4), (W, H)], fill=(25, 25, 45))
    if total_steps > 0:
        p = min(step_num / total_steps, 1.0)
        draw.rectangle([(0, H - 4), (int(W * p), H)], fill=mode_color)

    return frame


# ── Action sequences ──
# "scripted" = manually defined, "claude" = model-driven (needs API key)
SCRIPTED_ACTIONS = [
    {
        "action": "initial",
        "desc": "Wikipedia main page loaded",
        "callout": "Starting point. The model gets a full screenshot so it knows the page layout. Both modes start the same way — 2,042 tokens.",
    },
    {
        "action": "click", "x": 492, "y": 33,
        "desc": "The model clicks the search bar",
        "callout": "Barely anything changed on screen — just a cursor appeared in the search box. DeltaVision tells the model 'your click worked, the search bar region changed.' The baseline just sends the whole page again.",
    },
    {
        "action": "type", "text": "Alan Turing", "settle": 1.5,
        "desc": "The model types 'Alan Turing'",
        "callout": "Text appeared in the search bar and an autocomplete dropdown showed up. DeltaVision highlights those changed regions. Without this feedback, the model in baseline mode often doesn't realize it typed successfully.",
    },
    {
        "action": "click", "x": 364, "y": 79, "settle": 3.0,
        "desc": "The model clicks the first search suggestion",
        "callout": "The model sees the dropdown and clicks 'Alan Turing'. Wikipedia navigates to the article — the entire page changed. DeltaVision detects this and sends a full screenshot for the new context.",
    },
]


async def run_scripted(mode="deltavision"):
    is_dv = mode == "deltavision"
    config = DeltaVisionConfig()
    out = OUT_DIR / mode
    out.mkdir(parents=True, exist_ok=True)

    tokens = 0
    frames = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto("https://en.wikipedia.org")
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            await page.wait_for_load_state("domcontentloaded")

        t0 = await capture_screenshot(page)
        anchor = extract_anchor(t0, config)
        url_prev = page.url
        total = len(SCRIPTED_ACTIONS) - 1

        for i, step in enumerate(SCRIPTED_ACTIONS):
            if step["action"] == "initial":
                tokens += FULL_FRAME_TOKENS
                mp = build_fullframe_panel(t0)
                f = render_step(0, total, t0, mp, step["desc"], step["callout"],
                                tokens, is_dv, 0, 0)
                f.save(out / f"step_{i:03d}.png")
                frames.append(f)
                print(f"  Step {i}: initial")
                continue

            if step["action"] == "click":
                await page.mouse.click(step["x"], step["y"])
            elif step["action"] == "type":
                await page.keyboard.type(step["text"], delay=60)
            elif step["action"] == "key":
                await page.keyboard.press(step["key"])

            await asyncio.sleep(step.get("settle", 1.0))

            t1 = await capture_screenshot(page)
            url_now = page.url
            diff_result = compute_diff(t0, t1, config)
            cls = classify_transition(t0, t1, url_prev, url_now, anchor, config,
                                      diff_result=diff_result, last_action_type=step["action"])
            pdist = hamming_distance(compute_phash(t0), compute_phash(t1))

            if is_dv:
                if cls.transition == TransitionType.NEW_PAGE:
                    mp = build_fullframe_panel(t1)
                    tokens += FULL_FRAME_TOKENS
                else:
                    crops = extract_crops(t0, t1, diff_result.changed_bboxes, config.CROP_PADDING)
                    mp = build_delta_panel(diff_result, crops)
                    tokens += DELTA_TOKENS_AVG
            else:
                mp = build_fullframe_panel(t1, is_baseline=True)
                tokens += FULL_FRAME_TOKENS

            callout = step["callout"]
            if not is_dv:
                callout = "The model receives the entire page again, even though most of it looks exactly the same as before. Every unchanged pixel costs tokens."

            f = render_step(i, total, t1, mp, step["desc"], callout,
                            tokens, is_dv, diff_result.diff_ratio, pdist)
            f.save(out / f"step_{i:03d}.png")
            frames.append(f)
            print(f"  Step {i}: {cls.transition.value:<10} diff={diff_result.diff_ratio:.3f} phash={pdist} tokens={tokens:,}")

            if cls.transition == TransitionType.NEW_PAGE:
                t0 = t1
                anchor = extract_anchor(t0, config)
            url_prev = url_now

        await browser.close()

    print(f"  {mode}: {len(frames)} frames, {tokens:,} tokens")
    return frames, tokens


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["deltavision", "fullframe", "both"], default="both")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dv_tok = ff_tok = 0
    if args.mode in ("deltavision", "both"):
        print("=== DeltaVision ===")
        _, dv_tok = await run_scripted("deltavision")
    if args.mode in ("fullframe", "both"):
        print("\n=== Full-Frame ===")
        _, ff_tok = await run_scripted("fullframe")
    if args.mode == "both" and ff_tok > 0:
        print(f"\nDeltaVision: {dv_tok:,}  |  Full-frame: {ff_tok:,}  |  Savings: {(1 - dv_tok / ff_tok) * 100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
