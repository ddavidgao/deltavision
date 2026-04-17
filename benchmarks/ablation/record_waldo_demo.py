"""
"Where's Waldo" demo data capture.

Drives Playwright through a TodoMVC action sequence (headless — invisible to
the user) and saves BOTH observation strategies for every step:

  - FF observation:  full page screenshot (what a standard CU agent sees)
  - DV observation:  thumbnail-with-green-box + cropped detail regions

Same actions, same page state, two different observation packages per step.
The render script then builds a side-by-side comparison video.

Output:
  benchmarks/ablation/waldo_demo/
    step_NN/
      ff_fullpage.png         <- what full-frame mode sends
      dv_thumb.png            <- what DV sends (thumbnail)
      dv_crop_0.png           <- DV detail crop 0
      dv_crop_1.png           <- DV detail crop 1
      meta.json               <- classifier output + action taken
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image, ImageDraw
from playwright.async_api import async_playwright

from config import DeltaVisionConfig
from vision.capture import capture_screenshot
from vision.diff import compute_diff, extract_crops
from vision.classifier import classify_transition, extract_anchor, TransitionType


# A scripted sequence that exercises DELTA and NEW_PAGE paths on TodoMVC.
SEQUENCE = [
    {"action_type": "type", "text": "buy groceries", "label": "Type todo 1"},
    {"action_type": "key", "key": "Enter", "label": "Submit todo 1"},
    {"action_type": "type", "text": "write report", "label": "Type todo 2"},
    {"action_type": "key", "key": "Enter", "label": "Submit todo 2"},
    {"action_type": "type", "text": "review PR", "label": "Type todo 3"},
    {"action_type": "key", "key": "Enter", "label": "Submit todo 3"},
    # now click the first todo's checkbox (DOM selector)
    {"action_type": "click_selector", "selector": "ul.todo-list li:nth-child(1) input.toggle", "label": "Check todo 1"},
    # click the Active filter
    {"action_type": "click_selector", "selector": "ul.filters li:nth-child(2) a", "label": "Filter: Active"},
]


async def run(out_dir: Path, url: str = "https://todomvc.com/examples/react/dist/"):
    config = DeltaVisionConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,
            args=[f"--window-size={config.BROWSER_WIDTH},{config.BROWSER_HEIGHT}"])
        context = await browser.new_context(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )
        page = await context.new_page()

        print(f"Navigating to {url}")
        await page.goto(url)
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            await page.wait_for_load_state("domcontentloaded")

        # Focus the "What needs to be done?" input
        await page.locator(".new-todo").click()

        t0 = await capture_screenshot(page)
        anchor = extract_anchor(t0, config)
        url_t0 = page.url

        # Save step 0 (initial)
        step_dir = out_dir / "step_00"
        step_dir.mkdir(exist_ok=True)
        t0.save(step_dir / "ff_fullpage.png")
        # Initial step is full_frame for both
        (step_dir / "meta.json").write_text(json.dumps({
            "step": 0,
            "action_label": "navigate + focus input",
            "obs_type": "full_frame",
            "trigger": "initial",
            "diff_ratio": 0.0,
            "phash_distance": 0,
            "anchor_score": 1.0,
            "ff_tokens": 1600,
            "dv_tokens": 1600,
            "url": url_t0,
        }, indent=2))
        # The "DV view" on initial step is the same full page
        t0.save(step_dir / "dv_thumb.png")

        print(f"[step 0] initial  {t0.width}x{t0.height}")

        for i, step in enumerate(SEQUENCE, start=1):
            step_dir = out_dir / f"step_{i:02d}"
            step_dir.mkdir(exist_ok=True)

            # Execute the action
            atype = step["action_type"]
            label = step["label"]
            if atype == "type":
                await page.keyboard.type(step["text"], delay=30)
            elif atype == "key":
                await page.keyboard.press(step["key"])
            elif atype == "click_selector":
                await page.locator(step["selector"]).click()

            # Wait for page to settle
            await asyncio.sleep(0.5)

            t1 = await capture_screenshot(page)
            url_after = page.url
            diff = compute_diff(t0, t1, config)
            cls = classify_transition(
                t0=t0, t1=t1,
                url_before=url_t0, url_after=url_after,
                anchor_template=anchor, config=config,
                diff_result=diff, last_action_type=atype,
            )

            crops = extract_crops(t0, t1, diff.changed_bboxes, config.CROP_PADDING)

            # Save FF view — always the full page screenshot
            t1.save(step_dir / "ff_fullpage.png")

            # Build DV view: thumbnail with green boxes
            if cls.transition == TransitionType.NEW_PAGE:
                # DV sends full page on NEW_PAGE too, no green boxes needed
                t1.save(step_dir / "dv_thumb.png")
                dv_tokens = 1600
            else:
                thumb = t1.resize((320, 225), Image.LANCZOS)
                draw = ImageDraw.Draw(thumb)
                sx = 320 / t1.width
                sy = 225 / t1.height
                for c in crops:
                    x, y, w, h = c["bbox"]
                    draw.rectangle([
                        (int(x * sx) - 1, int(y * sy) - 1),
                        (int((x + w) * sx) + 1, int((y + h) * sy) + 1),
                    ], outline=(0, 255, 0), width=2)
                thumb.save(step_dir / "dv_thumb.png")
                # Save up to 2 crops for detail panels
                for ci, c in enumerate(crops[:2]):
                    c["crop_after"].save(step_dir / f"dv_crop_{ci}.png")
                dv_tokens = 240 + len(crops[:2]) * 180  # ~thumb + crops

            (step_dir / "meta.json").write_text(json.dumps({
                "step": i,
                "action_label": label,
                "obs_type": "full_frame" if cls.transition == TransitionType.NEW_PAGE else "delta",
                "trigger": cls.trigger,
                "transition": cls.transition.value,
                "diff_ratio": cls.diff_ratio,
                "phash_distance": cls.phash_distance,
                "anchor_score": cls.anchor_score,
                "num_crops": len(crops),
                "ff_tokens": 1600,
                "dv_tokens": dv_tokens,
                "url": url_after,
            }, indent=2))

            print(f"[step {i}] {label:<30s}  {cls.transition.value:<9s}  "
                  f"diff={cls.diff_ratio:.3f}  phash={cls.phash_distance:2d}  "
                  f"crops={len(crops)}  ff=1600 dv={dv_tokens}")

            # Re-anchor on NEW_PAGE (loop invariant)
            if cls.transition == TransitionType.NEW_PAGE:
                t0 = t1
                url_t0 = url_after
                anchor = extract_anchor(t0, config)

        await browser.close()

    print(f"\nArtifacts saved to {out_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("benchmarks/ablation/waldo_demo"))
    p.add_argument("--url", default="https://todomvc.com/examples/react/dist/")
    args = p.parse_args()

    asyncio.run(run(args.out, args.url))


if __name__ == "__main__":
    main()
