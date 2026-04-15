"""
Capture concrete visual frames from diverse sites.
Saves t0, t1, diff image, and cropped regions as PNGs for review.

Output: benchmarks/generalization/frames/<site>/<scenario>/
  - t0.png          (before)
  - t1.png          (after)
  - diff.png        (binary diff mask)
  - crop_N_before.png / crop_N_after.png
  - meta.json       (classifier output, all metrics)
"""

import asyncio
import time
import json
from pathlib import Path
from PIL import Image
from io import BytesIO

from playwright.async_api import async_playwright

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DeltaVisionConfig
from vision.classifier import classify_transition, extract_anchor
from vision.diff import compute_diff, extract_crops
from vision.phash import compute_phash, hamming_distance


SCENARIOS = [
    # Easy: URL change (Layer 1)
    {
        "site": "wikipedia",
        "name": "nav_article_to_article",
        "start_url": "https://en.wikipedia.org/wiki/Computer_vision",
        "action": "navigate",
        "target_url": "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
        "settle_ms": 2000,
    },
    # Medium: SPA small content change (should be DELTA)
    {
        "site": "todomvc",
        "name": "spa_add_items",
        "start_url": "https://todomvc.com/examples/react/dist/",
        "action": "js",
        "js": """
            const input = document.querySelector('.new-todo') || document.querySelector('input');
            if (input) {
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value').set;
                for (const t of ['Buy groceries', 'Study ML', 'Read paper']) {
                    setter.call(input, t);
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13, bubbles: true }));
                }
            }
        """,
        "settle_ms": 1000,
    },
    # Hard: full page content replacement via JS (Layer 3 - pHash)
    {
        "site": "dynamic_spa",
        "name": "full_content_replacement",
        "start_url": "https://example.com",
        "action": "js",
        "js": """
            document.body.innerHTML = '<div style="background:#1a1a2e;color:#e0e0e0;padding:40px;min-height:100vh">' +
                '<h1 style="font-size:48px">Dashboard</h1>' +
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:30px">' +
                '<div style="background:#16213e;padding:30px;border-radius:12px"><h2>Users</h2><p style="font-size:36px">12,847</p></div>' +
                '<div style="background:#16213e;padding:30px;border-radius:12px"><h2>Revenue</h2><p style="font-size:36px">$84,291</p></div>' +
                '<div style="background:#16213e;padding:30px;border-radius:12px"><h2>Sessions</h2><p style="font-size:36px">45,102</p></div>' +
                '<div style="background:#16213e;padding:30px;border-radius:12px"><h2>Bounce Rate</h2><p style="font-size:36px">23.4%</p></div>' +
                '</div></div>';
        """,
        "settle_ms": 500,
    },
    # Scroll (should be DELTA with scroll bypass)
    {
        "site": "wikipedia",
        "name": "scroll_long_article",
        "start_url": "https://en.wikipedia.org/wiki/Machine_learning",
        "action": "scroll",
        "scroll_y": 2000,
        "settle_ms": 1000,
    },
    # Idle baseline (should be DELTA, diff ~ 0)
    {
        "site": "hackernews",
        "name": "idle_no_change",
        "start_url": "https://news.ycombinator.com",
        "action": "idle",
        "settle_ms": 500,
    },
    # HumanBenchmark reaction page (colored background, unique layout)
    {
        "site": "humanbenchmark",
        "name": "nav_tests",
        "start_url": "https://humanbenchmark.com",
        "action": "navigate",
        "target_url": "https://humanbenchmark.com/tests/reactiontime",
        "settle_ms": 2000,
    },
]


async def capture(page) -> Image.Image:
    png = await page.screenshot(type="png")
    return Image.open(BytesIO(png))


async def run_captures():
    config = DeltaVisionConfig()
    base_dir = Path(__file__).parent / "frames"
    base_dir.mkdir(exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})

        for sc in SCENARIOS:
            name = f"{sc['site']}_{sc['name']}"
            out_dir = base_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- {name} ---")

            # Navigate to start
            await page.goto(sc["start_url"], wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(sc.get("settle_ms", 1500) / 1000)

            # Capture t0
            t0 = await capture(page)
            url_before = page.url
            anchor = extract_anchor(t0, config)
            t0.save(out_dir / "t0.png")

            # Execute action
            action = sc.get("action")
            if action == "navigate":
                await page.goto(sc["target_url"], wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(sc.get("settle_ms", 1500) / 1000)
            elif action == "js":
                await page.evaluate(sc["js"])
                await asyncio.sleep(sc.get("settle_ms", 1500) / 1000)
            elif action == "scroll":
                await page.mouse.wheel(0, sc.get("scroll_y", 800))
                await asyncio.sleep(sc.get("settle_ms", 1000) / 1000)
            elif action == "idle":
                await asyncio.sleep(0.5)

            # Capture t1
            t1 = await capture(page)
            url_after = page.url
            t1.save(out_dir / "t1.png")

            # Compute diff
            diff_result = compute_diff(t0, t1, config)

            # Save diff image
            if diff_result.diff_image:
                diff_result.diff_image.save(out_dir / "diff.png")

            # Save crops
            crops = extract_crops(t0, t1, diff_result.changed_bboxes, config.CROP_PADDING)
            for i, crop in enumerate(crops[:6]):
                crop["crop_before"].save(out_dir / f"crop_{i}_before.png")
                crop["crop_after"].save(out_dir / f"crop_{i}_after.png")

            # Classify
            action_type = sc.get("action")
            result = classify_transition(
                t0, t1, url_before, url_after, anchor, config,
                last_action_type=action_type,
            )

            # Also compute raw pHash for the record
            h0 = compute_phash(t0)
            h1 = compute_phash(t1)
            phash_dist = hamming_distance(h0, h1)

            meta = {
                "site": sc["site"],
                "scenario": sc["name"],
                "action": action,
                "url_before": url_before,
                "url_after": url_after,
                "classification": result.transition.value,
                "trigger": result.trigger,
                "diff_ratio": round(diff_result.diff_ratio, 4),
                "phash_distance": phash_dist,
                "anchor_score": round(result.anchor_score, 4),
                "num_changed_regions": len(diff_result.changed_bboxes),
                "num_crops": len(crops),
                "bboxes": diff_result.changed_bboxes[:6],
            }

            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  Classification: {result.transition.value} (trigger={result.trigger})")
            print(f"  Diff ratio: {diff_result.diff_ratio:.3f}, pHash: {phash_dist}, "
                  f"Anchor: {result.anchor_score:.3f}")
            print(f"  Changed regions: {len(diff_result.changed_bboxes)}, Crops saved: {len(crops[:6])}")
            print(f"  Saved to: {out_dir}")

        await browser.close()

    print(f"\nAll frames saved to: {base_dir}")
    print(f"Open any folder to see: t0.png, t1.png, diff.png, crop_*_before/after.png, meta.json")


if __name__ == "__main__":
    asyncio.run(run_captures())
