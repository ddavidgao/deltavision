"""
DeltaVision reaction time benchmark — LIVE.

Proves the core thesis: for simple state changes, the CV pipeline
IS the agent. No LLM call needed. Diff → detect green → click.

Usage: python benchmarks/reaction/run_reaction.py [--rounds 5]
"""

import asyncio
import time
from io import BytesIO

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright


def get_dominant_color(img: Image.Image) -> str:
    """Classify the top banner color as blue/red/green/unknown."""
    arr = np.array(img.convert("RGB"))
    top = arr[30 : arr.shape[0] // 2]  # skip nav bar, check banner
    r, g, b = top.mean(axis=(0, 1))

    if g > 100 and g > r * 1.2 and g > b * 1.2:
        return "green"
    if r > 120 and r > g * 1.4 and r > b * 1.2:
        return "red"
    if b > 80 and (r + b) > g * 1.5:
        return "blue"
    return f"unknown(r={r:.0f},g={g:.0f},b={b:.0f})"


async def capture(page) -> Image.Image:
    """Fast screenshot."""
    png = await page.screenshot(type="jpeg", quality=50)  # jpeg is faster
    return Image.open(BytesIO(png))


async def run_reaction_test(rounds: int = 5, headless: bool = False):
    results = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto("https://humanbenchmark.com/tests/reactiontime")
        await asyncio.sleep(2)

        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1}/{rounds} ---")

            # State machine: click through blue/result screens until we reach red
            for attempt in range(10):
                await page.mouse.click(640, 300)
                await asyncio.sleep(0.3)
                frame = await capture(page)
                state = get_dominant_color(frame)
                if state == "red":
                    break
                print(f"  Setup: got {state}, clicking again... (attempt {attempt+1})")
            else:
                print("  Could not reach red state after 10 attempts, skipping round")
                continue

            print("  In red/waiting state")

            # TIGHT POLLING LOOP — the DeltaVision core
            poll_count = 0
            detection_to_click_ms = 0
            timed_out = False

            while True:
                t0 = time.perf_counter()
                try:
                    frame = await capture(page)
                except Exception as e:
                    print(f"  Capture error: {e}, retrying...")
                    await asyncio.sleep(0.5)
                    continue

                capture_ms = (time.perf_counter() - t0) * 1000
                poll_count += 1
                state = get_dominant_color(frame)

                if state == "green":
                    green_at = time.perf_counter()
                    await page.mouse.click(640, 300)
                    click_at = time.perf_counter()
                    detection_to_click_ms = (click_at - green_at) * 1000

                    print(f"  GREEN! Detection->click: {detection_to_click_ms:.1f}ms")
                    print(f"  Capture latency: {capture_ms:.0f}ms, polls: {poll_count}")
                    break

                # Safety: if we've been polling for 15s, something is wrong
                if poll_count > 500:
                    print(f"  Timed out after {poll_count} polls")
                    timed_out = True
                    break

                await asyncio.sleep(0.01)  # ~100fps target

            if timed_out:
                continue

            # Wait for result to render
            await asyncio.sleep(1.5)

            # Read result from page
            try:
                result_text = await page.evaluate("""
                    () => {
                        // Try multiple selectors for the result
                        for (const sel of ['h1', '[class*="result"] h1', '.e4g6rdk0 h1']) {
                            const el = document.querySelector(sel);
                            if (el && el.textContent.includes('ms')) return el.textContent;
                        }
                        // Fallback: find any text with "ms"
                        const all = document.querySelectorAll('h1, h2, [class*="view"] div');
                        for (const el of all) {
                            if (el.textContent.match(/^\\d+\\s*ms$/)) return el.textContent;
                        }
                        return document.title;
                    }
                """)
            except Exception:
                result_text = "unknown"

            print(f"  Site result: {result_text}")

            try:
                ms = int("".join(c for c in result_text if c.isdigit()))
                results.append({
                    "round": round_num + 1,
                    "site_ms": ms,
                    "detection_to_click_ms": detection_to_click_ms,
                    "capture_ms": capture_ms,
                    "polls": poll_count,
                })
            except (ValueError, UnboundLocalError):
                print(f"  Could not parse: {result_text}")

            # Brief pause before next round
            await asyncio.sleep(0.5)

        await browser.close()

    # Summary
    if results:
        site_times = [r["site_ms"] for r in results]
        detect_times = [r["detection_to_click_ms"] for r in results]
        capture_times = [r["capture_ms"] for r in results]
        print("\n" + "=" * 60)
        print("DELTAVISION REACTION TIME RESULTS")
        print("=" * 60)
        print(f"Rounds completed: {len(results)}")
        print("")
        print(f"Site-reported reaction times: {site_times}")
        print(f"  Average: {sum(site_times) / len(site_times):.0f}ms")
        print(f"  Best:    {min(site_times)}ms")
        print(f"  Worst:   {max(site_times)}ms")
        print("")
        print(f"Detection→click latency: {[f'{d:.1f}' for d in detect_times]}ms")
        print(f"  Average: {sum(detect_times) / len(detect_times):.1f}ms")
        print(f"Screenshot capture avg: {sum(capture_times) / len(capture_times):.0f}ms")
        print("")
        print("--- Comparison ---")
        print("Human median:        273ms")
        avg = sum(site_times) / len(site_times)
        print(f"DeltaVision:         {avg:.0f}ms")
        print("Claude standard CU:  13491ms")
        print(f"Speedup vs CU:       {13491 / avg:.0f}x")
        print(f"vs Human:            {avg / 273:.1f}x slower")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--headless", action="store_true", help="Run headless")
    args = p.parse_args()
    asyncio.run(run_reaction_test(args.rounds, headless=args.headless))
