"""
Classifier generalization test — run the 4-layer cascade on REAL screenshots
from diverse sites. Catches overfitting to McGraw-Hill fixtures.

Captures before/after pairs for 3 transition types per site:
  1. Navigation (URL change) — should trigger Layer 1
  2. Content change (same URL) — should trigger Layer 2 or 3
  3. No change (idle) — should classify as DELTA

Sites tested: Wikipedia, HumanBenchmark, Hacker News (static -> SPA -> minimal)
"""

import asyncio
import time
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from dataclasses import dataclass, field, asdict

from playwright.async_api import async_playwright

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DeltaVisionConfig
from vision.classifier import classify_transition, extract_anchor, TransitionType
from vision.diff import compute_diff
from vision.phash import compute_phash, hamming_distance
from results.store import ResultStore


@dataclass
class CaptureResult:
    site: str
    transition: str  # "navigation", "content_change", "idle"
    expected: str     # "new_page" or "delta"
    actual: str       # classified result
    correct: bool
    trigger: str
    diff_ratio: float
    phash_distance: int
    anchor_score: float
    capture_ms: float
    classify_ms: float
    url_before: str = ""
    url_after: str = ""


async def capture(page) -> Image.Image:
    png = await page.screenshot(type="png")
    return Image.open(BytesIO(png))


async def test_site(page, site_name: str, config: DeltaVisionConfig, scenarios: list) -> list[CaptureResult]:
    """Run classification scenarios on a single site."""
    results = []

    for scenario in scenarios:
        print(f"  [{site_name}] {scenario['name']}...")

        try:
            # Navigate to start URL
            await page.goto(scenario["start_url"], wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(scenario.get("settle_ms", 1500) / 1000)

            # Capture t0
            t0_cap_start = time.perf_counter()
            t0 = await capture(page)
            url_before = page.url
            anchor = extract_anchor(t0, config)

            # Execute the transition action
            action = scenario.get("action")
            if action == "navigate":
                await page.goto(scenario["target_url"], wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(scenario.get("settle_ms", 1500) / 1000)
            elif action == "click":
                await page.click(scenario["selector"], timeout=5000)
                await asyncio.sleep(scenario.get("settle_ms", 1500) / 1000)
            elif action == "idle":
                await asyncio.sleep(0.5)
            elif action == "type":
                await page.fill(scenario["selector"], scenario["text"])
                await asyncio.sleep(0.3)
                if scenario.get("submit"):
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(scenario.get("settle_ms", 1500) / 1000)
            elif action == "js":
                await page.evaluate(scenario["js"])
                await asyncio.sleep(scenario.get("settle_ms", 1500) / 1000)
            elif action == "scroll":
                await page.mouse.wheel(0, scenario.get("scroll_y", 800))
                await asyncio.sleep(scenario.get("settle_ms", 1000) / 1000)

            # Capture t1
            t1 = await capture(page)
            url_after = page.url
            capture_ms = (time.perf_counter() - t0_cap_start) * 1000

            # Run classifier (pass action type for scroll-awareness)
            action_type = scenario.get("action")
            cls_start = time.perf_counter()
            result = classify_transition(
                t0, t1, url_before, url_after, anchor, config,
                last_action_type=action_type,
            )
            classify_ms = (time.perf_counter() - cls_start) * 1000

            expected = scenario["expected"]
            actual = result.transition.value
            correct = actual == expected

            cr = CaptureResult(
                site=site_name,
                transition=scenario["name"],
                expected=expected,
                actual=actual,
                correct=correct,
                trigger=result.trigger,
                diff_ratio=round(result.diff_ratio, 4),
                phash_distance=result.phash_distance,
                anchor_score=round(result.anchor_score, 4),
                capture_ms=round(capture_ms, 1),
                classify_ms=round(classify_ms, 1),
                url_before=url_before,
                url_after=url_after,
            )
            results.append(cr)

            status = "PASS" if correct else "FAIL"
            print(f"    {status}: expected={expected}, got={actual} "
                  f"(trigger={result.trigger}, diff={result.diff_ratio:.3f}, "
                  f"phash={result.phash_distance}, anchor={result.anchor_score:.3f})")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(CaptureResult(
                site=site_name, transition=scenario["name"],
                expected=scenario["expected"], actual="error", correct=False,
                trigger=f"error:{e}", diff_ratio=0, phash_distance=0,
                anchor_score=0, capture_ms=0, classify_ms=0,
            ))

    return results


# -- Scenario definitions per site --

WIKIPEDIA_SCENARIOS = [
    {
        "name": "nav: main -> article",
        "start_url": "https://en.wikipedia.org/wiki/Main_Page",
        "action": "navigate",
        "target_url": "https://en.wikipedia.org/wiki/Computer_vision",
        "expected": "new_page",
    },
    {
        "name": "nav: article -> article",
        "start_url": "https://en.wikipedia.org/wiki/Computer_vision",
        "action": "navigate",
        "target_url": "https://en.wikipedia.org/wiki/Machine_learning",
        "expected": "new_page",
    },
    {
        "name": "idle: no interaction",
        "start_url": "https://en.wikipedia.org/wiki/Computer_vision",
        "action": "idle",
        "expected": "delta",
    },
    {
        "name": "search: type in search box",
        "start_url": "https://en.wikipedia.org/wiki/Main_Page",
        "action": "type",
        "selector": "input[name='search']",
        "text": "neural network",
        "submit": True,
        "expected": "new_page",
        "settle_ms": 2000,
    },
]

HUMANBENCHMARK_SCENARIOS = [
    {
        "name": "nav: home -> reaction test",
        "start_url": "https://humanbenchmark.com",
        "action": "navigate",
        "target_url": "https://humanbenchmark.com/tests/reactiontime",
        "expected": "new_page",
    },
    {
        "name": "nav: reaction -> aim test",
        "start_url": "https://humanbenchmark.com/tests/reactiontime",
        "action": "navigate",
        "target_url": "https://humanbenchmark.com/tests/aim",
        "expected": "new_page",
    },
    {
        "name": "idle: reaction page sits",
        "start_url": "https://humanbenchmark.com/tests/reactiontime",
        "action": "idle",
        "expected": "delta",
    },
]

HACKERNEWS_SCENARIOS = [
    {
        "name": "nav: front page -> comments",
        "start_url": "https://news.ycombinator.com",
        "action": "click",
        "selector": ".subline a:last-child",  # comments link
        "expected": "new_page",
        "settle_ms": 2000,
    },
    {
        "name": "idle: front page sits",
        "start_url": "https://news.ycombinator.com",
        "action": "idle",
        "expected": "delta",
    },
    {
        "name": "nav: page 1 -> page 2",
        "start_url": "https://news.ycombinator.com",
        "action": "click",
        "selector": "a.morelink",  # "More" link
        "expected": "new_page",
        "settle_ms": 2000,
    },
]

EXAMPLE_COM_SCENARIOS = [
    {
        "name": "idle: static page (absolute baseline)",
        "start_url": "https://example.com",
        "action": "idle",
        "expected": "delta",
        "settle_ms": 1000,
    },
]


# -- SPA / same-URL transitions (stress Layers 2-4) --

# TodoMVC is a classic SPA: URL hash changes but base URL stays same,
# and content changes are entirely in-page DOM manipulation.
TODOMVC_SCENARIOS = [
    {
        "name": "SPA: add todo items (same URL, content change)",
        "start_url": "https://todomvc.com/examples/react/dist/",
        "action": "js",
        "js": """
            const input = document.querySelector('.new-todo') || document.querySelector('input');
            if (input) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value').set;
                nativeInputValueSetter.call(input, 'Test item 1');
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13, bubbles: true }));
                nativeInputValueSetter.call(input, 'Test item 2');
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13, bubbles: true }));
                nativeInputValueSetter.call(input, 'Test item 3');
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13, bubbles: true }));
            }
        """,
        "expected": "delta",  # small content additions, not a full page change
        "settle_ms": 1000,
    },
]

# Wikipedia scroll -- large visual shift, same content
SCROLL_SCENARIOS = [
    {
        "name": "scroll: large scroll on long article (same page, big visual change)",
        "start_url": "https://en.wikipedia.org/wiki/Machine_learning",
        "action": "scroll",
        "scroll_y": 3000,
        "expected": "delta",  # scrolling is NOT a new page
        "settle_ms": 1000,
    },
    {
        "name": "scroll: small scroll (minimal change)",
        "start_url": "https://en.wikipedia.org/wiki/Machine_learning",
        "action": "scroll",
        "scroll_y": 200,
        "expected": "delta",
        "settle_ms": 500,
    },
]

# Tab switching within the same page (SPA pattern)
GITHUB_PUBLIC_SCENARIOS = [
    {
        "name": "SPA: switch tabs on repo (same base URL, content swap)",
        "start_url": "https://github.com/anthropics/anthropic-cookbook",
        "action": "click",
        "selector": "a[data-tab-item='i1issues-tab']",
        "expected": "new_page",  # GitHub uses pushState, URL changes
        "settle_ms": 2000,
    },
]

# Dynamic content injection (no URL change, no nav, just DOM update)
DYNAMIC_SCENARIOS = [
    {
        "name": "SPA: inject large content block via JS (simulated SPA nav)",
        "start_url": "https://example.com",
        "action": "js",
        "js": """
            document.body.innerHTML = '<h1>Completely New Content</h1>' +
                '<p>This is a simulated SPA navigation where the entire page ' +
                'content was replaced via JavaScript without a URL change.</p>' +
                '<div style=\"background:blue;height:400px;width:100%\"></div>';
        """,
        "expected": "new_page",  # full content replacement = new page
        "settle_ms": 500,
    },
    {
        "name": "SPA: small text update via JS (minor DOM change)",
        "start_url": "https://example.com",
        "action": "js",
        "js": "document.querySelector('p').textContent += ' [updated]';",
        "expected": "delta",  # tiny change = delta
        "settle_ms": 500,
    },
]


ALL_SITES = {
    "wikipedia": WIKIPEDIA_SCENARIOS,
    "humanbenchmark": HUMANBENCHMARK_SCENARIOS,
    "hackernews": HACKERNEWS_SCENARIOS,
    "example.com": EXAMPLE_COM_SCENARIOS,
    "scroll": SCROLL_SCENARIOS,
    "todomvc": TODOMVC_SCENARIOS,
    "github_pub": GITHUB_PUBLIC_SCENARIOS,
    "dynamic_spa": DYNAMIC_SCENARIOS,
}


async def run_generalization_test():
    config = DeltaVisionConfig()  # DEFAULT config, NOT McGraw-Hill preset
    all_results = []

    print("=" * 70)
    print("DELTAVISION CLASSIFIER GENERALIZATION TEST")
    print(f"Config: DEFAULT (diff={config.NEW_PAGE_DIFF_THRESHOLD}, "
          f"phash={config.PHASH_DISTANCE_THRESHOLD}, "
          f"anchor={config.ANCHOR_MATCH_THRESHOLD})")
    print("=" * 70)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})

        for site_name, scenarios in ALL_SITES.items():
            print(f"\n--- {site_name} ({len(scenarios)} scenarios) ---")
            results = await test_site(page, site_name, config, scenarios)
            all_results.extend(results)

        await browser.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(all_results)
    correct = sum(1 for r in all_results if r.correct)
    errors = sum(1 for r in all_results if r.actual == "error")

    print(f"Total: {total}, Correct: {correct}, Errors: {errors}, "
          f"Accuracy: {correct/(total-errors)*100:.0f}% (excl. errors)")

    # Per-site breakdown
    sites = set(r.site for r in all_results)
    for site in sorted(sites):
        site_results = [r for r in all_results if r.site == site]
        site_correct = sum(1 for r in site_results if r.correct)
        site_errors = sum(1 for r in site_results if r.actual == "error")
        valid = len(site_results) - site_errors
        acc = site_correct / valid * 100 if valid > 0 else 0
        print(f"  {site:<20} {site_correct}/{valid} correct ({acc:.0f}%)")

    # Detailed results table
    print(f"\n{'Site':<16} {'Scenario':<30} {'Expect':<10} {'Got':<10} {'Trigger':<14} "
          f"{'Diff':>6} {'pHash':>6} {'Anchor':>7} {'ms':>6}")
    print("-" * 115)
    for r in all_results:
        status = "OK" if r.correct else "MISS" if r.actual != "error" else "ERR"
        print(f"{r.site:<16} {r.transition:<30} {r.expected:<10} {r.actual:<10} "
              f"{r.trigger:<14} {r.diff_ratio:>6.3f} {r.phash_distance:>6} "
              f"{r.anchor_score:>7.3f} {r.classify_ms:>6.1f}  {status}")

    # Save to SQLite
    db = ResultStore()
    metrics = {
        "total_scenarios": total,
        "correct": correct,
        "errors": errors,
        "accuracy_pct": round(correct / (total - errors) * 100, 1) if total > errors else 0,
        "per_site": {},
    }
    for site in sorted(sites):
        sr = [r for r in all_results if r.site == site]
        sc = sum(1 for r in sr if r.correct)
        se = sum(1 for r in sr if r.actual == "error")
        metrics["per_site"][site] = {
            "correct": sc, "total": len(sr) - se,
            "accuracy_pct": round(sc / (len(sr) - se) * 100, 1) if len(sr) > se else 0,
        }

    db.save(
        benchmark="classifier_generalization",
        backend="cv_pipeline",
        metrics=metrics,
        config=asdict(config),
        transition_log=[asdict(r) for r in all_results],
        notes=f"Default config, {len(ALL_SITES)} sites, {total} scenarios",
    )
    db.close()

    # Also dump JSON
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nDetailed results: {out_path}")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_generalization_test())
