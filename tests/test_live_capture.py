"""
Live browser test — captures real screenshots and runs the full vision pipeline.
Requires: playwright install chromium

Run with: pytest tests/test_live_capture.py -v -s
"""

import pytest
from playwright.async_api import async_playwright

from config import DeltaVisionConfig
from vision.capture import capture_screenshot, get_current_url
from vision.classifier import TransitionType, classify_transition, extract_anchor
from vision.diff import compute_diff, extract_crops


@pytest.fixture
def config():
    return DeltaVisionConfig()


@pytest.mark.asyncio
async def test_capture_and_diff_identical(config):
    """Two captures of same static page should produce near-zero diff."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.set_content("<h1>Hello DeltaVision</h1><p>Static content</p>")
        await page.wait_for_load_state("networkidle")

        t0 = await capture_screenshot(page)
        t1 = await capture_screenshot(page)

        assert t0.size == (1280, 900)
        assert t1.size == (1280, 900)

        diff = compute_diff(t0, t1, config)
        assert diff.diff_ratio < 0.001
        assert not diff.action_had_effect

        await browser.close()


@pytest.mark.asyncio
async def test_click_causes_delta(config):
    """Clicking a button should produce a detectable delta."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.set_content("""
            <body style="margin:0;padding:50px;background:#222">
                <button id="btn"
                        onclick="this.style.background='lime';this.textContent='Clicked!'"
                        style="padding:20px 40px;font-size:24px;background:#666;color:white;border:none">
                    Click Me
                </button>
            </body>
        """)
        await page.wait_for_load_state("networkidle")

        t0 = await capture_screenshot(page)
        url_before = get_current_url(page)

        await page.click("#btn")
        await page.wait_for_timeout(300)

        t1 = await capture_screenshot(page)
        url_after = get_current_url(page)

        diff = compute_diff(t0, t1, config)
        assert diff.action_had_effect, f"Expected effect, got diff_ratio={diff.diff_ratio}"
        assert len(diff.changed_bboxes) >= 1

        anchor = extract_anchor(t0, config)
        cls = classify_transition(t0, t1, url_before, url_after, anchor, config, diff)
        assert cls.transition == TransitionType.DELTA

        crops = extract_crops(t0, t1, diff.changed_bboxes, config.CROP_PADDING)
        assert len(crops) >= 1
        assert crops[0]["change_magnitude"] > 0

        await browser.close()


@pytest.mark.asyncio
async def test_navigation_causes_new_page(config):
    """Navigating to a different URL should trigger NEW_PAGE."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})

        await page.set_content("<h1>Page A</h1>")
        await page.wait_for_load_state("networkidle")
        t0 = await capture_screenshot(page)
        url_before = get_current_url(page)
        anchor = extract_anchor(t0, config)

        await page.goto("data:text/html,<h1>Page B - Different</h1><p>New content</p>")
        await page.wait_for_load_state("networkidle")
        t1 = await capture_screenshot(page)
        url_after = get_current_url(page)

        cls = classify_transition(t0, t1, url_before, url_after, anchor, config)
        assert cls.transition == TransitionType.NEW_PAGE
        assert cls.trigger == "url_change"

        await browser.close()


@pytest.mark.asyncio
async def test_spa_content_swap_detected(config):
    """SPA-style content swap (same URL, big visual change) -> NEW_PAGE."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})

        # Start with page A content
        await page.set_content("""
            <body style="margin:0;background:#111">
                <div id="content" style="padding:50px;color:white;font-size:48px">
                    Page Content A - Lorem ipsum dolor sit amet consectetur adipiscing
                </div>
            </body>
        """)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(200)

        t0 = await capture_screenshot(page)
        url_before = get_current_url(page)
        anchor = extract_anchor(t0, config)

        # Swap content dramatically via JS (simulating SPA navigation)
        await page.evaluate("""
            document.body.style.background = '#009';
            document.getElementById('content').innerHTML =
                '<div style="background:#900;padding:100px;font-size:72px;color:yellow">' +
                'COMPLETELY DIFFERENT PAGE B - Navigation happened without URL change' +
                '</div>';
        """)
        await page.wait_for_timeout(300)

        t1 = await capture_screenshot(page)
        url_after = get_current_url(page)

        diff = compute_diff(t0, t1, config)
        cls = classify_transition(t0, t1, url_before, url_after, anchor, config, diff)

        assert url_before == url_after, "URL should not have changed"
        assert diff.diff_ratio > 0.1, f"Expected large diff, got {diff.diff_ratio}"
        assert cls.transition == TransitionType.NEW_PAGE
        assert cls.trigger in ("diff_ratio", "phash", "anchor_loss")

        await browser.close()
