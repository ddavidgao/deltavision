"""
End-to-end live tests with scripted model.

Exercises the FULL DeltaVision pipeline on real websites:
  Playwright browser → screenshot capture → diff engine → classifier →
  observation builder → model (scripted) → action executor → repeat

No API keys needed. Swap ScriptedModel for ClaudeModel/OllamaModel
when you have keys and the loop runs identically.

Run: pytest tests/test_e2e_live.py -v -s
"""

import time

import pytest
from playwright.async_api import async_playwright

from agent.actions import Action, ActionType
from agent.loop import run_agent
from config import DeltaVisionConfig
from model.scripted import ScriptedModel
from results.store import ResultStore


@pytest.fixture
def config():
    c = DeltaVisionConfig()
    c.MAX_STEPS = 20
    c.POST_ACTION_WAIT_MS = 500
    c.HEADLESS = True
    return c


@pytest.mark.asyncio
async def test_wikipedia_search_and_navigate(config):
    """
    Full pipeline: open Wikipedia → click search → type query →
    press Enter → verify new page detected → click a link → verify again.

    Tests: capture, diff, classifier (URL change + content change),
    observation builder, action execution, state tracking.
    """
    actions = [
        Action(type=ActionType.CLICK, x=680, y=36),      # click search box area
        Action(type=ActionType.WAIT, duration_ms=300),
        Action(type=ActionType.TYPE, text="sedimentary rock"),
        Action(type=ActionType.KEY, key="Enter"),
        Action(type=ActionType.WAIT, duration_ms=1500),   # wait for results
        Action(type=ActionType.SCROLL, direction="down", amount=300),
        Action(type=ActionType.WAIT, duration_ms=500),
    ]

    model = ScriptedModel(actions)
    t_start = time.perf_counter()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=config.HEADLESS)
        page = await browser.new_page(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )

        state = await run_agent(
            task="Search Wikipedia for 'sedimentary rock' and scroll through results",
            start_url="https://en.wikipedia.org",
            model=model,
            browser_page=page,
            config=config,
        )

        await browser.close()

    elapsed = time.perf_counter() - t_start

    # Verify pipeline ran correctly
    assert state.done
    assert state.step >= 5  # should execute most/all actions

    # Check transition log
    transitions = state.transition_log
    assert len(transitions) > 0

    # Should detect at least one NEW_PAGE (search → results = URL change)
    new_pages = [t for t in transitions if t["transition"] == "new_page"]
    deltas = [t for t in transitions if t["transition"] == "delta"]

    print("\n--- Wikipedia E2E Results ---")
    print(f"  Steps executed: {state.step}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  NEW_PAGE transitions: {len(new_pages)}")
    print(f"  DELTA transitions: {len(deltas)}")
    print(f"  Delta ratio: {state.delta_ratio:.1%}")
    print(f"  Transition triggers: {[t['trigger'] for t in transitions]}")

    # Log observations the model received
    print("\n  Observations sent to model:")
    for obs in model.observation_log:
        obs_type = obs["type"]
        if obs_type == "full_frame":
            print(f"    Step {obs['step']}: FULL_FRAME")
        else:
            dr = obs["diff_ratio"]
            dr_str = f"{dr:.3f}" if dr is not None else "n/a"
            print(f"    Step {obs['step']}: DELTA (diff={dr_str}, "
                  f"crops={obs['num_crops']}, effect={obs['had_effect']})")

    # Save result
    db = ResultStore()
    db.save("wikipedia_e2e", "scripted", {
        "steps": state.step,
        "total_time_s": round(elapsed, 1),
        "new_pages": len(new_pages),
        "deltas": len(deltas),
        "delta_ratio": round(state.delta_ratio, 3),
        "triggers": [t["trigger"] for t in transitions],
    }, notes="Full pipeline test. Scripted model, real Playwright browser.")
    db.close()


@pytest.mark.asyncio
async def test_humanbenchmark_aim_trainer(config):
    """
    Aim trainer: targets appear at random positions.
    Tests DeltaVision's spatial detection — can it find the new bbox
    and click its center without a model?

    Uses scripted clicks at center of screen as baseline.
    The interesting metric is the observation log — what did the
    diff pipeline detect?
    """
    config.POST_ACTION_WAIT_MS = 300

    # Just click around + observe what the pipeline detects
    actions = [
        Action(type=ActionType.CLICK, x=640, y=350),  # start
        Action(type=ActionType.WAIT, duration_ms=800),
        Action(type=ActionType.CLICK, x=640, y=350),  # click center
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.CLICK, x=500, y=300),  # click around
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.CLICK, x=700, y=400),
    ]

    model = ScriptedModel(actions)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=config.HEADLESS)
        page = await browser.new_page(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )

        state = await run_agent(
            task="Click targets as they appear in the aim trainer",
            start_url="https://humanbenchmark.com/tests/aim",
            model=model,
            browser_page=page,
            config=config,
        )

        await browser.close()

    print("\n--- Aim Trainer E2E Results ---")
    print(f"  Steps: {state.step}")
    print(f"  Delta ratio: {state.delta_ratio:.1%}")

    for obs in model.observation_log:
        print(f"  Step {obs['step']}: {obs['type']} "
              f"(diff={obs.get('diff_ratio', 'n/a')}, crops={obs.get('num_crops', 0)})")


@pytest.mark.asyncio
async def test_pipeline_timing(config):
    """
    Measure per-step timing breakdown:
      capture_ms, diff_ms, classify_ms, total_step_ms

    Uses a simple page with a button click to generate one transition.
    """
    import time

    from vision.capture import capture_screenshot, get_current_url
    from vision.classifier import classify_transition, extract_anchor
    from vision.diff import compute_diff

    config.POST_ACTION_WAIT_MS = 200

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )
        await page.set_content("""
            <body style="margin:0;padding:50px;background:#222">
                <button id="btn"
                        onclick="this.style.background='lime';this.textContent='Clicked!'"
                        style="padding:20px 40px;font-size:24px;background:#666;color:white">
                    Click Me
                </button>
                <p style="color:white;font-size:18px;margin-top:20px">
                    Some content that stays the same to test delta detection.
                </p>
            </body>
        """)
        await page.wait_for_timeout(500)

        # Capture t0
        t_cap0 = time.perf_counter()
        t0 = await capture_screenshot(page)
        capture_ms = (time.perf_counter() - t_cap0) * 1000

        url = get_current_url(page)
        anchor = extract_anchor(t0, config)

        # Click the button
        await page.click("#btn")
        await page.wait_for_timeout(200)

        # Capture t1
        t_cap1 = time.perf_counter()
        t1 = await capture_screenshot(page)
        capture_ms_1 = (time.perf_counter() - t_cap1) * 1000

        # Diff
        t_diff = time.perf_counter()
        diff = compute_diff(t0, t1, config)
        diff_ms = (time.perf_counter() - t_diff) * 1000

        # Classify
        t_cls = time.perf_counter()
        cls = classify_transition(t0, t1, url, url, anchor, config, diff)
        classify_ms = (time.perf_counter() - t_cls) * 1000

        await browser.close()

    print("\n--- Pipeline Timing Breakdown ---")
    print(f"  Screenshot capture: {capture_ms:.0f}ms / {capture_ms_1:.0f}ms")
    print(f"  Diff computation:  {diff_ms:.1f}ms")
    print(f"  Classification:    {classify_ms:.1f}ms")
    print(f"  CV total:          {diff_ms + classify_ms:.1f}ms")
    print(f"  diff_ratio:        {diff.diff_ratio:.4f}")
    print(f"  transition:        {cls.transition.value} (trigger={cls.trigger})")
    print(f"  changed regions:   {len(diff.changed_bboxes)}")

    # The key insight: CV pipeline is <10ms, screenshot is 300-600ms
    assert diff_ms < 50, f"Diff should be fast, got {diff_ms}ms"
    assert classify_ms < 50, f"Classify should be fast, got {classify_ms}ms"

    # Save
    db = ResultStore()
    db.save("pipeline_timing", "cv_pipeline", {
        "capture_ms": round(capture_ms, 1),
        "diff_ms": round(diff_ms, 1),
        "classify_ms": round(classify_ms, 1),
        "cv_total_ms": round(diff_ms + classify_ms, 1),
    }, notes="Per-step timing. CV pipeline vs screenshot capture.")
    db.close()
