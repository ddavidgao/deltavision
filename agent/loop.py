"""
Core agent loop. Orchestrates the full DeltaVision pipeline.

Loop invariants:
- t0 always holds the last anchor frame (full frame at last NEW_PAGE)
- t1 is always the current observation
- The model NEVER makes transition type decisions
- Retry logic is purely threshold-based, not model-driven
"""

import asyncio
import logging
from PIL import Image

from vision.capture import capture_screenshot, get_current_url
from vision.diff import compute_diff, extract_crops
from vision.classifier import (
    classify_transition,
    extract_anchor,
    TransitionType,
)
from vision.elements import extract_page_state
from observation.builder import build_observation
from agent.state import AgentState
from agent.actions import execute_action
from safety import SafetyLayer

logger = logging.getLogger(__name__)


async def run_agent(task: str, start_url: str, model, browser_page, config, safety: SafetyLayer | None = None) -> AgentState:
    """
    Main DeltaVision agent loop.

    Terminates when:
    - model returns done=True
    - max_steps exceeded
    - max_consecutive_failures exceeded
    """
    state = AgentState(task=task)

    # Bootstrap: navigate, capture initial full frame, establish anchor
    await browser_page.goto(start_url)
    try:
        await browser_page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        # Some sites (HumanBenchmark, SPAs) never reach networkidle
        await browser_page.wait_for_load_state("domcontentloaded")

    t0 = await capture_screenshot(browser_page)
    url_t0 = get_current_url(browser_page)
    anchor_template = extract_anchor(t0, config)
    initial_state = await extract_page_state(browser_page)

    obs = build_observation(
        obs_type="full_frame",
        task=task,
        step=0,
        last_action=None,
        frame=t0,
        url=url_t0,
        trigger_reason="initial",
        clickable_elements=initial_state.get("elements", []),
        focus=initial_state.get("focus"),
    )
    state.add_observation(obs)

    while not state.done and state.step < config.MAX_STEPS:
        # Get next action from model
        response = await model.predict(obs, state)
        state.add_response(response)

        if response.action is None or response.done:
            state.done = True
            logger.info(
                "Agent finished at step %d. Reason: %s",
                state.step,
                response.reasoning,
            )
            break

        action = response.action
        logger.info("Step %d: %s (confidence=%.2f)", state.step, action, response.confidence)

        # Safety check — runs regardless of which model backend generated the action
        if safety is not None:
            url_now = get_current_url(browser_page)
            check = safety.check_action(action, url_now)
            if not check.allowed:
                logger.warning("SAFETY BLOCK: %s", check.reason)
                state.step += 1
                # Tell model its action was blocked
                sb_state = await extract_page_state(browser_page)
                obs = build_observation(
                    obs_type="full_frame",
                    task=task,
                    step=state.step,
                    last_action=action,
                    frame=await capture_screenshot(browser_page),
                    url=url_now,
                    trigger_reason=f"safety_block:{check.reason}",
                    clickable_elements=sb_state.get("elements", []),
                    focus=sb_state.get("focus"),
                )
                state.add_observation(obs)
                continue

        # Execute in browser
        url_before = get_current_url(browser_page)
        await execute_action(action, browser_page, config)

        # Wait for page to react
        await asyncio.sleep(config.POST_ACTION_WAIT_MS / 1000)

        # Capture post-action frame + DOM state (clickables + focus)
        t1 = await capture_screenshot(browser_page)
        url_after = get_current_url(browser_page)
        dom_state = await extract_page_state(browser_page)
        clickables = dom_state.get("elements", [])
        focus_state = dom_state.get("focus")

        # Compute diff (always — needed for classification and observation)
        diff_result = compute_diff(t0, t1, config)

        # Classify transition — pure CV, no model
        classification = classify_transition(
            t0=t0,
            t1=t1,
            url_before=url_before,
            url_after=url_after,
            anchor_template=anchor_template,
            config=config,
            diff_result=diff_result,
            last_action_type=action.type.value,
        )
        state.log_transition(classification, action, state.step)

        logger.debug(
            "Transition: %s (trigger=%s, diff=%.3f, phash=%d, anchor=%.2f)",
            classification.transition.value,
            classification.trigger,
            classification.diff_ratio,
            classification.phash_distance,
            classification.anchor_score,
        )

        # Ablation: FORCE_FULL_FRAME overrides delta gating — always send full frame
        force_full = getattr(config, 'FORCE_FULL_FRAME', False)

        if classification.transition == TransitionType.NEW_PAGE or force_full:
            # Re-anchor on new context (or forced full frame)
            t0 = t1
            url_t0 = url_after
            anchor_template = extract_anchor(t0, config)

            if classification.transition == TransitionType.NEW_PAGE:
                state.reset_no_change_streak()
                state.increment_new_page_count()

            trigger = classification.trigger if not force_full else f"forced_full|{classification.trigger}"
            obs = build_observation(
                obs_type="full_frame",
                task=task,
                step=state.step,
                last_action=action,
                frame=t1,
                url=url_after,
                trigger_reason=trigger,
                clickable_elements=clickables,
                focus=focus_state,
            )

        else:  # DELTA
            # Re-anchor after scroll since viewport position changed
            if action.type.value == "scroll":
                t0 = t1
                anchor_template = extract_anchor(t0, config)

            crops = extract_crops(t0, t1, diff_result.changed_bboxes, config.CROP_PADDING)

            if not diff_result.action_had_effect:
                state.increment_no_change_streak()
            else:
                state.reset_no_change_streak()

            # Force full frame refresh if stuck
            if state.no_change_streak >= config.MAX_NO_EFFECT_RETRIES:
                logger.warning(
                    "No-effect streak hit %d — forcing full frame refresh",
                    state.no_change_streak,
                )
                t0_refresh = await capture_screenshot(browser_page)
                refresh_state = await extract_page_state(browser_page)
                obs = build_observation(
                    obs_type="full_frame",
                    task=task,
                    step=state.step,
                    last_action=action,
                    frame=t0_refresh,
                    url=url_after,
                    trigger_reason="force_refresh_no_effect",
                    clickable_elements=refresh_state.get("elements", []),
                    focus=refresh_state.get("focus"),
                )
                state.reset_no_change_streak()
                t0 = t0_refresh
                anchor_template = extract_anchor(t0, config)
            else:
                obs = build_observation(
                    obs_type="delta",
                    task=task,
                    step=state.step,
                    last_action=action,
                    diff_result=diff_result,
                    crops=crops,
                    action_had_effect=diff_result.action_had_effect,
                    no_change_count=state.no_change_streak,
                    current_frame=t1,
                    clickable_elements=clickables,
                    focus=focus_state,
                )

        state.add_observation(obs)
        state.step += 1

    logger.info(
        "Run complete. Steps: %d, Delta ratio: %.1f%%, New pages: %d",
        state.step,
        state.delta_ratio * 100,
        state.new_page_count,
    )
    return state
