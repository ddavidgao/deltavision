"""
Ablation study: DeltaVision (delta gating) vs full-frame-only baseline.

Same task, same model, same site. Only difference: whether the agent receives
delta observations or full screenshots every step.

Measures:
- Steps to complete
- Observation types sent (delta vs full_frame)
- Estimated token cost (image tokens)
- Per-step model call latency
- Total wall time

Usage:
  python benchmarks/ablation/run_ablation.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agent.loop import run_agent
from config import DeltaVisionConfig
from model.ollama import OllamaModel
from results.store import ResultStore

# Token estimation for Claude API pricing model
# A 1280x900 screenshot ≈ 1600 tokens (base64 PNG)
# A delta crop ~200x150 ≈ 100-200 tokens
# Multiple crops per delta ≈ 300-600 tokens total
FULL_FRAME_TOKENS = 1600
DELTA_TOKENS_AVG = 400  # conservative: diff image + 2-3 crops


TASKS = [
    {
        "name": "wikipedia_search",
        "task": "Go to Wikipedia and search for 'computer vision'. Click the first result. Report done when you see the article.",
        "url": "https://en.wikipedia.org",
        "max_steps": 20,
    },
]


async def run_one(task_def: dict, model_name: str, force_full: bool, headless: bool = True):
    """Run a single agent task. Returns metrics dict."""
    config = DeltaVisionConfig()
    config.HEADLESS = headless
    config.FORCE_FULL_FRAME = force_full

    model = OllamaModel(model=model_name, vision=True)
    mode = "full_frame_only" if force_full else "deltavision"

    print(f"\n{'='*60}")
    print(f"  Mode: {mode} | Model: {model_name}")
    print(f"  Task: {task_def['name']}")
    print(f"{'='*60}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )
        page = await context.new_page()

        t_start = time.perf_counter()
        state = await run_agent(
            task=task_def["task"],
            start_url=task_def["url"],
            model=model,
            browser_page=page,
            config=config,
            safety=None,
        )
        wall_time_s = time.perf_counter() - t_start
        await browser.close()

    # Count observation types from transition log
    full_frames = 0
    deltas = 0
    for entry in state.transition_log:
        # The transition log tracks classifier decisions
        if isinstance(entry, dict):
            if entry.get("transition") == "new_page" or "forced_full" in entry.get("trigger", ""):
                full_frames += 1
            else:
                deltas += 1

    # In force_full mode, everything is full_frame
    if force_full:
        full_frames = state.step + 1  # +1 for initial observation
        deltas = 0
    else:
        full_frames = state.new_page_count + 1  # +1 for initial
        deltas = state.step - state.new_page_count

    estimated_tokens = (full_frames * FULL_FRAME_TOKENS) + (deltas * DELTA_TOKENS_AVG)

    metrics = {
        "mode": mode,
        "model": model_name,
        "task": task_def["name"],
        "steps": state.step,
        "done": state.done,
        "delta_ratio": round(state.delta_ratio, 3),
        "full_frames_sent": full_frames,
        "deltas_sent": deltas,
        "new_page_count": state.new_page_count,
        "estimated_image_tokens": estimated_tokens,
        "wall_time_s": round(wall_time_s, 1),
        "avg_step_time_s": round(wall_time_s / max(state.step, 1), 1),
    }

    print(f"\n  Results ({mode}):")
    print(f"    Steps: {state.step}, Done: {state.done}")
    print(f"    Full frames: {full_frames}, Deltas: {deltas}")
    print(f"    Estimated tokens: {estimated_tokens:,}")
    print(f"    Wall time: {wall_time_s:.1f}s ({wall_time_s/max(state.step,1):.1f}s/step)")

    return metrics


async def run_ablation():
    model_name = "qwen2.5vl:7b"
    all_results = []

    for task_def in TASKS:
        # Run with DeltaVision (delta gating ON)
        delta_metrics = await run_one(task_def, model_name, force_full=False)
        all_results.append(delta_metrics)

        await asyncio.sleep(2)  # brief pause between runs

        # Run with full-frame-only (delta gating OFF)
        full_metrics = await run_one(task_def, model_name, force_full=True)
        all_results.append(full_metrics)

    # Comparison
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)

    for task_def in TASKS:
        dv = next(r for r in all_results if r["task"] == task_def["name"] and r["mode"] == "deltavision")
        ff = next(r for r in all_results if r["task"] == task_def["name"] and r["mode"] == "full_frame_only")

        token_savings = ff["estimated_image_tokens"] - dv["estimated_image_tokens"]
        token_pct = (token_savings / ff["estimated_image_tokens"] * 100) if ff["estimated_image_tokens"] > 0 else 0

        print(f"\nTask: {task_def['name']}")
        print(f"  {'Metric':<30} {'DeltaVision':>15} {'Full-Frame':>15} {'Savings':>15}")
        print(f"  {'-'*75}")
        print(f"  {'Steps':<30} {dv['steps']:>15} {ff['steps']:>15} {'':>15}")
        print(f"  {'Full frames sent':<30} {dv['full_frames_sent']:>15} {ff['full_frames_sent']:>15} {'':>15}")
        print(f"  {'Deltas sent':<30} {dv['deltas_sent']:>15} {ff['deltas_sent']:>15} {'':>15}")
        print(f"  {'Delta ratio':<30} {dv['delta_ratio']:>14.0%} {ff['delta_ratio']:>14.0%} {'':>15}")
        print(f"  {'Est. image tokens':<30} {dv['estimated_image_tokens']:>15,} {ff['estimated_image_tokens']:>15,} {token_savings:>14,} ({token_pct:.0f}%)")
        print(f"  {'Wall time (s)':<30} {dv['wall_time_s']:>15.1f} {ff['wall_time_s']:>15.1f} {'':>15}")

    # Save to SQLite
    db = ResultStore()
    for r in all_results:
        db.save(
            benchmark=f"ablation_{r['task']}",
            backend=f"ollama_{model_name}_{r['mode']}",
            metrics=r,
            config={"force_full_frame": r["mode"] == "full_frame_only"},
            notes=f"Ablation: {r['mode']}, {r['steps']} steps",
        )
    db.close()

    # Save JSON
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path} and SQLite")


if __name__ == "__main__":
    asyncio.run(run_ablation())
