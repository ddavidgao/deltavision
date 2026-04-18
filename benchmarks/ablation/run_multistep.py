"""
Multi-step task ablation: longer tasks show token savings compound.

Task: Navigate Wikipedia, visit 3 different articles in sequence.
Compare DeltaVision (delta gating) vs full-frame on the same task.
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

FULL_FRAME_TOKENS = 1600
DELTA_TOKENS_AVG = 400

TASK = {
    "name": "wikipedia_multistep",
    "task": (
        "Do these steps in order: "
        "1. Search Wikipedia for 'neural network'. "
        "2. Click the first result to open the article. "
        "3. Find and click the link to 'deep learning' within the article. "
        "Report done when you can see the Deep Learning article."
    ),
    "url": "https://en.wikipedia.org",
    "max_steps": 30,
}


async def run_one(model_name: str, force_full: bool):
    config = DeltaVisionConfig()
    config.HEADLESS = True
    config.FORCE_FULL_FRAME = force_full
    config.MAX_STEPS = TASK["max_steps"]

    model = OllamaModel(model=model_name, vision=True)
    mode = "full_frame_only" if force_full else "deltavision"

    print(f"\n{'='*60}")
    print(f"  Mode: {mode} | Model: {model_name}")
    print(f"  Task: {TASK['name']}")
    print(f"{'='*60}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )

        t_start = time.perf_counter()
        state = await run_agent(
            task=TASK["task"],
            start_url=TASK["url"],
            model=model,
            browser_page=page,
            config=config,
            safety=None,
        )
        wall_time = time.perf_counter() - t_start
        await browser.close()

    if force_full:
        full_frames = state.step + 1
        deltas = 0
    else:
        full_frames = state.new_page_count + 1
        deltas = state.step - state.new_page_count

    tokens = (full_frames * FULL_FRAME_TOKENS) + (deltas * DELTA_TOKENS_AVG)

    metrics = {
        "mode": mode,
        "model": model_name,
        "task": TASK["name"],
        "steps": state.step,
        "done": state.done,
        "delta_ratio": round(state.delta_ratio, 3),
        "full_frames_sent": full_frames,
        "deltas_sent": deltas,
        "new_page_count": state.new_page_count,
        "estimated_image_tokens": tokens,
        "wall_time_s": round(wall_time, 1),
    }

    print(f"  Steps: {state.step}, Done: {state.done}, Delta ratio: {state.delta_ratio:.0%}")
    print(f"  Tokens: {tokens:,}, Wall time: {wall_time:.1f}s")

    return metrics


async def main():
    model = "qwen2.5vl:7b"
    results = []

    # DeltaVision mode
    dv = await run_one(model, force_full=False)
    results.append(dv)

    await asyncio.sleep(2)

    # Full-frame mode
    ff = await run_one(model, force_full=True)
    results.append(ff)

    # Comparison
    print("\n" + "=" * 70)
    print("MULTI-STEP ABLATION COMPARISON")
    print("=" * 70)
    savings = ff["estimated_image_tokens"] - dv["estimated_image_tokens"]
    pct = savings / ff["estimated_image_tokens"] * 100 if ff["estimated_image_tokens"] > 0 else 0
    print(f"  {'':30} {'DeltaVision':>15} {'Full-Frame':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Steps':<30} {dv['steps']:>15} {ff['steps']:>15}")
    print(f"  {'Task completed':<30} {str(dv['done']):>15} {str(ff['done']):>15}")
    print(f"  {'Delta ratio':<30} {dv['delta_ratio']:>14.0%} {ff['delta_ratio']:>14.0%}")
    print(f"  {'Est. tokens':<30} {dv['estimated_image_tokens']:>15,} {ff['estimated_image_tokens']:>15,}")
    print(f"  {'Token savings':<30} {'':>15} {savings:>14,} ({pct:.0f}%)")
    print(f"  {'Wall time':<30} {dv['wall_time_s']:>14.1f}s {ff['wall_time_s']:>14.1f}s")

    # Save
    db = ResultStore()
    for r in results:
        db.save(
            benchmark=f"multistep_{r['task']}",
            backend=f"ollama_{model}_{r['mode']}",
            metrics=r,
            notes=f"Multi-step: {r['mode']}, {r['steps']} steps, done={r['done']}",
        )
    db.close()

    out = Path(__file__).parent / "multistep_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to SQLite and {out}")


if __name__ == "__main__":
    asyncio.run(main())
