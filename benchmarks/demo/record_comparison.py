"""
Record side-by-side comparison videos: DeltaVision vs full-frame baseline.

Playwright natively records browser sessions as .webm videos.
Outputs:
  demo/videos/deltavision_<task>.webm
  demo/videos/fullframe_<task>.webm

Usage:
  python benchmarks/demo/record_comparison.py
  python benchmarks/demo/record_comparison.py --task wikipedia
  python benchmarks/demo/record_comparison.py --task hackernews
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from datetime import datetime

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DeltaVisionConfig
from agent.loop import run_agent
from model.ollama import OllamaModel


VIDEO_DIR = Path(__file__).parent / "videos"

TASKS = {
    "wikipedia": {
        "task": "Go to Wikipedia and search for 'computer vision'. Click the first result. Report done ONLY when you can see the Computer Vision article page.",
        "url": "https://en.wikipedia.org",
        "max_steps": 25,
        "success_keywords": ["computer_vision", "computer%20vision"],
    },
    "hackernews": {
        "task": "Navigate to Hacker News. Click on the top story title to read the article. Report done when you see the article.",
        "url": "https://news.ycombinator.com",
        "max_steps": 10,
        "success_keywords": [],  # external link, can't predict
    },
    "wikipedia_multi": {
        "task": (
            "Do these steps in order: "
            "1. Search Wikipedia for 'neural network'. "
            "2. Click the first result to open the article. "
            "3. Find and click the link to 'deep learning' within the article. "
            "Report done ONLY when you can see the Deep Learning article page."
        ),
        "url": "https://en.wikipedia.org",
        "max_steps": 30,
        "success_keywords": ["deep_learning", "deep%20learning"],
    },
}


async def record_run(task_name: str, task_def: dict, mode: str, model_name: str):
    """Record a single agent run as a video."""
    force_full = mode == "fullframe"
    config = DeltaVisionConfig()
    config.HEADLESS = True  # headless works for video recording too
    config.FORCE_FULL_FRAME = force_full
    config.MAX_STEPS = task_def["max_steps"]

    model = OllamaModel(model=model_name, vision=True)

    video_subdir = VIDEO_DIR / f"{mode}_{task_name}"
    video_subdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Recording: {mode} | {task_name}")
    print(f"  Video dir: {video_subdir}")
    print(f"{'='*60}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[f"--window-size={config.BROWSER_WIDTH},{config.BROWSER_HEIGHT}"],
        )
        context = await browser.new_context(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT},
            record_video_dir=str(video_subdir),
            record_video_size={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT},
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
        wall_time = time.perf_counter() - t_start
        final_url = page.url

        # Close context to finalize video
        video_path = await page.video.path()
        await context.close()
        await browser.close()

    # Check if task actually completed (URL should have changed from start)
    success_keywords = task_def.get("success_keywords", [])
    actually_completed = (
        state.done
        and state.step >= 2
        and (not success_keywords or any(kw in final_url.lower() for kw in success_keywords))
    )

    print(f"  Steps: {state.step}, Done: {state.done}, Delta ratio: {state.delta_ratio:.0%}")
    print(f"  Final URL: {final_url}")
    print(f"  Actually completed: {actually_completed}")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Video: {video_path}")

    return {
        "mode": mode,
        "task": task_name,
        "steps": state.step,
        "done": state.done,
        "actually_completed": actually_completed,
        "delta_ratio": round(state.delta_ratio, 3),
        "wall_time_s": round(wall_time, 1),
        "final_url": final_url,
        "video_path": str(video_path),
    }


async def record_with_retries(task_name, task_def, mode, model_name, max_retries=3):
    """Retry recording until the model actually completes the task."""
    for attempt in range(max_retries):
        result = await record_run(task_name, task_def, mode, model_name)
        if result["actually_completed"]:
            return result
        print(f"  Attempt {attempt+1}/{max_retries} failed (model quit early). Retrying...")
        await asyncio.sleep(2)
    print(f"  WARNING: {mode} did not complete task after {max_retries} attempts. Using last recording.")
    return result


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS.keys()), default="wikipedia")
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--skip-fullframe", action="store_true",
                        help="Only record DeltaVision mode")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    task_def = TASKS[args.task]
    results = []

    # Record DeltaVision mode (retry until task actually completes)
    dv = await record_with_retries(args.task, task_def, "deltavision", args.model, args.retries)
    results.append(dv)

    if not args.skip_fullframe:
        await asyncio.sleep(2)
        # Record full-frame mode (may fail — that's the point)
        ff = await record_run(args.task, task_def, "fullframe", args.model)
        results.append(ff)

    # Summary
    print("\n" + "=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    for r in results:
        print(f"  {r['mode']:<15} steps={r['steps']:<4} done={r['done']:<6} "
              f"time={r['wall_time_s']}s  video={r['video_path']}")

    # Save metadata
    meta_path = VIDEO_DIR / f"meta_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetadata: {meta_path}")
    print(f"\nTo create a side-by-side comparison, use ffmpeg:")
    if len(results) == 2:
        print(f"  ffmpeg -i \"{results[0]['video_path']}\" -i \"{results[1]['video_path']}\" "
              f"-filter_complex hstack=inputs=2 \"demo_comparison_{args.task}.mp4\"")


if __name__ == "__main__":
    asyncio.run(main())
