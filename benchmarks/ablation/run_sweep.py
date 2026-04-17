"""
Multi-model, multi-site ablation sweep.

Runs (model × site × mode × run_idx) matrix and logs every run to SQLite.
Supports filtering by model so we can run one model at a time while others
download / swap in VRAM (16GB won't fit 3 concurrent).

Usage:
  # Run full sweep for one model (pass model family tag)
  python benchmarks/ablation/run_sweep.py --model mai-ui-8b

  # Resume / rerun one site only
  python benchmarks/ablation/run_sweep.py --model mai-ui-8b --site todomvc

  # Headless (default) or with browser visible
  python benchmarks/ablation/run_sweep.py --model mai-ui-8b --no-headless
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DeltaVisionConfig
from agent.loop import run_agent
from results.store import ResultStore


FULL_FRAME_TOKENS = 1600
DELTA_TOKENS_AVG = 400


# ── Task definitions (deterministic, reproducible) ──────────────────────

TASKS = {
    "wikipedia": {
        "url": "https://en.wikipedia.org",
        "task": "Search Wikipedia for 'computer vision' using the search box. Click the first matching result link. Report done when the article page is loaded.",
        "max_steps": 15,
    },
    "todomvc": {
        "url": "https://todomvc.com/examples/react",
        "task": "Add three todos by typing each into the input and pressing Enter: 'buy milk', 'feed cat', 'pay rent'. Then click the checkbox next to 'feed cat' to mark it complete. Report done when all three todos are visible and one is checked.",
        "max_steps": 20,
    },
    "hackernews": {
        "url": "https://news.ycombinator.com",
        "task": "Click the 'More' link at the bottom of the page to navigate to page 2. Report done when the URL contains '?p=2'.",
        "max_steps": 10,
    },
}


# ── Model configs ──────────────────────────────────────────────────────

MODELS = {
    "mai-ui-8b": {
        "backend": "openai",
        "base_url": "http://localhost:8080/v1",
        "model_name": "MAI-UI-8B-q5_k_m.gguf",
        "backend_label": "llamacpp_mai-ui-8b",
    },
    "qwen3-vl-8b": {
        "backend": "openai",
        "base_url": "http://localhost:8080/v1",
        "model_name": "qwen3-vl-8b-instruct-q5_k_m.gguf",
        "backend_label": "llamacpp_qwen3-vl-8b",
    },
    "qwen2.5-vl-7b": {
        "backend": "ollama",
        "model_name": "qwen2.5vl:7b",
        "backend_label": "ollama_qwen2.5vl-7b",
    },
}

RUNS_PER_CELL = 3
MODES = ["deltavision", "full_frame_only"]


def build_model(cfg: dict):
    if cfg["backend"] == "openai":
        from model.openai import OpenAIModel
        return OpenAIModel(
            api_key="sk-no-key-required",
            model=cfg["model_name"],
            base_url=cfg["base_url"],
        )
    elif cfg["backend"] == "ollama":
        from model.ollama import OllamaModel
        return OllamaModel(model=cfg["model_name"], vision=True)
    raise ValueError(f"Unknown backend: {cfg['backend']}")


async def run_one(task_name: str, task_def: dict, model_cfg: dict, force_full: bool, headless: bool, run_idx: int):
    config = DeltaVisionConfig()
    config.HEADLESS = headless
    config.FORCE_FULL_FRAME = force_full
    config.MAX_STEPS = task_def["max_steps"]

    mode = "full_frame_only" if force_full else "deltavision"
    model = build_model(model_cfg)

    print(f"\n{'='*70}")
    print(f"  [{run_idx}/{RUNS_PER_CELL}] {model_cfg['backend_label']} | {task_name} | {mode}")
    print(f"{'='*70}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )
        page = await context.new_page()

        t_start = time.perf_counter()
        try:
            state = await run_agent(
                task=task_def["task"],
                start_url=task_def["url"],
                model=model,
                browser_page=page,
                config=config,
                safety=None,
            )
            error = None
        except Exception as e:
            state = None
            error = str(e)
        wall_time_s = time.perf_counter() - t_start
        await browser.close()

    if error:
        print(f"  ERROR: {error}")
        return {
            "run_idx": run_idx,
            "mode": mode,
            "model": model_cfg["backend_label"],
            "task": task_name,
            "steps": 0,
            "done": False,
            "delta_ratio": 0.0,
            "full_frames_sent": 0,
            "deltas_sent": 0,
            "new_page_count": 0,
            "estimated_image_tokens": 0,
            "wall_time_s": round(wall_time_s, 1),
            "error": error,
        }

    if force_full:
        full_frames = state.step + 1
        deltas = 0
    else:
        full_frames = state.new_page_count + 1
        deltas = state.step - state.new_page_count

    estimated_tokens = (full_frames * FULL_FRAME_TOKENS) + (deltas * DELTA_TOKENS_AVG)

    metrics = {
        "run_idx": run_idx,
        "mode": mode,
        "model": model_cfg["backend_label"],
        "task": task_name,
        "steps": state.step,
        "done": state.done,
        "delta_ratio": round(state.delta_ratio, 3),
        "full_frames_sent": full_frames,
        "deltas_sent": deltas,
        "new_page_count": state.new_page_count,
        "estimated_image_tokens": estimated_tokens,
        "wall_time_s": round(wall_time_s, 1),
        "avg_step_time_s": round(wall_time_s / max(state.step, 1), 1),
        "error": None,
    }

    print(f"  -> steps={state.step}, done={state.done}, delta={state.delta_ratio:.0%}, tokens~{estimated_tokens:,}, {wall_time_s:.0f}s")
    return metrics


async def run_sweep(model_key: str, site_filter: str = None, headless: bool = True):
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Options: {', '.join(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    model_cfg = MODELS[model_key]
    sites = [site_filter] if site_filter else list(TASKS.keys())
    for s in sites:
        if s not in TASKS:
            print(f"Unknown site: {s}. Options: {', '.join(TASKS.keys())}", file=sys.stderr)
            sys.exit(1)

    db = ResultStore()
    all_metrics = []

    for site in sites:
        task_def = TASKS[site]
        for mode in MODES:
            force_full = (mode == "full_frame_only")
            for run_idx in range(1, RUNS_PER_CELL + 1):
                metrics = await run_one(site, task_def, model_cfg, force_full, headless, run_idx)
                all_metrics.append(metrics)

                # Save each run immediately (crash resilience)
                db.save(
                    benchmark=f"sweep_{site}",
                    backend=f"{model_cfg['backend_label']}_{mode}",
                    metrics=metrics,
                    config={
                        "force_full_frame": force_full,
                        "run_idx": run_idx,
                        "max_steps": task_def["max_steps"],
                    },
                    notes=f"Sweep run {run_idx}/{RUNS_PER_CELL}: {mode} on {site} with {model_cfg['backend_label']}",
                )

                await asyncio.sleep(1)

    db.close()

    # Summary table
    print("\n" + "=" * 90)
    print(f"SWEEP COMPLETE: {model_cfg['backend_label']}")
    print("=" * 90)
    print(f"  {'Site':<15} {'Mode':<18} {'Mean Steps':>12} {'Mean Tokens':>14} {'Done %':>8} {'Avg s':>8}")
    for site in sites:
        for mode in MODES:
            cells = [m for m in all_metrics if m["task"] == site and m["mode"] == mode]
            if not cells:
                continue
            mean_steps = sum(m["steps"] for m in cells) / len(cells)
            mean_tokens = sum(m["estimated_image_tokens"] for m in cells) / len(cells)
            done_pct = sum(1 for m in cells if m["done"]) / len(cells) * 100
            mean_time = sum(m["wall_time_s"] for m in cells) / len(cells)
            print(f"  {site:<15} {mode:<18} {mean_steps:>12.1f} {mean_tokens:>14,.0f} {done_pct:>7.0f}% {mean_time:>7.1f}s")

    out_path = Path(__file__).parent / f"sweep_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved: {out_path} + SQLite")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--site", choices=list(TASKS.keys()) + [None], default=None)
    p.add_argument("--no-headless", action="store_true")
    args = p.parse_args()

    asyncio.run(run_sweep(
        model_key=args.model,
        site_filter=args.site,
        headless=not args.no_headless,
    ))
