"""
DeltaVision CLI — Delta-first computer use agent.

Usage:
  # Claude API
  python main.py --task "Complete the quiz" --url https://example.com

  # OpenAI
  python main.py --task "..." --url ... --backend openai

  # Local VLM via transformers (Qwen2.5-VL)
  python main.py --task "..." --url ... --backend local

  # Any model via Ollama (Hermes, Qwen, LLaVA, etc.)
  python main.py --task "..." --url ... --backend ollama --model hermes3:8b

  # With safety layer
  python main.py --task "..." --url ... --safety strict
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env — check cwd, script dir, then walk up parents
for _start in [Path.cwd(), Path(__file__).resolve().parent]:
    for _d in [_start, *_start.parents]:
        _env = _d / ".env"
        if _env.exists():
            load_dotenv(_env, override=True)
            break

from playwright.async_api import async_playwright

from agent.loop import run_agent
from config import MCGRAWHILL_CONFIG, DeltaVisionConfig


def get_model(backend: str, config: DeltaVisionConfig, model_override: str = None, base_url: str = None):
    if backend == "claude":
        from model.claude import ClaudeModel

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        return ClaudeModel(api_key=api_key, model=model_override or config.CLAUDE_MODEL)

    elif backend == "openai":
        from model.openai import OpenAIModel

        # base_url points at an OpenAI-compatible server (llama-server, vLLM, SGLang).
        # When set, we skip the OPENAI_API_KEY requirement — local servers don't need one.
        if base_url:
            return OpenAIModel(
                api_key="sk-no-key-required",
                model=model_override or "local-model",
                base_url=base_url,
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set (or pass --base-url for local endpoint)", file=sys.stderr)
            sys.exit(1)
        return OpenAIModel(api_key=api_key, model=model_override or "gpt-4o")

    elif backend == "ollama":
        from model.ollama import OllamaModel

        model_name = model_override or "qwen2.5-vl:7b"
        # Auto-detect vision capability from model name
        vision = any(v in model_name.lower() for v in ["vl", "llava", "vision", "cogagent", "molmo", "minicpm-v", "minicpm_v", "ui-tars", "uitars", "gemma4"])
        if not vision:
            print(f"Note: {model_name} detected as text-only (no vision). "
                  "DeltaVision will send structured text descriptions instead of images.",
                  file=sys.stderr)
        return OllamaModel(model=model_name, vision=vision)

    elif backend == "local":
        from model.local import LocalModel

        return LocalModel(
            model_name=model_override or config.LOCAL_MODEL,
            quantization=config.LOCAL_QUANTIZATION,
        )
    else:
        print(f"Unknown backend: {backend}. Options: claude, openai, ollama, local", file=sys.stderr)
        sys.exit(1)


def get_safety(safety_mode: str):
    if safety_mode == "none":
        return None
    from safety import EDUCATIONAL, PERMISSIVE, STRICT
    modes = {"permissive": PERMISSIVE, "strict": STRICT, "educational": EDUCATIONAL}
    if safety_mode in modes:
        return modes[safety_mode]
    print(f"Unknown safety mode: {safety_mode}. Options: none, permissive, strict, educational",
          file=sys.stderr)
    sys.exit(1)


async def main(args):
    # Select config preset
    if args.preset == "mcgrawhill":
        config = MCGRAWHILL_CONFIG
    else:
        config = DeltaVisionConfig()

    # CLI overrides
    if args.model:
        if args.backend == "claude":
            config.CLAUDE_MODEL = args.model
        else:
            config.LOCAL_MODEL = args.model
    if args.quantization:
        config.LOCAL_QUANTIZATION = args.quantization
    if args.max_steps:
        config.MAX_STEPS = args.max_steps
    if args.headless:
        config.HEADLESS = True
    if args.force_full_frame:
        config.FORCE_FULL_FRAME = True

    model = get_model(args.backend, config, args.model, base_url=args.base_url)
    safety = get_safety(args.safety)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=config.HEADLESS,
            args=[f"--window-size={config.BROWSER_WIDTH},{config.BROWSER_HEIGHT}"],
        )
        context = await browser.new_context(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )
        page = await context.new_page()

        state = await run_agent(
            task=args.task,
            start_url=args.url,
            model=model,
            browser_page=page,
            config=config,
            safety=safety,
        )

        # Dump results
        result = {
            "task": state.task,
            "steps": state.step,
            "done": state.done,
            "delta_ratio": round(state.delta_ratio, 3),
            "new_page_count": state.new_page_count,
            "transition_log": state.transition_log,
            "timestamp": datetime.now().isoformat(),
        }

        out_path = args.output or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {out_path}")
        print(f"Steps: {state.step}, Delta ratio: {state.delta_ratio:.1%}, New pages: {state.new_page_count}")

        await browser.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeltaVision — delta-first computer use agent")
    parser.add_argument("--task", required=True, help="Task description for the agent")
    parser.add_argument("--url", required=True, help="Starting URL")
    parser.add_argument("--backend", choices=["claude", "openai", "ollama", "local"], default="claude", help="Model backend")
    parser.add_argument("--model", help="Override model name/ID")
    parser.add_argument("--quantization", choices=["4bit", "8bit"], help="Quantization for local models")
    parser.add_argument("--preset", choices=["default", "mcgrawhill"], default="default", help="Config preset")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--safety", choices=["none", "permissive", "strict", "educational"], default="permissive", help="Safety mode")
    parser.add_argument("--force-full-frame", action="store_true", help="Ablation: always send full frame, disable delta gating")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for local VLM servers (llama-server, vLLM). Only applies with --backend openai. Example: http://localhost:8080/v1")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def cli_entry():
    """Console-script entrypoint (pyproject.toml scripts table)."""
    args = _build_parser().parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_entry()
