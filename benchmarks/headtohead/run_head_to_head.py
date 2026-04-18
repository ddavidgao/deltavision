"""
Head-to-head agent benchmark: DV-wrapped vs FF-baseline.

Why this exists
---------------
The scripted 77.2% spreadsheet benchmark measures *compression* on a matched
trajectory. It doesn't prove DV helps the agent succeed — it only proves DV
emits fewer tokens when both sides see the same screens.

This benchmark answers the other question: with a real Claude agent making
its own decisions, does DV-wrapped beat FF-baseline on task success AND
token cost, with measurable variance?

Method
------
- Same task, same model (claude-sonnet-4-20250514), same MAX_STEPS.
- DV config vs FF config (config.FORCE_FULL_FRAME = True).
- N=3 trials per config (we're shipping fast, not publishing a paper).
- Capture per-call Anthropic usage via a ClaudeModel subclass.
- Log: steps, total_input_tokens, total_output_tokens, done (success),
  delta_ratio, wall_time, per-step usage trace.

Output
------
- benchmarks/headtohead/head_to_head_results.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

# Ensure repo root on path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
for _d in [Path.cwd(), REPO]:
    env = _d / ".env"
    if env.exists():
        load_dotenv(env, override=True)

import anthropic
from playwright.async_api import async_playwright

from config import DeltaVisionConfig
from model.claude import ClaudeModel
from model.base import ModelResponse
from model._response_parser import extract_json, normalize_response, get_confidence
from agent.actions import parse_action
from agent.loop import run_agent

log = logging.getLogger("head_to_head")


class TokenTrackingClaudeModel(ClaudeModel):
    """ClaudeModel subclass that captures per-call Anthropic usage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_log: list[dict] = []

    async def predict(self, observation, state) -> ModelResponse:
        messages = self._build_messages(observation, state)

        # Single attempt here — we want clean variance, not retry-masked variance.
        # If the API errors, the trial fails and we log it.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=(  # re-import to avoid monkey touching parent module state
                self._system_prompt if hasattr(self, "_system_prompt") else _SYSTEM_PROMPT_FALLBACK
            ),
            messages=messages,
        )

        usage = response.usage
        self.usage_log.append({
            "step": observation.step,
            "obs_type": observation.obs_type,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        })

        raw_text = response.content[0].text
        parsed = normalize_response(extract_json(raw_text))
        action = parse_action(parsed.get("action")) if not parsed.get("done") else None

        return ModelResponse(
            action=action,
            done=parsed.get("done", False),
            reasoning=parsed.get("reasoning", ""),
            confidence=get_confidence(parsed),
            raw_response=parsed,
        )


# Import SYSTEM_PROMPT from claude.py (keeping it consistent)
from model.claude import SYSTEM_PROMPT as _SYSTEM_PROMPT_FALLBACK
TokenTrackingClaudeModel._system_prompt = _SYSTEM_PROMPT_FALLBACK


TASKS = [
    {
        "name": "todomvc_add_and_complete",
        "url": "https://todomvc.com/examples/react/dist/",
        "task": (
            "Add three todo items by typing each one and pressing Enter: "
            "'Buy groceries', 'Walk the dog', 'Call mom'. "
            "Then click the checkbox next to 'Buy groceries' to mark it as completed."
        ),
        "max_steps": 12,
    },
]

N_TRIALS = 3


async def run_trial(task_spec: dict, dv_enabled: bool, trial: int) -> dict:
    config = DeltaVisionConfig()
    config.MAX_STEPS = task_spec["max_steps"]
    config.HEADLESS = True
    config.FORCE_FULL_FRAME = not dv_enabled

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    # Use the verified date-stamped model (matches integration tests)
    model = TokenTrackingClaudeModel(api_key=api_key, model="claude-sonnet-4-20250514")

    started = time.time()
    result: dict = {
        "task": task_spec["name"],
        "config": "DV" if dv_enabled else "FF",
        "trial": trial,
    }

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[f"--window-size={config.BROWSER_WIDTH},{config.BROWSER_HEIGHT}"],
        )
        ctx = await browser.new_context(
            viewport={"width": config.BROWSER_WIDTH, "height": config.BROWSER_HEIGHT}
        )
        page = await ctx.new_page()

        try:
            state = await run_agent(
                task=task_spec["task"],
                start_url=task_spec["url"],
                model=model,
                browser_page=page,
                config=config,
                safety=None,
            )
            elapsed = time.time() - started

            total_in = sum(u["input_tokens"] for u in model.usage_log)
            total_out = sum(u["output_tokens"] for u in model.usage_log)

            result.update({
                "steps": state.step,
                "done": state.done,
                "delta_ratio": round(state.delta_ratio, 3),
                "new_page_count": state.new_page_count,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "n_api_calls": len(model.usage_log),
                "wall_time_sec": round(elapsed, 1),
                "usage_log": model.usage_log,
                "transition_log": state.transition_log,
            })
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["partial_usage"] = model.usage_log
            log.exception("trial failed")
        finally:
            await browser.close()

    return result


def summarize(results: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if "error" in r:
            continue
        k = (r["task"], r["config"])
        groups.setdefault(k, []).append(r)

    summary: dict = {}
    for (task, cfg), rs in groups.items():
        steps = [r["steps"] for r in rs]
        in_tok = [r["total_input_tokens"] for r in rs]
        out_tok = [r["total_output_tokens"] for r in rs]
        wall = [r["wall_time_sec"] for r in rs]
        dones = [r["done"] for r in rs]

        def mean_std(xs):
            return {
                "mean": round(statistics.mean(xs), 1),
                "stdev": round(statistics.stdev(xs) if len(xs) > 1 else 0, 1),
                "min": round(min(xs), 1),
                "max": round(max(xs), 1),
                "values": xs,
            }

        summary[f"{task}/{cfg}"] = {
            "n_trials": len(rs),
            "success_rate": f"{sum(dones)}/{len(rs)}",
            "steps": mean_std(steps),
            "input_tokens": mean_std(in_tok),
            "output_tokens": mean_std(out_tok),
            "wall_time_sec": mean_std(wall),
        }

    # Compute DV vs FF deltas per task
    by_task: dict = {}
    for task_name, _ in groups.keys():
        dv = summary.get(f"{task_name}/DV")
        ff = summary.get(f"{task_name}/FF")
        if dv and ff and dv["input_tokens"]["mean"] and ff["input_tokens"]["mean"]:
            by_task[task_name] = {
                "token_savings_pct": round(
                    (ff["input_tokens"]["mean"] - dv["input_tokens"]["mean"])
                    / ff["input_tokens"]["mean"] * 100, 1
                ),
                "step_reduction_pct": round(
                    (ff["steps"]["mean"] - dv["steps"]["mean"])
                    / ff["steps"]["mean"] * 100, 1
                ) if ff["steps"]["mean"] else 0,
                "wall_time_change_pct": round(
                    (ff["wall_time_sec"]["mean"] - dv["wall_time_sec"]["mean"])
                    / ff["wall_time_sec"]["mean"] * 100, 1
                ) if ff["wall_time_sec"]["mean"] else 0,
            }
    summary["_per_task_delta"] = by_task
    return summary


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    out_path = REPO / "benchmarks" / "headtohead" / "head_to_head_results.json"
    all_results: list[dict] = []

    total = len(TASKS) * 2 * N_TRIALS
    done = 0

    for task in TASKS:
        for dv_enabled in [True, False]:
            cfg_label = "DV" if dv_enabled else "FF"
            for trial in range(N_TRIALS):
                done += 1
                log.info("[%d/%d] %s config=%s trial=%d starting",
                         done, total, task["name"], cfg_label, trial)
                r = await run_trial(task, dv_enabled, trial)
                if "error" in r:
                    log.warning("  ERROR: %s", r["error"])
                else:
                    log.info("  done: steps=%d success=%s tokens_in=%d wall=%.1fs",
                             r["steps"], r["done"], r["total_input_tokens"], r["wall_time_sec"])
                all_results.append(r)

                # Save after each trial in case of crash
                payload = {
                    "framing": (
                        "Head-to-head DV-wrapped vs FF-baseline Claude agent. "
                        "Same task, same model, different observation pipeline. "
                        "Measures UTILITY (does DV agent still succeed?) and TOKEN COST with "
                        "REAL agent variance — complements the matched-trajectory 55.6% and "
                        "scripted 77.2% compression numbers."
                    ),
                    "model": "claude-sonnet-4-20250514",
                    "n_trials_per_config": N_TRIALS,
                    "results": all_results,
                    "summary": summarize(all_results),
                }
                out_path.write_text(json.dumps(payload, indent=2, default=str))

    log.info("DONE. Results: %s", out_path)
    print(json.dumps(summarize(all_results), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
