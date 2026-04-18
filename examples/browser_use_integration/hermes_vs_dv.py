"""
Hermes-class local VLM + Browser Use, WITH vs WITHOUT DeltaVision.

Model: UI-TARS-1.5-7B (ByteDance, GUI-specialized — closest local-OSS equivalent
to a Hermes-family CU model on David's 5080 right now).
Framework: Browser Use 0.12 (88k stars, the OSS CU baseline)
Task: fixed TodoMVC sequence, same on both runs.

Wires the DeltaVisionObserver into Browser Use via a ONE-FUNCTION monkey-patch
on BrowserSession.get_browser_state_summary. Zero changes to Browser Use.

Measures wall clock, screenshot payload bytes, and task completion.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Let the script find the parent deltavision package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from browser_use import Agent
from browser_use.browser.session import BrowserSession

# IMPORTANT: we use ChatOpenAI pointed at Ollama's OpenAI-compat /v1 endpoint
# instead of Browser Use's native ChatOllama. Native ChatOllama crashes
# Ollama's model runner (HTTP 500) on Browser Use's large state payloads —
# orthogonal to DeltaVision. The /v1 path works reliably.
from browser_use.llm.openai.chat import ChatOpenAI

from observer import DeltaVisionObserver

# ------------------------------------------------------------ config

MODEL = "qwen2.5vl:7b"
OLLAMA_OPENAI_ENDPOINT = "http://127.0.0.1:11434/v1"  # SSH tunnel to Windows 5080

TASK = (
    "Go to https://todomvc.com/examples/react/dist/ and add a single todo "
    "that says 'hello from deltavision'. Then report done."
)

MAX_STEPS = 8


# ------------------------------------------------------------ monkey-patch

def install_deltavision_patch(observer: DeltaVisionObserver) -> callable:
    """
    Monkey-patch BrowserSession.get_browser_state_summary so its returned
    `screenshot` field is the Observer-packaged composite instead of the raw
    full frame. Returns an uninstall function.
    """
    original = BrowserSession.get_browser_state_summary
    payload_log: list[dict] = []

    async def patched(self, *args, **kwargs):
        summary = await original(self, *args, **kwargs)
        if summary.screenshot:
            # summary.screenshot is a base64 PNG string
            url = summary.url
            obs = observer.observe(summary.screenshot, url=url)
            new_b64 = obs.to_browser_use_screenshot_b64()
            # Record what we sent and the savings
            orig_bytes = len(summary.screenshot)
            new_bytes = len(new_b64)
            payload_log.append({
                "url": url,
                "orig_b64_bytes": orig_bytes,
                "dv_b64_bytes": new_bytes,
                "obs_type": obs.obs_type,
                "trigger": obs.trigger,
                "diff_ratio": obs.diff_ratio,
                "phash": obs.phash_distance,
            })
            # Mutate the summary in place
            summary.screenshot = new_b64
        return summary

    BrowserSession.get_browser_state_summary = patched

    def uninstall():
        BrowserSession.get_browser_state_summary = original

    uninstall.log = payload_log
    return uninstall


def install_payload_sniffer() -> callable:
    """
    Sniff screenshot bytes without altering them (baseline measurement).
    Same hook point — just records sizes.
    """
    original = BrowserSession.get_browser_state_summary
    payload_log: list[dict] = []

    async def patched(self, *args, **kwargs):
        summary = await original(self, *args, **kwargs)
        if summary.screenshot:
            payload_log.append({
                "url": summary.url,
                "b64_bytes": len(summary.screenshot),
            })
        return summary

    BrowserSession.get_browser_state_summary = patched

    def uninstall():
        BrowserSession.get_browser_state_summary = original

    uninstall.log = payload_log
    return uninstall


# ------------------------------------------------------------ runs

async def run_once(label: str, install_hook) -> dict:
    print(f"\n--- RUN: {label} ---")
    uninstall = install_hook()
    llm = ChatOpenAI(
        model=MODEL,
        base_url=OLLAMA_OPENAI_ENDPOINT,
        api_key="ollama-no-auth",
        temperature=0.1,
        max_completion_tokens=512,   # keep response small — the model's
        reasoning_effort="none",       # thinking should be quick
        dont_force_structured_output=True,
        add_schema_to_system_prompt=True,
        timeout=300.0,                 # qwen on a 5080 via tunnel can take a minute
    )

    t0 = time.time()
    agent = Agent(task=TASK, llm=llm)
    try:
        result = await agent.run(max_steps=MAX_STEPS)
        success = bool(getattr(result, "is_done", lambda: False)())
    except Exception as e:
        print(f"  run error: {e}")
        success = False
    elapsed = time.time() - t0

    uninstall()

    sum_bytes = 0
    for entry in uninstall.log:
        if "dv_b64_bytes" in entry:
            sum_bytes += entry["dv_b64_bytes"]
        else:
            sum_bytes += entry["b64_bytes"]

    return {
        "label": label,
        "success": success,
        "elapsed_s": elapsed,
        "n_observations": len(uninstall.log),
        "total_b64_bytes": sum_bytes,
        "log": uninstall.log,
    }


async def main():
    print(f"Model:    {MODEL}")
    print(f"Endpoint: {OLLAMA_OPENAI_ENDPOINT}")
    print(f"Task:    {TASK}")

    # 1. Baseline: plain Browser Use, just sniff payload sizes
    baseline = await run_once("BASELINE (full-frame)", install_payload_sniffer)

    # 2. DeltaVision: install observer, same task
    observer = DeltaVisionObserver()
    dv_run = await run_once(
        "WITH DeltaVisionObserver",
        lambda: install_deltavision_patch(observer),
    )

    # Report
    print("\n" + "=" * 80)
    print(f"{'':>4} {'label':<30} {'steps':>6} {'time':>8} {'payload':>11} {'success'}")
    print("-" * 80)
    for run in (baseline, dv_run):
        print(f"{'':>4} {run['label']:<30} {run['n_observations']:>6d} "
              f"{run['elapsed_s']:>7.1f}s {run['total_b64_bytes']/1024:>9.1f} KB   "
              f"{run['success']}")
    if baseline["total_b64_bytes"]:
        saved = baseline["total_b64_bytes"] - dv_run["total_b64_bytes"]
        pct = saved / baseline["total_b64_bytes"] * 100
        print()
        print(f"{'':>4} Payload savings: {saved/1024:.1f} KB  ({pct:.1f}%)")
    print("=" * 80)

    # Save artifacts
    out = {
        "model": MODEL,
        "task": TASK,
        "baseline": baseline,
        "with_deltavision": dv_run,
    }
    Path("hermes_vs_dv_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved results to hermes_vs_dv_results.json")


if __name__ == "__main__":
    asyncio.run(main())
