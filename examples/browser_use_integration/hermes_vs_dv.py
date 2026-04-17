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
import base64
import io
import json
import sys
import time
from pathlib import Path
from types import MethodType

# Let the script find the parent deltavision package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from browser_use import Agent
from browser_use.browser.session import BrowserSession
from browser_use.llm.ollama.chat import ChatOllama

from observer import DeltaVisionObserver


# ------------------------------------------------------------ config

# UI-TARS-1.5-7B is F16 = 15GB, tight fit in the 5080 Laptop's 16GB VRAM.
# qwen2.5vl:7b Q4_K_M is 6GB and also GUI-capable — verified working earlier
# in this session. Swap here if you want to try UI-TARS after clearing VRAM.
MODEL = "qwen2.5vl:7b"
OLLAMA_HOST = "http://127.0.0.1:11434"  # SSH tunnel to Windows 5080

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
    llm = ChatOllama(model=MODEL, host=OLLAMA_HOST, timeout=180.0)

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
    print(f"Model: {MODEL}")
    print(f"Ollama:  {OLLAMA_HOST}")
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
    print(f"\nSaved results to hermes_vs_dv_results.json")


if __name__ == "__main__":
    asyncio.run(main())
