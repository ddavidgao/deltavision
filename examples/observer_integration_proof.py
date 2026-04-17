"""
Integration proof: side-by-side comparison of a CU agent loop WITH and
WITHOUT DeltaVisionObserver, using the new public API.

Drives Playwright through a fixed TodoMVC action sequence twice — once as a
vanilla full-frame agent, once wrapped in `DeltaVisionObserver`. Measures
the real payload size of the tool_result that would be sent to Anthropic's
Messages API at each step.

Why this is the proof:
  - Same browser, same page, same actions
  - Same message format (Anthropic tool_result blocks)
  - The ONLY difference is whether screenshots go through the Observer
  - Result: measured base64 byte count per step, summed across the run

Run:
    python examples/observer_integration_proof.py
"""

import asyncio
import base64
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.async_api import async_playwright

from observer import DeltaVisionObserver


# The script we'll run through twice.
SEQUENCE = [
    ("type", "buy groceries"),
    ("key", "Enter"),
    ("type", "write report"),
    ("key", "Enter"),
    ("type", "review PR"),
    ("key", "Enter"),
    ("click", "ul.todo-list li:nth-child(1) input.toggle"),
    ("click", "ul.filters li:nth-child(2) a"),
]


@dataclass
class StepPayload:
    step: int
    action: str
    obs_type: str      # "full_frame" | "delta" | "ff_baseline"
    trigger: str
    diff_ratio: float
    phash: int
    num_images: int
    payload_bytes: int
    estimated_tokens: int


def _b64_png_size(img) -> tuple[int, int]:
    """Returns (raw_png_bytes, base64_char_count) — base64 is what gets sent."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = len(buf.getvalue())
    b64 = len(base64.standard_b64encode(buf.getvalue()))
    return raw, b64


def _image_tokens(img) -> int:
    """Same formula the Observer uses; matches Anthropic's rough pricing."""
    return max(75, int((img.width * img.height) / 750))


def count_payload_bytes(content: list[dict]) -> tuple[int, int, int]:
    """
    Total base64 chars in an Anthropic tool_result content list.
    Returns (total_b64_bytes, num_image_blocks, estimated_tokens).
    """
    b64_total = 0
    tokens = 0
    n_imgs = 0
    for block in content:
        if block.get("type") == "image":
            data = block["source"]["data"]
            b64_total += len(data)
            n_imgs += 1
            # Decode to measure w/h for token estimate
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(base64.b64decode(data)))
                tokens += _image_tokens(img)
            except Exception:
                tokens += 1600
        elif block.get("type") == "text":
            tokens += max(1, len(block["text"]) // 4)
    return b64_total, n_imgs, tokens


async def _execute_action(page, action_type, arg):
    if action_type == "type":
        await page.keyboard.type(arg, delay=30)
    elif action_type == "key":
        await page.keyboard.press(arg)
    elif action_type == "click":
        await page.locator(arg).click()
    await asyncio.sleep(0.5)


async def _capture_png_bytes(page) -> bytes:
    return await page.screenshot(type="png")


async def run_baseline(page) -> list[StepPayload]:
    """Full-frame baseline: every step, we'd send the full screenshot."""
    results: list[StepPayload] = []

    # Step 0: initial
    png = await _capture_png_bytes(page)
    content = [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": base64.standard_b64encode(png).decode()},
    }]
    b64_total, n_imgs, toks = count_payload_bytes(content)
    results.append(StepPayload(0, "navigate", "ff_baseline", "initial",
                                0.0, 0, n_imgs, b64_total, toks))

    for i, (atype, arg) in enumerate(SEQUENCE, start=1):
        await _execute_action(page, atype, arg)
        png = await _capture_png_bytes(page)
        content = [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(png).decode()},
        }]
        b64_total, n_imgs, toks = count_payload_bytes(content)
        results.append(StepPayload(i, f"{atype}({arg})", "ff_baseline",
                                    "full_frame", 0.0, 0, n_imgs, b64_total, toks))

    return results


async def run_with_observer(page) -> list[StepPayload]:
    """Same actions, but wrap every capture in DeltaVisionObserver."""
    observer = DeltaVisionObserver()
    results: list[StepPayload] = []

    # Step 0
    png = await _capture_png_bytes(page)
    url = page.url
    obs = observer.observe(png, url=url, last_action=None)
    content = obs.to_anthropic_tool_result_content()
    b64_total, n_imgs, toks = count_payload_bytes(content)
    results.append(StepPayload(0, "navigate", obs.obs_type, obs.trigger,
                                obs.diff_ratio, obs.phash_distance,
                                n_imgs, b64_total, toks))

    for i, (atype, arg) in enumerate(SEQUENCE, start=1):
        await _execute_action(page, atype, arg)
        png = await _capture_png_bytes(page)
        obs = observer.observe(png, url=page.url, last_action=f"{atype}({arg})")
        content = obs.to_anthropic_tool_result_content()
        b64_total, n_imgs, toks = count_payload_bytes(content)
        results.append(StepPayload(i, f"{atype}({arg})", obs.obs_type,
                                    obs.trigger, obs.diff_ratio,
                                    obs.phash_distance, n_imgs,
                                    b64_total, toks))

    return results


def print_table(title: str, rows: list[StepPayload]):
    print(f"\n{title}")
    print("-" * 96)
    print(f"{'step':>4} {'action':<30} {'obs_type':<12} {'trigger':<14} "
          f"{'diff':>7} {'pH':>3} {'imgs':>4} {'kb64':>6} {'tokens':>7}")
    print("-" * 96)
    total_b64 = 0
    total_tok = 0
    for r in rows:
        print(f"{r.step:>4} {r.action[:29]:<30} {r.obs_type:<12} {r.trigger:<14} "
              f"{r.diff_ratio:>7.3f} {r.phash:>3d} {r.num_images:>4d} "
              f"{r.payload_bytes/1024:>6.1f} {r.estimated_tokens:>7d}")
        total_b64 += r.payload_bytes
        total_tok += r.estimated_tokens
    print("-" * 96)
    print(f"{'':>4} {'TOTAL':<30} {'':<12} {'':<14} {'':>7} {'':>3} {'':>4} "
          f"{total_b64/1024:>6.1f} {total_tok:>7d}")
    return total_b64, total_tok


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,
            args=["--window-size=1280,900"])
        context = await browser.new_context(viewport={"width": 1280, "height": 900})

        # Baseline run
        page = await context.new_page()
        print("Running BASELINE (full-frame every step)...")
        await page.goto("https://todomvc.com/examples/react/dist/")
        await page.wait_for_load_state("networkidle", timeout=8000)
        await page.locator(".new-todo").click()
        baseline = await run_baseline(page)
        await page.close()

        # Observer run
        page = await context.new_page()
        print("Running with DELTAVISIONOBSERVER...")
        await page.goto("https://todomvc.com/examples/react/dist/")
        await page.wait_for_load_state("networkidle", timeout=8000)
        await page.locator(".new-todo").click()
        observed = await run_with_observer(page)
        await page.close()

        await browser.close()

    ff_b64, ff_tok = print_table("BASELINE (full-frame every step)", baseline)
    dv_b64, dv_tok = print_table("WITH DeltaVisionObserver", observed)

    print("\n" + "=" * 96)
    print(f"{'':>4} Anthropic tool_result payload size (base64 chars, what travels over the wire)")
    print(f"{'':>4} Baseline:    {ff_b64/1024:>7.1f} KB   ({ff_b64:,} chars)")
    print(f"{'':>4} Observer:    {dv_b64/1024:>7.1f} KB   ({dv_b64:,} chars)")
    print(f"{'':>4} Saved:       {(ff_b64-dv_b64)/1024:>7.1f} KB   "
          f"({(ff_b64-dv_b64)/ff_b64*100:>5.1f}%)")
    print()
    print(f"{'':>4} Estimated image tokens (matches Anthropic pricing)")
    print(f"{'':>4} Baseline:    {ff_tok:,} tokens")
    print(f"{'':>4} Observer:    {dv_tok:,} tokens")
    print(f"{'':>4} Saved:       {ff_tok-dv_tok:,} tokens "
          f"({(ff_tok-dv_tok)/ff_tok*100:.1f}%)")
    print("=" * 96)

    # Save the artifact so it can be linked from the README
    out = {
        "baseline": [r.__dict__ for r in baseline],
        "observer": [r.__dict__ for r in observed],
        "totals": {
            "baseline_bytes": ff_b64, "observer_bytes": dv_b64,
            "baseline_tokens": ff_tok, "observer_tokens": dv_tok,
            "byte_savings_pct": (ff_b64-dv_b64)/ff_b64*100,
            "token_savings_pct": (ff_tok-dv_tok)/ff_tok*100,
        },
    }
    Path("examples/observer_proof_results.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved to examples/observer_proof_results.json")


if __name__ == "__main__":
    asyncio.run(main())
