"""
WebVoyager-subset DeltaVision observation benchmark.

Takes 10 real websites from the WebVoyager benchmark (community standard for
browser-agent evaluation, 643 tasks across 15 sites) and runs a fixed
exploration script on each: navigate + scroll + simulate typical agent
interaction. Measures the DeltaVisionObserver's token savings against a
full-frame baseline using the Anthropic tool_result format.

Honest framing: this measures **observation-layer cost**, not task
completion. We don't run an LLM agent here — we use scripted actions
that mimic what a CU agent's browsing pattern looks like, so the results
are deterministic and reproducible.

For a full WebVoyager task-completion eval with DV wrapping a real agent,
see the `examples/browser_use_integration/` and `examples/openclaw_integration/`
adapter docs — those drop DV into a real autonomous agent.

Run:
    python examples/webvoyager_subset.py
"""

import asyncio
import base64
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from playwright.async_api import async_playwright

from observer import DeltaVisionObserver


# 10 diverse WebVoyager sites, hand-picked for breadth (static / SPA / media-rich
# / scroll-heavy / form-driven).
WEBVOYAGER_SUBSET = [
    {"name": "arxiv",       "url": "https://arxiv.org/list/cs.AI/recent"},
    {"name": "bbc_news",    "url": "https://www.bbc.com/news"},
    {"name": "cambridge",   "url": "https://dictionary.cambridge.org/dictionary/english/"},
    {"name": "coursera",    "url": "https://www.coursera.org/"},
    {"name": "espn",        "url": "https://www.espn.com/"},
    {"name": "github",      "url": "https://github.com/trending"},
    {"name": "huggingface", "url": "https://huggingface.co/models"},
    {"name": "apple",       "url": "https://www.apple.com/"},
    {"name": "wolfram",     "url": "https://www.wolframalpha.com/"},
    {"name": "allrecipes",  "url": "https://www.allrecipes.com/"},
]


# A generic "agent browsing" script — same on every site so results are comparable
EXPLORATION_STEPS = [
    {"kind": "scroll", "direction": "down", "pixels": 500},
    {"kind": "scroll", "direction": "down", "pixels": 500},
    {"kind": "scroll", "direction": "down", "pixels": 500},
    {"kind": "scroll", "direction": "up",   "pixels": 300},
    {"kind": "wait",   "ms": 500},
    {"kind": "scroll", "direction": "down", "pixels": 800},
]


# =========================================================== measurement

def _image_tokens(img: Image.Image) -> int:
    return max(75, int((img.width * img.height) / 750))


def _count_anthropic_payload(content: list[dict]) -> tuple[int, int]:
    """Return (b64_bytes, estimated_tokens) for an Anthropic tool_result content list."""
    b64_total = 0
    tokens = 0
    for block in content:
        if block.get("type") == "image":
            data = block["source"]["data"]
            b64_total += len(data)
            try:
                img = Image.open(io.BytesIO(base64.b64decode(data)))
                tokens += _image_tokens(img)
            except Exception:
                tokens += 1600
        elif block.get("type") == "text":
            tokens += max(1, len(block["text"]) // 4)
    return b64_total, tokens


async def _execute(page, action):
    kind = action["kind"]
    if kind == "scroll":
        dy = action["pixels"] * (-1 if action["direction"] == "up" else 1)
        await page.mouse.wheel(0, dy)
    elif kind == "wait":
        await asyncio.sleep(action["ms"] / 1000)


def _action_str(a):
    if a["kind"] == "scroll":
        return f'scroll({a["direction"]},{a["pixels"]}px)'
    if a["kind"] == "wait":
        return f'wait({a["ms"]}ms)'
    return a["kind"]


async def run_site(page, site):
    observer = DeltaVisionObserver()
    steps_out = []
    ff_total = 0
    dv_total = 0
    ff_tok_total = 0
    dv_tok_total = 0

    print(f"\n== {site['name']}  [{site['url']}] ==")
    try:
        await page.goto(site["url"], timeout=25000)
    except Exception as e:
        print(f"  navigation failed: {e}")
        return None
    try:
        await page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        await page.wait_for_load_state("domcontentloaded")

    # Accept cookies if a generic banner is visible (best-effort, silent failure)
    for sel in ["button:has-text('Accept')", "button:has-text('I Accept')",
                "button:has-text('Agree')", "[aria-label='Accept cookies']"]:
        try:
            await page.locator(sel).first.click(timeout=1500)
            break
        except Exception:
            continue

    # Step 0 — initial capture
    png = await page.screenshot(type="png", full_page=False)
    b64 = base64.standard_b64encode(png).decode()
    ff_b, ff_t = _count_anthropic_payload([{"type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64}}])

    obs = observer.observe(png, url=page.url)
    dv_b, dv_t = _count_anthropic_payload(obs.to_anthropic_tool_result_content())

    steps_out.append({
        "step": 0, "action": "navigate", "obs_type": obs.obs_type,
        "trigger": obs.trigger, "diff_ratio": obs.diff_ratio,
        "phash": obs.phash_distance, "anchor": obs.anchor_score,
        "ff_bytes": ff_b, "dv_bytes": dv_b,
        "ff_tokens": ff_t, "dv_tokens": dv_t,
    })
    ff_total += ff_b; dv_total += dv_b
    ff_tok_total += ff_t; dv_tok_total += dv_t
    print(f"  0 navigate                       ff={ff_b//1024}KB dv={dv_b//1024}KB")

    for i, action in enumerate(EXPLORATION_STEPS, start=1):
        try:
            await _execute(page, action)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"  {i} exec error: {e}")
            continue

        png = await page.screenshot(type="png", full_page=False)
        b64 = base64.standard_b64encode(png).decode()
        ff_b, ff_t = _count_anthropic_payload([{"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64}}])

        obs = observer.observe(png, url=page.url, last_action=_action_str(action))
        dv_b, dv_t = _count_anthropic_payload(obs.to_anthropic_tool_result_content())

        steps_out.append({
            "step": i, "action": _action_str(action), "obs_type": obs.obs_type,
            "trigger": obs.trigger, "diff_ratio": obs.diff_ratio,
            "phash": obs.phash_distance, "anchor": obs.anchor_score,
            "ff_bytes": ff_b, "dv_bytes": dv_b,
            "ff_tokens": ff_t, "dv_tokens": dv_t,
        })
        ff_total += ff_b; dv_total += dv_b
        ff_tok_total += ff_t; dv_tok_total += dv_t
        tag = "D" if obs.obs_type == "delta" else "N"
        print(f"  {i} {_action_str(action):<28} {tag} trig={obs.trigger:<14} "
              f"diff={obs.diff_ratio:.3f} ff={ff_b//1024}KB dv={dv_b//1024}KB")

    return {
        "name": site["name"], "url": site["url"],
        "n_steps": len(steps_out),
        "ff_total_bytes": ff_total, "dv_total_bytes": dv_total,
        "ff_total_tokens": ff_tok_total, "dv_total_tokens": dv_tok_total,
        "byte_savings_pct": round((ff_total - dv_total) / max(1, ff_total) * 100, 1),
        "token_savings_pct": round((ff_tok_total - dv_tok_total) / max(1, ff_tok_total) * 100, 1),
        "steps": steps_out,
    }


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--window-size=1280,900"])
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"),
        )
        results = []
        for site in WEBVOYAGER_SUBSET:
            page = await context.new_page()
            try:
                r = await run_site(page, site)
                if r:
                    results.append(r)
            except Exception as e:
                print(f"  hard failure on {site['name']}: {e}")
            finally:
                await page.close()

        await browser.close()

    # Aggregate
    total_ff_b = sum(r["ff_total_bytes"] for r in results)
    total_dv_b = sum(r["dv_total_bytes"] for r in results)
    total_ff_t = sum(r["ff_total_tokens"] for r in results)
    total_dv_t = sum(r["dv_total_tokens"] for r in results)
    total_steps = sum(r["n_steps"] for r in results)

    total_bp = (total_ff_b - total_dv_b) / max(1, total_ff_b) * 100
    total_tp = (total_ff_t - total_dv_t) / max(1, total_ff_t) * 100

    print("\n" + "=" * 96)
    print(f"{'site':<20} {'URL':<42} {'steps':>5} {'FF KB':>7} {'DV KB':>7} {'bytes%':>7} {'tokens%':>8}")
    print("-" * 96)
    for r in results:
        print(f"{r['name']:<20} {r['url'][:41]:<42} {r['n_steps']:>5d} "
              f"{r['ff_total_bytes']//1024:>7d} {r['dv_total_bytes']//1024:>7d} "
              f"{r['byte_savings_pct']:>6.1f}% {r['token_savings_pct']:>7.1f}%")
    print("-" * 96)
    print(f"{'TOTAL':<20} {f'{len(results)} / 10 WebVoyager sites':<42} {total_steps:>5d} "
          f"{total_ff_b//1024:>7d} {total_dv_b//1024:>7d} "
          f"{total_bp:>6.1f}% {total_tp:>7.1f}%")
    print("=" * 96)

    out = {
        "framing": "WebVoyager-subset observation-cost benchmark (not task completion)",
        "summary": {
            "n_sites": len(results),
            "n_steps_total": total_steps,
            "ff_total_bytes": total_ff_b, "dv_total_bytes": total_dv_b,
            "ff_total_tokens": total_ff_t, "dv_total_tokens": total_dv_t,
            "byte_savings_pct": round(total_bp, 1),
            "token_savings_pct": round(total_tp, 1),
        },
        "sites": results,
    }
    Path("examples/webvoyager_subset_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote examples/webvoyager_subset_results.json")


if __name__ == "__main__":
    asyncio.run(main())
