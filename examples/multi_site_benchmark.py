"""
Multi-site integration benchmark.

Drives Playwright through N task sequences across 5+ different sites. For
each step, measures what an Anthropic-format `tool_result` payload would
weigh (base64 image bytes + estimated image tokens) in both:

  (a) Baseline: full screenshot every step
  (b) With DeltaVisionObserver: thumbnail + crops on DELTA, full on NEW_PAGE

This extends `observer_integration_proof.py` from a single TodoMVC task to
a multi-site suite. Output: `examples/multi_site_results.json` + a per-task
Markdown table.

Tasks chosen to exercise different transition patterns:
  - Wikipedia: real URL-level navigation → NEW_PAGE heavy
  - TodoMVC: pure SPA → DELTA heavy
  - Hacker News: sticky nav, article navigation
  - example.com: minimal baseline
  - GitHub issues: mixed

Run:
    python examples/multi_site_benchmark.py
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


# =========================================================== task definitions

def _action(kind, **kw):
    return {"kind": kind, **kw}


TASKS = [
    {
        "name": "todomvc_add_three",
        "site": "TodoMVC (React SPA)",
        "url": "https://todomvc.com/examples/react/dist/",
        "bootstrap": [_action("click", selector=".new-todo")],
        "steps": [
            _action("type", text="buy groceries"),
            _action("key", key="Enter"),
            _action("type", text="write report"),
            _action("key", key="Enter"),
            _action("type", text="review PR"),
            _action("key", key="Enter"),
            _action("click", selector="ul.todo-list li:nth-child(1) input.toggle"),
            _action("click", selector="ul.filters li:nth-child(2) a"),
        ],
    },
    {
        "name": "todomvc_delete_and_clear",
        "site": "TodoMVC (React SPA)",
        "url": "https://todomvc.com/examples/react/dist/",
        "bootstrap": [_action("click", selector=".new-todo")],
        "steps": [
            _action("type", text="first"),
            _action("key", key="Enter"),
            _action("type", text="second"),
            _action("key", key="Enter"),
            _action("type", text="third"),
            _action("key", key="Enter"),
            # mark first two complete
            _action("click", selector="ul.todo-list li:nth-child(1) input.toggle"),
            _action("click", selector="ul.todo-list li:nth-child(2) input.toggle"),
            _action("click", selector="button.clear-completed"),
        ],
    },
    {
        "name": "wikipedia_search_navigate",
        "site": "Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "bootstrap": [],
        "steps": [
            _action("click", selector="input#searchInput"),
            _action("type", text="machine learning"),
            _action("key", key="Enter"),
            # Wait for search results then click first result
            _action("wait", ms=1500),
            _action("click", selector="div.mw-search-result-heading a, li.mw-search-result a"),
            _action("wait", ms=1000),
            # Scroll down to exercise scroll-bypass
            _action("scroll", direction="down", pixels=600),
            _action("scroll", direction="down", pixels=600),
        ],
    },
    {
        "name": "hackernews_browse",
        "site": "Hacker News",
        "url": "https://news.ycombinator.com/",
        "bootstrap": [],
        "steps": [
            _action("scroll", direction="down", pixels=400),
            _action("scroll", direction="down", pixels=400),
            _action("scroll", direction="up", pixels=200),
            # click on first story comments link
            _action("click", selector="span.subline a:has-text('comments'), span.subline a:has-text('discuss')"),
            _action("wait", ms=1000),
            _action("scroll", direction="down", pixels=800),
        ],
    },
    {
        "name": "example_com_idle",
        "site": "example.com (static baseline)",
        "url": "https://example.com",
        "bootstrap": [],
        "steps": [
            _action("wait", ms=500),
            _action("wait", ms=500),
            _action("scroll", direction="down", pixels=100),
            _action("wait", ms=500),
        ],
    },
]


# =========================================================== measurement

@dataclass
class StepMeasure:
    step: int
    action: str
    obs_type: str
    trigger: str
    diff_ratio: float
    phash: int
    anchor: float
    num_images: int
    ff_bytes: int
    dv_bytes: int
    ff_tokens: int
    dv_tokens: int


@dataclass
class TaskResult:
    name: str
    site: str
    url: str
    steps: list[StepMeasure] = field(default_factory=list)
    ff_total_bytes: int = 0
    dv_total_bytes: int = 0
    ff_total_tokens: int = 0
    dv_total_tokens: int = 0
    elapsed_s: float = 0.0

    def to_dict(self):
        return {
            "name": self.name, "site": self.site, "url": self.url,
            "elapsed_s": round(self.elapsed_s, 2),
            "ff_total_bytes": self.ff_total_bytes,
            "dv_total_bytes": self.dv_total_bytes,
            "ff_total_tokens": self.ff_total_tokens,
            "dv_total_tokens": self.dv_total_tokens,
            "byte_savings_pct": round((self.ff_total_bytes - self.dv_total_bytes) / max(1, self.ff_total_bytes) * 100, 1),
            "token_savings_pct": round((self.ff_total_tokens - self.dv_total_tokens) / max(1, self.ff_total_tokens) * 100, 1),
            "steps": [vars(s) for s in self.steps],
        }


def _image_tokens(img: Image.Image) -> int:
    return max(75, int((img.width * img.height) / 750))


def _count_anthropic_payload(content: list[dict]) -> tuple[int, int, int]:
    """Return (b64_bytes, num_images, estimated_tokens) for an Anthropic
    tool_result content list."""
    b64_total = 0
    tokens = 0
    n = 0
    for block in content:
        if block.get("type") == "image":
            data = block["source"]["data"]
            b64_total += len(data)
            n += 1
            try:
                img = Image.open(io.BytesIO(base64.b64decode(data)))
                tokens += _image_tokens(img)
            except Exception:
                tokens += 1600
        elif block.get("type") == "text":
            tokens += max(1, len(block["text"]) // 4)
    return b64_total, n, tokens


# =========================================================== execution

async def _execute(page, action, timeout=5000):
    kind = action["kind"]
    if kind == "click":
        try:
            await page.locator(action["selector"]).first.click(timeout=timeout)
        except Exception as e:
            print(f"    click skipped: {e}")
    elif kind == "type":
        await page.keyboard.type(action["text"], delay=25)
    elif kind == "key":
        await page.keyboard.press(action["key"])
    elif kind == "scroll":
        dy = action["pixels"] * (-1 if action["direction"] == "up" else 1)
        await page.mouse.wheel(0, dy)
    elif kind == "wait":
        await asyncio.sleep(action["ms"] / 1000)


def _action_str(a):
    kind = a["kind"]
    if kind == "type":
        return f'type({a["text"][:20]!r})'
    if kind == "click":
        return f'click({a["selector"][:30]})'
    if kind == "scroll":
        return f'scroll({a["direction"]},{a["pixels"]}px)'
    if kind == "key":
        return f'key({a["key"]})'
    if kind == "wait":
        return f'wait({a["ms"]}ms)'
    return kind


async def run_task(page, task: dict) -> TaskResult:
    observer = DeltaVisionObserver()
    result = TaskResult(name=task["name"], site=task["site"], url=task["url"])
    t_start = time.time()

    print(f"\n== {task['name']}  [{task['site']}] ==")
    await page.goto(task["url"])
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        await page.wait_for_load_state("domcontentloaded")

    for boot in task["bootstrap"]:
        await _execute(page, boot)

    # Step 0 — initial capture (both paths identical)
    png = await page.screenshot(type="png")
    b64 = base64.standard_b64encode(png).decode()
    ff_content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]
    ff_bytes, _, ff_tok = _count_anthropic_payload(ff_content)

    obs = observer.observe(png, url=page.url)
    dv_content = obs.to_anthropic_tool_result_content()
    dv_bytes, n_dv, dv_tok = _count_anthropic_payload(dv_content)

    sm = StepMeasure(
        step=0, action="navigate", obs_type=obs.obs_type, trigger=obs.trigger,
        diff_ratio=obs.diff_ratio, phash=obs.phash_distance, anchor=obs.anchor_score,
        num_images=n_dv, ff_bytes=ff_bytes, dv_bytes=dv_bytes,
        ff_tokens=ff_tok, dv_tokens=dv_tok,
    )
    result.steps.append(sm)
    result.ff_total_bytes += ff_bytes
    result.dv_total_bytes += dv_bytes
    result.ff_total_tokens += ff_tok
    result.dv_total_tokens += dv_tok
    print(f"  {0:2d} {'navigate':<30} {obs.obs_type:<10} ff={ff_bytes//1024}KB dv={dv_bytes//1024}KB")

    for i, action in enumerate(task["steps"], start=1):
        try:
            await _execute(page, action)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"    step {i} execute error: {e} — continuing")
            continue

        png = await page.screenshot(type="png")
        b64 = base64.standard_b64encode(png).decode()
        ff_content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]
        ff_bytes, _, ff_tok = _count_anthropic_payload(ff_content)

        obs = observer.observe(png, url=page.url, last_action=_action_str(action))
        dv_content = obs.to_anthropic_tool_result_content()
        dv_bytes, n_dv, dv_tok = _count_anthropic_payload(dv_content)

        sm = StepMeasure(
            step=i, action=_action_str(action), obs_type=obs.obs_type,
            trigger=obs.trigger, diff_ratio=obs.diff_ratio,
            phash=obs.phash_distance, anchor=obs.anchor_score,
            num_images=n_dv, ff_bytes=ff_bytes, dv_bytes=dv_bytes,
            ff_tokens=ff_tok, dv_tokens=dv_tok,
        )
        result.steps.append(sm)
        result.ff_total_bytes += ff_bytes
        result.dv_total_bytes += dv_bytes
        result.ff_total_tokens += ff_tok
        result.dv_total_tokens += dv_tok
        tag = "D" if obs.obs_type == "delta" else "N"
        print(f"  {i:2d} {_action_str(action):<30} {tag}  trig={obs.trigger:<12} diff={obs.diff_ratio:.3f} ff={ff_bytes//1024}KB dv={dv_bytes//1024}KB")

    result.elapsed_s = time.time() - t_start
    print(f"  totals: ff={result.ff_total_bytes//1024}KB dv={result.dv_total_bytes//1024}KB "
          f"bytes-saved={100*(result.ff_total_bytes - result.dv_total_bytes)/max(1,result.ff_total_bytes):.1f}% "
          f"tokens-saved={100*(result.ff_total_tokens - result.dv_total_tokens)/max(1,result.ff_total_tokens):.1f}%")
    return result


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--window-size=1280,900"])
        context = await browser.new_context(viewport={"width": 1280, "height": 900})

        results = []
        for task in TASKS:
            page = await context.new_page()
            try:
                tr = await run_task(page, task)
                results.append(tr)
            except Exception as e:
                print(f"  task {task['name']} hard-failed: {e}")
            finally:
                await page.close()

        await browser.close()

    # Aggregate
    total_ff_b = sum(r.ff_total_bytes for r in results)
    total_dv_b = sum(r.dv_total_bytes for r in results)
    total_ff_t = sum(r.ff_total_tokens for r in results)
    total_dv_t = sum(r.dv_total_tokens for r in results)
    total_steps = sum(len(r.steps) for r in results)

    print("\n" + "=" * 96)
    print(f"{'task':<30} {'site':<26} {'steps':>5} {'FF KB':>7} {'DV KB':>7} {'bytes%':>7} {'tokens%':>8}")
    print("-" * 96)
    for r in results:
        bp = (r.ff_total_bytes - r.dv_total_bytes) / max(1, r.ff_total_bytes) * 100
        tp = (r.ff_total_tokens - r.dv_total_tokens) / max(1, r.ff_total_tokens) * 100
        print(f"{r.name:<30} {r.site[:25]:<26} {len(r.steps):>5d} "
              f"{r.ff_total_bytes//1024:>7d} {r.dv_total_bytes//1024:>7d} "
              f"{bp:>6.1f}% {tp:>7.1f}%")
    print("-" * 96)
    total_bp = (total_ff_b - total_dv_b) / max(1, total_ff_b) * 100
    total_tp = (total_ff_t - total_dv_t) / max(1, total_ff_t) * 100
    print(f"{'TOTAL':<30} {f'{len(results)} tasks':<26} {total_steps:>5d} "
          f"{total_ff_b//1024:>7d} {total_dv_b//1024:>7d} "
          f"{total_bp:>6.1f}% {total_tp:>7.1f}%")
    print("=" * 96)

    # Write JSON
    out = {
        "summary": {
            "n_tasks": len(results),
            "n_steps_total": total_steps,
            "ff_total_bytes": total_ff_b,
            "dv_total_bytes": total_dv_b,
            "ff_total_tokens": total_ff_t,
            "dv_total_tokens": total_dv_t,
            "byte_savings_pct": round(total_bp, 1),
            "token_savings_pct": round(total_tp, 1),
        },
        "tasks": [r.to_dict() for r in results],
    }
    Path("examples/multi_site_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote examples/multi_site_results.json")


if __name__ == "__main__":
    asyncio.run(main())
