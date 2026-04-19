"""
agent_head_to_head.py — free-running Claude agent on a multi-tab apartment task.

Same task, same model, same tools, same browser. The ONLY thing that changes
between arms is how screenshots are delivered to Claude:

  --arm ff:   raw screenshot, native resolution (baseline)
  --arm dv:   DeltaVisionObserver wraps each screenshot (delta-first)

Agent free-runs until it calls `done` or hits max_steps. Captures video +
per-step metrics + final sheet state for comparison.

Usage:
  python agent_head_to_head.py --arm dv --out runs_agent_dv
  python agent_head_to_head.py --arm ff --out runs_agent_ff
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 stdio on Windows (default cp1252 chokes on star emoji etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright

_here = Path(__file__).resolve().parent
_root = None
for _cand in (_here, _here.parent, Path("C:/Users/david/Projects/deltavision")):
    if (_cand / "observer.py").exists():
        sys.path.insert(0, str(_cand))
        _root = _cand
        break

load_dotenv(_root / ".env" if _root and (_root / ".env").exists() else None)

from observer import DeltaVisionObserver  # noqa: E402
from anthropic import Anthropic, RateLimitError  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
VIEWPORT = {"width": 1280, "height": 800}
MAX_STEPS = 30

SHEET_URL = "https://docs.google.com/spreadsheets/d/1_WQ2e9-7CS6NbFZ-3WsdP5Mrfptb1e4_CWIlVOszKzc/edit?usp=sharing"
MAPS_URL = "https://www.google.com/maps/search/apartments+brooklyn+ny"

SYSTEM = """You are an apartment-research agent working inside a browser. You have two tabs:

  Tab 0: Google Maps with a search for apartments in Brooklyn NY (START HERE)
  Tab 1: A blank Google Sheet for documenting what you find

Your task, in order:
  1. On the Maps tab, research 2 DIFFERENT apartments. For each, click the
     listing to see its details (name, address, rating).
  2. Switch to the Sheet tab. Put a header row in row 1:
        A1=Apartment  B1=Address  C1=Rating  D1=Note
  3. Add one row per apartment (rows 2, 3) with its details. Note is a
     short 1-sentence observation (e.g. "great reviews", "close to G train").
  4. When both data rows are filled, call `done`.

EFFICIENCY GUIDELINES — the task should take ~18-22 steps:
- You do NOT need to screenshot after every action. Trust that type/Tab/Enter
  work. Only screenshot when you need to verify a tab switch, page load, or
  unexpected state.
- Type whole cell values at once (not character-by-character). Use Tab to
  advance columns efficiently.
- When you see the data you need, move on. Don't linger.

Behave like a thoughtful user. Prefer batching related text (type a whole
cell value, then Tab to next column) over single characters. Only screenshot
when you need to confirm what changed — not after every tiny action. Use
switch_tab(0) for Maps and switch_tab(1) for the Sheet."""

TASK_PROMPT = f"You're on the Maps tab. The Sheet is tab 1. Sheet URL: {SHEET_URL}. Begin."


def build_tools():
    return [
        {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": VIEWPORT["width"],
            "display_height_px": VIEWPORT["height"],
            "display_number": 1,
        },
        {
            "name": "navigate",
            "description": "Navigate the current tab to a URL.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
        {
            "name": "switch_tab",
            "description": "Switch to tab by index. 0=Maps, 1=Sheet.",
            "input_schema": {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
        },
        {
            "name": "done",
            "description": "Task complete. Call when Sheet is filled.",
            "input_schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    ]


class Arm:
    """Wraps observation delivery. FF = raw PNG. DV = DeltaVisionObserver output."""
    def __init__(self, name: str, page, observer):
        self.name = name
        self.page = page
        self.observer = observer
        self.screenshot_count = 0
        self.total_image_tokens_estimate = 0

    def take_screenshot_bytes(self) -> bytes:
        b = self.page.screenshot()
        self.screenshot_count += 1
        return b

    def make_tool_result_image_content(self, png_bytes: bytes) -> list:
        if self.observer is None:
            # FF arm — raw PNG
            b64 = base64.b64encode(png_bytes).decode()
            w, h = Image.open(io.BytesIO(png_bytes)).size
            self.total_image_tokens_estimate += max(75, int(w * h / 750))
            return [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }]
        # DV arm
        obs = self.observer.observe(png_bytes, url=self.page.url, last_action="screenshot")
        content = obs.to_anthropic_tool_result_content()
        self.total_image_tokens_estimate += obs.estimated_image_tokens()
        return content


def translate_key(raw: str) -> str:
    """Claude uses xdotool-style ('ctrl+a'); Playwright wants ('Control+A')."""
    parts = raw.split("+")
    mods = {"ctrl": "Control", "alt": "Alt", "shift": "Shift",
            "meta": "Meta", "cmd": "Meta", "super": "Meta"}
    named = {"return": "Enter", "kp_enter": "Enter", "escape": "Escape",
             "esc": "Escape", "page_down": "PageDown", "page_up": "PageUp",
             "space": "Space", "tab": "Tab", "backspace": "Backspace",
             "delete": "Delete", "end": "End", "home": "Home",
             "up": "ArrowUp", "down": "ArrowDown",
             "left": "ArrowLeft", "right": "ArrowRight"}
    out = []
    for p in parts:
        lp = p.lower()
        if lp in mods:
            out.append(mods[lp])
        elif lp in named:
            out.append(named[lp])
        elif len(p) == 1:
            out.append(p.upper() if p.isalpha() else p)
        else:
            out.append(p)
    return "+".join(out)


def do_computer_action(page, action: str, **kw):
    if action == "screenshot":
        return "screenshot"
    if action == "left_click":
        x, y = kw["coordinate"]; page.mouse.click(x, y)
    elif action == "right_click":
        x, y = kw["coordinate"]; page.mouse.click(x, y, button="right")
    elif action == "double_click":
        x, y = kw["coordinate"]; page.mouse.dblclick(x, y)
    elif action == "triple_click":
        x, y = kw["coordinate"]; page.mouse.click(x, y, click_count=3)
    elif action == "mouse_move":
        x, y = kw["coordinate"]; page.mouse.move(x, y)
    elif action == "left_mouse_down":
        page.mouse.down()
    elif action == "left_mouse_up":
        page.mouse.up()
    elif action == "left_click_drag":
        sx, sy = kw.get("start_coordinate", kw.get("coordinate"))
        ex, ey = kw["coordinate"]
        page.mouse.move(sx, sy); page.mouse.down()
        page.mouse.move(ex, ey, steps=12); page.mouse.up()
    elif action == "type":
        page.keyboard.type(kw["text"], delay=25)
    elif action == "key":
        try:
            page.keyboard.press(translate_key(kw["text"]))
        except Exception as e:
            return f"key press failed: {e}"
    elif action == "scroll":
        dy = 300 * kw.get("scroll_amount", 3)
        if kw.get("scroll_direction") == "up":
            dy = -dy
        page.mouse.wheel(0, dy)
    elif action == "wait":
        page.wait_for_timeout(1000)
    elif action == "hold_key":
        page.keyboard.down(translate_key(kw["text"]))
        page.wait_for_timeout(kw.get("duration", 1) * 1000)
        page.keyboard.up(translate_key(kw["text"]))
    elif action == "cursor_position":
        pass
    else:
        return f"unknown action: {action}"
    return None


def clear_sheet(page):
    """Wipe the sheet before the agent starts so stale data doesn't confuse it."""
    try:
        page.keyboard.press("Control+Home"); page.wait_for_timeout(300)
        page.keyboard.press("Control+A"); page.wait_for_timeout(300)
        page.keyboard.press("Control+A"); page.wait_for_timeout(300)
        page.keyboard.press("Delete"); page.wait_for_timeout(800)
        page.keyboard.press("Control+Home"); page.wait_for_timeout(300)
    except Exception as e:
        print(f"(clear_sheet warning: {e})")


def save_summary(out_dir: Path, arm_name: str, step: int, done: bool, arm: Arm,
                 total_in: int, total_out: int, step_log: list, t0: float,
                 crash_info: str | None):
    summary = {
        "arm": arm_name,
        "model": MODEL,
        "task": "apartment research -> Google Sheet",
        "sheet_url": SHEET_URL,
        "steps": step,
        "done": done,
        "max_steps_hit": step >= MAX_STEPS and not done,
        "crashed": crash_info is not None,
        "crash_info": crash_info,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "screenshots_taken": arm.screenshot_count if arm else 0,
        "image_tokens_sent_estimate": arm.total_image_tokens_estimate if arm else 0,
        "elapsed_s": round(time.time() - t0, 1),
        "step_log": step_log,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "step_log"}, indent=2))
    return summary


def run(arm_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "video").mkdir(exist_ok=True)
    client = Anthropic()
    observer = DeltaVisionObserver() if arm_name == "dv" else None

    step_log = []
    total_in = 0
    total_out = 0
    done = False
    step = 0
    arm = None
    crash_info = None
    t0 = time.time()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            viewport=VIEWPORT,
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
            locale="en-US",
            timezone_id="America/New_York",
            record_video_dir=str(out_dir / "video"),
            record_video_size=VIEWPORT,
        )

        try:
            # Set up both tabs
            tab_maps = ctx.new_page()
            tab_maps.goto(MAPS_URL, wait_until="domcontentloaded", timeout=30000)
            tab_maps.wait_for_timeout(4000)

            tab_sheet = ctx.new_page()
            tab_sheet.goto(SHEET_URL, wait_until="domcontentloaded", timeout=30000)
            tab_sheet.wait_for_timeout(3500)

            # Clear any stale data left in the sheet
            tab_sheet.bring_to_front()
            tab_sheet.wait_for_timeout(800)
            clear_sheet(tab_sheet)

            tab_maps.bring_to_front()
            arm = Arm(arm_name, tab_maps, observer)

            def ss_content():
                png = arm.take_screenshot_bytes()
                (out_dir / f"ss_{arm.screenshot_count:03d}.png").write_bytes(png)
                return arm.make_tool_result_image_content(png)

            tools = build_tools()
            messages = [{"role": "user", "content": TASK_PROMPT}]

            while not done and step < MAX_STEPS:
                step += 1
                print(f"\n--- step {step} ({arm_name}) ---")
                # Retry loop for rate limits. Tier-1 Anthropic cap = 30K input
                # tok/min; free-running agents with growing context easily hit
                # it. Sleep + retry with jitter.
                resp = None
                for attempt in range(6):
                    try:
                        resp = client.beta.messages.create(
                            model=MODEL,
                            max_tokens=1024,
                            system=SYSTEM,
                            tools=tools,
                            messages=messages,
                            betas=["computer-use-2025-01-24"],
                        )
                        break
                    except RateLimitError as rle:
                        wait = 65 + attempt * 30  # 65, 95, 125, 155, 185, 215
                        print(f"  ! 429 (attempt {attempt+1}); sleeping {wait}s ...")
                        time.sleep(wait)
                if resp is None:
                    raise RuntimeError("rate limit exhausted after 6 retries")
                total_in += resp.usage.input_tokens
                total_out += resp.usage.output_tokens
                print(f"  in={resp.usage.input_tokens} out={resp.usage.output_tokens} "
                      f"stop={resp.stop_reason}")

                assistant_content = []
                tool_results = []
                for block in resp.content:
                    assistant_content.append(block)
                    if block.type == "text":
                        txt = block.text.strip()
                        if txt:
                            print(f"  say: {txt[:160]}")
                    elif block.type == "tool_use":
                        name = block.name
                        inp = block.input
                        print(f"  tool: {name} {json.dumps(inp)[:160]}")
                        try:
                            if name == "computer":
                                act = inp.get("action")
                                if act == "screenshot":
                                    content = ss_content()
                                else:
                                    err = do_computer_action(
                                        arm.page, act,
                                        **{k: v for k, v in inp.items() if k != "action"},
                                    )
                                    arm.page.wait_for_timeout(800)
                                    if err and err != "screenshot":
                                        content = [{"type": "text", "text": err}]
                                    else:
                                        content = ss_content()
                                tool_results.append({
                                    "type": "tool_result", "tool_use_id": block.id, "content": content,
                                })
                            elif name == "navigate":
                                arm.page.goto(inp["url"], wait_until="domcontentloaded", timeout=25000)
                                arm.page.wait_for_timeout(2500)
                                tool_results.append({
                                    "type": "tool_result", "tool_use_id": block.id,
                                    "content": ss_content(),
                                })
                            elif name == "switch_tab":
                                idx = inp["index"]
                                pages = ctx.pages
                                if 0 <= idx < len(pages):
                                    pages[idx].bring_to_front()
                                    arm.page = pages[idx]
                                    arm.page.wait_for_timeout(800)
                                    tool_results.append({
                                        "type": "tool_result", "tool_use_id": block.id,
                                        "content": ss_content(),
                                    })
                                else:
                                    tool_results.append({
                                        "type": "tool_result", "tool_use_id": block.id,
                                        "content": [{"type": "text", "text": f"invalid tab index {idx}"}],
                                    })
                            elif name == "done":
                                print(f"  DONE: {inp.get('summary', '')}")
                                done = True
                                tool_results.append({
                                    "type": "tool_result", "tool_use_id": block.id,
                                    "content": [{"type": "text", "text": "ok"}],
                                })
                        except Exception as inner:
                            # Never let a single bad tool call abort the run
                            print(f"  tool error: {inner}")
                            tool_results.append({
                                "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": [{"type": "text", "text": f"error: {inner}"}],
                            })

                step_log.append({
                    "step": step,
                    "in_toks": resp.usage.input_tokens,
                    "out_toks": resp.usage.output_tokens,
                    "stop": resp.stop_reason,
                    "screenshot_count": arm.screenshot_count,
                    "image_tokens_sent_estimate": arm.total_image_tokens_estimate,
                    "t_rel": round(time.time() - t0, 1),
                })

                messages.append({"role": "assistant", "content": assistant_content})
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                elif resp.stop_reason == "end_turn":
                    break

            # Final screenshot of sheet
            try:
                tab_sheet.bring_to_front()
                tab_sheet.wait_for_timeout(1500)
                tab_sheet.screenshot(path=str(out_dir / "final_sheet.png"))
            except Exception as e:
                print(f"(final sheet shot warning: {e})")

        except Exception as e:
            crash_info = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"\n*** CRASH: {crash_info}")

        # Always save summary, even on crash
        save_summary(out_dir, arm_name, step, done, arm, total_in, total_out,
                     step_log, t0, crash_info)

        try:
            ctx.close()
            browser.close()
        except Exception:
            pass

    # Rename the recorded video
    vids = list((out_dir / "video").glob("*.webm"))
    if vids:
        tgt = out_dir / "browser.webm"
        if tgt.exists():
            tgt.unlink()
        vids[0].rename(tgt)
        print(f"Video: {tgt} ({tgt.stat().st_size / 1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["dv", "ff"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.arm, args.out)


if __name__ == "__main__":
    main()
