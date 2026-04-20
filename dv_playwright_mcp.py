"""
DeltaVision Playwright MCP Proxy
=================================
A transparent MCP proxy that wraps the Playwright MCP server and applies
DeltaVision's CV classifier to every browser_take_screenshot response.

Instead of returning the full screenshot to the model, it:
  - On NEW_PAGE: returns the full frame (DV agrees it's a new scene)
  - On DELTA: returns only the changed region crops (saving tokens)

Every decision is logged to dv_proxy.log for post-run analysis.

Usage: register this as an MCP server in Claude Desktop or Claude Code settings.
The server starts Playwright MCP as a subprocess and proxies everything through it.

Config (env vars):
  DV_LOG_DIR      — directory to write dv_proxy_run_*.jsonl logs (default: ./dv_runs)
  DV_PLAYWRIGHT_ARGS — extra args to pass to playwright mcp (default: empty)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ImageContent,
    TextContent,
    Tool,
)
from PIL import Image

# --- DeltaVision imports ---
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from config import DeltaVisionConfig
from vision.classifier import classify_transition, extract_anchor, TransitionType
from vision.diff import compute_diff, extract_crops

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("dv_proxy")

DV_CONFIG = DeltaVisionConfig()

LOG_DIR = Path(os.environ.get("DV_LOG_DIR", "dv_runs"))
LOG_DIR.mkdir(exist_ok=True)
RUN_ID = int(time.time())
LOG_FILE = LOG_DIR / f"dv_proxy_run_{RUN_ID}.jsonl"

FULL_FRAME_TOKENS = 1365
CROP_BASE_TOKENS = 85
CROP_PER_TILE = 170


def _estimate_crop_tokens(crops: list[dict]) -> int:
    total = 0
    for c in crops:
        img = c.get("crop_after")
        if img is None:
            continue
        w, h = img.size
        tw = max(1, (w + 511) // 512)
        th = max(1, (h + 511) // 512)
        total += CROP_BASE_TOKENS + tw * th * CROP_PER_TILE
    return max(total, CROP_BASE_TOKENS)


def _img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_img(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _log(entry: dict) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# DV state (per-session, single browser context assumption)
# ---------------------------------------------------------------------------

class DVState:
    def __init__(self):
        self.t0: Image.Image | None = None
        self.anchor: Image.Image | None = None
        self.step: int = 0
        self.ff_tokens: int = 0
        self.dv_tokens: int = 0

    def process_screenshot(self, t1: Image.Image) -> tuple[list[dict], str, str]:
        """
        Given a new screenshot t1, classify transition from t0 → t1.
        Returns (mcp_content_items, transition_type, trigger).
        mcp_content_items: list of MCP ImageContent-compatible dicts to return to model.
        """
        self.step += 1

        if self.t0 is None:
            # First screenshot — always full frame
            self.t0 = t1
            self.anchor = extract_anchor(t1, DV_CONFIG)
            tokens = FULL_FRAME_TOKENS
            self.ff_tokens += tokens
            self.dv_tokens += tokens
            content = [{"type": "image", "data": _img_to_b64(t1), "mimeType": "image/png"}]
            _log({"step": self.step, "transition": "initial", "trigger": "initial",
                  "ff_tokens": tokens, "dv_tokens": tokens, "ff_cumulative": self.ff_tokens,
                  "dv_cumulative": self.dv_tokens})
            return content, "initial", "initial"

        diff_result = compute_diff(self.t0, t1, DV_CONFIG)
        result = classify_transition(
            t0=self.t0, t1=t1,
            url_before="browser://unknown", url_after="browser://unknown",
            anchor_template=self.anchor,
            config=DV_CONFIG,
            diff_result=diff_result,
        )

        ff_cost = FULL_FRAME_TOKENS
        self.ff_tokens += ff_cost

        if result.transition == TransitionType.NEW_PAGE:
            # Send full frame — new page, anchor resets
            dv_cost = FULL_FRAME_TOKENS
            self.dv_tokens += dv_cost
            self.t0 = t1
            self.anchor = extract_anchor(t1, DV_CONFIG)
            content = [{"type": "image", "data": _img_to_b64(t1), "mimeType": "image/png"}]
            transition = "new_page"
        else:
            # DELTA — send only changed crops
            crops = extract_crops(self.t0, t1, diff_result.changed_bboxes, DV_CONFIG.CROP_PADDING)
            if crops:
                dv_cost = _estimate_crop_tokens(crops)
                content = []
                for c in crops:
                    img = c["crop_after"]
                    bbox = c["bbox"]
                    content.append({
                        "type": "image",
                        "data": _img_to_b64(img),
                        "mimeType": "image/png",
                    })
                    # Include bbox as text context for the model
                    content.append({
                        "type": "text",
                        "text": f"[DeltaVision: changed region at x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}]"
                    })
            else:
                # No visible change — send minimal signal
                dv_cost = CROP_BASE_TOKENS
                content = [{"type": "text", "text": "[DeltaVision: no visible change detected]"}]

            self.dv_tokens += dv_cost
            self.t0 = t1
            transition = "delta"

        savings_pct = round((self.ff_tokens - self.dv_tokens) / self.ff_tokens * 100, 1)
        _log({
            "step": self.step,
            "transition": transition,
            "trigger": result.trigger,
            "diff_ratio": round(result.diff_ratio, 4),
            "phash_distance": result.phash_distance,
            "ff_tokens": ff_cost,
            "dv_tokens": dv_cost,
            "ff_cumulative": self.ff_tokens,
            "dv_cumulative": self.dv_tokens,
            "savings_pct_cumulative": savings_pct,
        })

        return content, transition, result.trigger


# ---------------------------------------------------------------------------
# MCP proxy server
# ---------------------------------------------------------------------------

dv_state = DVState()
server = Server("dv-playwright-proxy")

# We'll store the Playwright client session globally once connected
_pw_session: ClientSession | None = None
_pw_tools: list[Tool] = []


async def _get_pw_session() -> ClientSession:
    global _pw_session, _pw_tools
    if _pw_session is not None:
        return _pw_session
    raise RuntimeError("Playwright session not initialized")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return _pw_tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list:
    session = await _get_pw_session()
    result: CallToolResult = await session.call_tool(name, arguments)

    # Pass through everything except browser_take_screenshot
    if name != "browser_take_screenshot":
        return result.content

    # Find the image in the response
    new_content = []
    screenshot_img: Image.Image | None = None

    for item in result.content:
        if hasattr(item, "type") and item.type == "image":
            screenshot_img = _b64_to_img(item.data)
        else:
            new_content.append(item)

    if screenshot_img is None:
        # No image in response — pass through unchanged
        return result.content

    # Reset DV state if viewport size changed (avoids OpenCV size-mismatch crash)
    if dv_state.t0 is not None and screenshot_img.size != dv_state.t0.size:
        log.warning(f"Viewport size changed {dv_state.t0.size} → {screenshot_img.size}, resetting DV state")
        dv_state.t0 = None
        dv_state.anchor = None

    # Run DV classifier
    dv_content, transition, trigger = dv_state.process_screenshot(screenshot_img)

    # Build MCP-compatible content list
    output = list(new_content)  # preserve any text items (page URL etc.)
    for item in dv_content:
        if item["type"] == "image":
            output.append(ImageContent(type="image", data=item["data"], mimeType=item["mimeType"]))
        else:
            output.append(TextContent(type="text", text=item["text"]))

    # Append DV metadata as text
    savings = round((dv_state.ff_tokens - dv_state.dv_tokens) / max(dv_state.ff_tokens, 1) * 100, 1)
    output.append(TextContent(
        type="text",
        text=f"[DV: step={dv_state.step} transition={transition} trigger={trigger} "
             f"ff_cumulative={dv_state.ff_tokens} dv_cumulative={dv_state.dv_tokens} savings={savings}%]"
    ))

    return output


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main():
    global _pw_session, _pw_tools

    # Start Playwright MCP as a subprocess
    pw_args = os.environ.get("DV_PLAYWRIGHT_ARGS", "").split()
    pw_cmd = ["npx", "@playwright/mcp@latest",
               "--browser", "chrome",
               "--user-data-dir", str(Path.home() / ".playwright-mcp-profile"),
               ] + pw_args

    pw_params = StdioServerParameters(command=pw_cmd[0], args=pw_cmd[1:])

    async with stdio_client(pw_params) as (pw_read, pw_write):
        async with ClientSession(pw_read, pw_write) as session:
            _pw_session = session
            await session.initialize()

            # Fetch tools from Playwright MCP and expose them
            tools_result = await session.list_tools()
            _pw_tools = tools_result.tools

            _log({"event": "proxy_started", "run_id": RUN_ID,
                  "log_file": str(LOG_FILE), "n_tools": len(_pw_tools)})

            # Run our proxy server on stdio
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream,
                    server.create_initialization_options()
                )


if __name__ == "__main__":
    asyncio.run(main())
