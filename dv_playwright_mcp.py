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
import atexit
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

from config import DeltaVisionConfig  # noqa: E402  (sys.path mutation above)
from results.trace import (  # noqa: E402
    TraceHeader,
    TraceStep,
    TraceWriter,
    bytes_sha256,
    canonical_image_manifest,
    config_hash,
    payload_text_sha256,
)
from vision.classifier import TransitionType, classify_transition, extract_anchor  # noqa: E402
from vision.diff import compute_diff, extract_crops  # noqa: E402

# DV version the trace says it was emitted by. Read from package metadata so
# updates to deltavision==X.Y.Z are reflected automatically without touching
# this file. Falls back to "unknown" if metadata isn't available (e.g. running
# from a source checkout without pip install).
try:
    from importlib.metadata import version as _pkg_version  # noqa: E402
    _DV_VERSION = _pkg_version("deltavision")
except Exception:
    _DV_VERSION = "unknown"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("dv_proxy")

DV_CONFIG = DeltaVisionConfig()

# Ablation: set DV_FORCE_FULL_FRAME=1 to bypass classification and always return
# the full screenshot. Log entries will record ff_tokens==dv_tokens==FULL_FRAME_TOKENS
# so post-run analysis can diff FF vs DV runs apples-to-apples.
FORCE_FULL_FRAME = os.environ.get("DV_FORCE_FULL_FRAME", "").lower() in ("1", "true", "yes")

LOG_DIR = Path(os.environ.get("DV_LOG_DIR", "dv_runs"))
LOG_DIR.mkdir(exist_ok=True)
RUN_ID = int(time.time())
# FF runs get an _ff suffix so the log filename alone tells you which ablation you're looking at.
_suffix = "_ff" if FORCE_FULL_FRAME else ""
LOG_FILE = LOG_DIR / f"dv_proxy_run_{RUN_ID}{_suffix}.jsonl"

# Side-by-side v1 BenchmarkTrace file. The legacy `dv_proxy_run_*.jsonl` above
# is preserved unchanged for downstream tooling; this new file emits the
# canonical schema validated by `deltavision verify-trace`. Both files write
# during the same run and any schema/log inconsistency between them is itself
# diagnostic. See results/trace.py for the schema.
TRACE_FILE = LOG_DIR / f"dv_trace_v1_{RUN_ID}{_suffix}.jsonl"

FULL_FRAME_TOKENS = 1365
CROP_BASE_TOKENS = 85
CROP_PER_TILE = 170

# When the agent takes multiple screenshots in a row that produce no visible change,
# something is wrong — either the action didn't land or the agent is stuck. We escalate:
#   streak == 1  -> soft nudge (text hint appended to the no-change response)
#   streak >= 2  -> hard nudge (force a full-frame return + hint)
# This is the proxy-side equivalent of agent/loop.py's MAX_NO_EFFECT_RETRIES but applies
# to any MCP client (the dv-playwright proxy is used by Claude Code subagents that don't
# run agent/loop.py). Tunable via DV_NO_CHANGE_HARD_NUDGE env (default 2).
NO_CHANGE_HARD_NUDGE = int(os.environ.get("DV_NO_CHANGE_HARD_NUDGE", "2"))

# Periodic full-frame refresh: after N consecutive DELTA frames, force the next
# response to be a full frame even if the diff is small. Rationale: when the
# agent is interacting with a dialog/dropdown, every step's diff is tiny and
# DV ships a small crop. The agent loses spatial context (sees "the dropdown
# changed" without seeing the surrounding form state) and starts second-guessing
# itself, leading to extra retry steps that DON'T happen with full-frame
# observation. Periodic re-anchoring caps how far the spatial drift can go.
# Empirical observation that motivated this: in the SF Maps→Sheets parallel
# trial, DV spent 16 frames on a sort-dialog interaction that FF did in 7.
# Tunable via DV_DELTA_REFRESH_EVERY env (default 5).
DELTA_REFRESH_EVERY = int(os.environ.get("DV_DELTA_REFRESH_EVERY", "5"))


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
# v1 trace writer (alongside legacy log, never replaces it)
# ---------------------------------------------------------------------------
#
# The trace writer is lazy-initialized on the first step so we capture the
# task identity from env vars that may be set after this module is imported.
# It's closed via atexit so a clean process termination produces a valid
# summary line; an unclean termination (kill -9, OOM, SIGSEGV) leaves the
# trace summary-less, which the verifier surfaces as a warning.

_TRACE_WRITER: TraceWriter | None = None


def _trace_writer() -> TraceWriter:
    """Return the per-run trace writer, opening it on first call."""
    global _TRACE_WRITER
    if _TRACE_WRITER is None:
        # Task identity comes from env so the same process serves multiple
        # benchmarks if needed. Defaults are intentionally generic — the
        # paired A/B harness sets these explicitly per-trial.
        task_id = os.environ.get("DV_TASK_ID", f"unknown-{RUN_ID}")
        task_description = os.environ.get(
            "DV_TASK_DESCRIPTION", "DV-Playwright proxy run"
        )
        trial_group_id = os.environ.get("DV_TRIAL_GROUP_ID", f"run-{RUN_ID}")
        observation_mode = "ff" if FORCE_FULL_FRAME else "dv"
        header = TraceHeader(
            trace_id=f"dv-proxy-{RUN_ID}{_suffix}",
            trial_group_id=trial_group_id,
            observation_mode=observation_mode,
            dv_version=_DV_VERSION,
            dv_config_hash=config_hash(DV_CONFIG),
            task_id=task_id,
            task_description=task_description,
            model=os.environ.get("DV_MODEL"),
        )
        _TRACE_WRITER = TraceWriter(TRACE_FILE, header)
    return _TRACE_WRITER


def _close_trace_writer() -> None:
    """Idempotent close. Registered with atexit so a clean process exit
    writes the summary line; the validator treats a missing summary as
    'this run did not complete cleanly,' which is correct behavior on a
    crash."""
    global _TRACE_WRITER
    if _TRACE_WRITER is not None:
        try:
            _TRACE_WRITER.close(write_summary=True)
        except Exception:
            # Don't let an atexit handler raise — that masks the real error.
            log.exception("trace writer close failed")
        _TRACE_WRITER = None


atexit.register(_close_trace_writer)


def _payload_image_block(img: Image.Image) -> tuple[str, dict]:
    """Encode a PIL image as PNG bytes and produce a manifest entry.

    Returns (b64_str_for_mcp, manifest_dict_for_trace). The b64 string is what
    we send back to the MCP client; the manifest dict is what the trace
    records. Both come from the SAME bytes, so the manifest hash is provably
    a hash of what the model actually saw.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    manifest = {
        "sha256": bytes_sha256(raw),
        "media_type": "image/png",
        "bytes": len(raw),
    }
    return b64, manifest


def _frame_sha256(img: Image.Image) -> str:
    """sha256 of the canonical PNG bytes of `img`. Used for frame_sha256
    (the bytes DV consumed). Distinct from the payload-image hash because
    DV may consume a 1280×800 frame and ship a 256×128 crop — different bytes,
    different hashes.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return bytes_sha256(buf.getvalue())


def _emit_trace_step(
    *,
    step_idx: int,
    full_frame: Image.Image,
    payload_images_manifest: list[dict],
    payload_text_blocks: list[str],
    obs_type: str,
    trigger: str,
    model_facing_tokens: int,
    crop_bboxes_px: list[list[int]],
    url: str | None = None,
) -> None:
    """Emit one TraceStep to the v1 trace file.

    Args:
        full_frame: the screenshot DV consumed this step (always full frame —
            DV needs the whole thing internally to compute next-step diff).
            Hashed for frame_sha256 and counted as dv_internal_tokens.
        payload_images_manifest: pre-built manifest entries (one per image
            block in the model-facing payload). Empty list for text-only
            payloads (e.g. soft no-change nudge).
        payload_text_blocks: list of text strings the proxy puts in the
            payload, joined with newlines for the text-hash input.
        obs_type: "full_frame" if the model saw the full frame, "delta" if
            it saw crops or a text-only response.
        trigger: free-form classifier-trigger string (initial, new_page,
            periodic_refresh, etc.).
        model_facing_tokens: estimated token cost of the payload sent to
            the model. Equal to dv_internal_tokens on full-frame steps,
            smaller on delta steps.
        crop_bboxes_px: [[x, y, w, h], ...] in source pixel coords. Empty
            on full-frame steps.
        url: optional URL the screenshot was taken on (proxy doesn't always
            know).
    """
    text_payload = "\n".join(payload_text_blocks)
    img_hash, canonical = canonical_image_manifest(payload_images_manifest)
    step = TraceStep(
        step_idx=step_idx,
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        url=url,
        obs_type=obs_type,
        trigger=trigger,
        dv_internal_tokens=FULL_FRAME_TOKENS,  # always — DV consumed full frame
        model_facing_tokens=model_facing_tokens,
        frame_sha256=_frame_sha256(full_frame),
        payload_image_sha256=img_hash,
        payload_text_sha256=payload_text_sha256(text_payload),
        payload_images=canonical,
        crop_bboxes_px=crop_bboxes_px,
    )
    _trace_writer().write_step(step)


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
        # Consecutive screenshots that produced no visible change. Incremented when
        # the diff engine finds no crops to send; reset on any actual change. Drives
        # the escalating nudge behavior below.
        self.no_change_streak: int = 0
        # Consecutive DELTA frames (any non-NEW_PAGE, non-initial step). Drives the
        # periodic full-frame refresh: after DELTA_REFRESH_EVERY deltas in a row, the
        # next step force-sends a full frame so the agent re-anchors its spatial
        # context. Reset on any new_page or forced refresh.
        self.delta_streak: int = 0

    def process_screenshot(self, t1: Image.Image) -> tuple[list[dict], str, str]:
        """
        Given a new screenshot t1, classify transition from t0 → t1.
        Returns (mcp_content_items, transition_type, trigger).
        mcp_content_items: list of MCP ImageContent-compatible dicts to return to model.
        """
        self.step += 1

        # FF ablation mode: always send the full screenshot, skip all classification.
        # Used to establish the apples-to-apples FF baseline against a DV run.
        if FORCE_FULL_FRAME:
            self.t0 = t1
            self.ff_tokens += FULL_FRAME_TOKENS
            self.dv_tokens += FULL_FRAME_TOKENS
            b64, manifest = _payload_image_block(t1)
            content = [{"type": "image", "data": b64, "mimeType": "image/png"}]
            savings_pct = 0.0  # FF has no savings by construction
            _log({
                "step": self.step,
                "transition": "ff_full",
                "trigger": "force_full_frame",
                "ff_tokens": FULL_FRAME_TOKENS,
                "dv_tokens": FULL_FRAME_TOKENS,
                "ff_cumulative": self.ff_tokens,
                "dv_cumulative": self.dv_tokens,
                "savings_pct_cumulative": savings_pct,
            })
            _emit_trace_step(
                step_idx=self.step, full_frame=t1,
                payload_images_manifest=[manifest],
                payload_text_blocks=[],
                obs_type="full_frame", trigger="force_full_frame",
                model_facing_tokens=FULL_FRAME_TOKENS,
                crop_bboxes_px=[],
            )
            return content, "ff_full", "force_full_frame"

        if self.t0 is None:
            # First screenshot — always full frame
            self.t0 = t1
            self.anchor = extract_anchor(t1, DV_CONFIG)
            tokens = FULL_FRAME_TOKENS
            self.ff_tokens += tokens
            self.dv_tokens += tokens
            b64, manifest = _payload_image_block(t1)
            content = [{"type": "image", "data": b64, "mimeType": "image/png"}]
            _log({"step": self.step, "transition": "initial", "trigger": "initial",
                  "ff_tokens": tokens, "dv_tokens": tokens, "ff_cumulative": self.ff_tokens,
                  "dv_cumulative": self.dv_tokens})
            _emit_trace_step(
                step_idx=self.step, full_frame=t1,
                payload_images_manifest=[manifest],
                payload_text_blocks=[],
                obs_type="full_frame", trigger="initial",
                model_facing_tokens=tokens,
                crop_bboxes_px=[],
            )
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

        # Per-step trace state, populated by whichever branch fires below.
        # Captured here at function scope so the shared trailing _log() block
        # can also emit a v1 trace step. Branches that early-return (FF, initial,
        # periodic_refresh, no_change_hard_nudge) emit their own trace step
        # inline before the return; this state is only consumed by the shared
        # tail emit.
        trace_payload_images: list[dict] = []
        trace_payload_text: list[str] = []
        trace_obs_type: str = "delta"
        trace_bboxes: list[list[int]] = []

        if result.transition == TransitionType.NEW_PAGE:
            # Send full frame — new page, anchor resets. Page navigation is an
            # actual change (even if our own agent triggered it), so reset the
            # no-change streak.
            dv_cost = FULL_FRAME_TOKENS
            self.dv_tokens += dv_cost
            self.t0 = t1
            self.anchor = extract_anchor(t1, DV_CONFIG)
            self.no_change_streak = 0
            self.delta_streak = 0
            b64, manifest = _payload_image_block(t1)
            content = [{"type": "image", "data": b64, "mimeType": "image/png"}]
            transition = "new_page"
            trace_payload_images = [manifest]
            trace_obs_type = "full_frame"
        elif self.delta_streak >= DELTA_REFRESH_EVERY:
            # Periodic full-frame refresh — the agent has been seeing only deltas for
            # too long and may have lost spatial context. Send a full frame so it
            # re-anchors. Logged as `transition=delta` with `trigger=periodic_refresh`
            # so post-run analysis can see how often this fires.
            dv_cost = FULL_FRAME_TOKENS
            self.dv_tokens += dv_cost
            self.t0 = t1
            self.anchor = extract_anchor(t1, DV_CONFIG)
            self.delta_streak = 0
            refresh_text = (
                f"[DeltaVision: periodic refresh after {DELTA_REFRESH_EVERY} deltas — "
                f"sending full frame so you can re-anchor on the surrounding state.]"
            )
            b64, manifest = _payload_image_block(t1)
            content = [
                {"type": "image", "data": b64, "mimeType": "image/png"},
                {"type": "text", "text": refresh_text},
            ]
            savings_pct = round((self.ff_tokens - self.dv_tokens) / self.ff_tokens * 100, 1)
            _log({
                "step": self.step,
                "transition": "delta",
                "trigger": "periodic_refresh",
                "diff_ratio": round(result.diff_ratio, 4),
                "phash_distance": result.phash_distance,
                "ff_tokens": ff_cost,
                "dv_tokens": dv_cost,
                "ff_cumulative": self.ff_tokens,
                "dv_cumulative": self.dv_tokens,
                "savings_pct_cumulative": savings_pct,
            })
            # Trace step: trigger=periodic_refresh, obs_type=full_frame (we DID
            # send a full frame; the proxy's legacy log labels it transition=delta
            # for streak-tracking continuity, but at the wire level it is a
            # full_frame observation).
            _emit_trace_step(
                step_idx=self.step, full_frame=t1,
                payload_images_manifest=[manifest],
                payload_text_blocks=[refresh_text],
                obs_type="full_frame", trigger="periodic_refresh",
                model_facing_tokens=dv_cost,
                crop_bboxes_px=[],
            )
            return content, "delta", "periodic_refresh"
        else:
            # DELTA — send only changed crops
            crops = extract_crops(self.t0, t1, diff_result.changed_bboxes, DV_CONFIG.CROP_PADDING)
            if crops:
                # Real visible change — reset the no-change streak, advance the
                # delta streak (drives the periodic-refresh check above on the
                # NEXT call).
                self.no_change_streak = 0
                self.delta_streak += 1
                dv_cost = _estimate_crop_tokens(crops)
                # Token-cap guard: if fragmenting into N crops would cost MORE than a full
                # frame (can happen when many small scattered regions each add their base
                # overhead), fall back to full frame. Guarantees DV is never strictly
                # worse than FF on any individual step. Without this guard we observed
                # DV spending 2,210 tokens on a 9%-diff frame vs FF's fixed 1,365 cost.
                if dv_cost > FULL_FRAME_TOKENS:
                    dv_cost = FULL_FRAME_TOKENS
                    self.anchor = extract_anchor(t1, DV_CONFIG)
                    cap_text = "[DeltaVision: crops exceeded full-frame cost — sending full frame instead]"
                    b64, manifest = _payload_image_block(t1)
                    content = [
                        {"type": "image", "data": b64, "mimeType": "image/png"},
                        {"type": "text", "text": cap_text},
                    ]
                    trace_payload_images = [manifest]
                    trace_payload_text = [cap_text]
                    trace_obs_type = "full_frame"
                else:
                    content = []
                    for c in crops:
                        img = c["crop_after"]
                        bbox = c["bbox"]
                        b64, manifest = _payload_image_block(img)
                        content.append({
                            "type": "image",
                            "data": b64,
                            "mimeType": "image/png",
                        })
                        # Include bbox as text context for the model
                        bbox_text = f"[DeltaVision: changed region at x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}]"
                        content.append({"type": "text", "text": bbox_text})
                        trace_payload_images.append(manifest)
                        trace_payload_text.append(bbox_text)
                        trace_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                    trace_obs_type = "delta"
            else:
                # No visible change. Escalating nudge:
                #   streak 1        -> minimal response + soft hint
                #   streak >= hard  -> full frame + hard hint (break the loop)
                self.no_change_streak += 1

                if self.no_change_streak >= NO_CHANGE_HARD_NUDGE:
                    # Hard nudge: give the agent the full frame so it can re-orient,
                    # plus explicit guidance to try a different action. Reset streak
                    # and anchor so we don't fire this again immediately.
                    dv_cost = FULL_FRAME_TOKENS
                    self.anchor = extract_anchor(t1, DV_CONFIG)
                    self.no_change_streak = 0
                    nudge = (
                        f"[DeltaVision: {NO_CHANGE_HARD_NUDGE} consecutive no-change "
                        "screenshots — sending full frame to re-anchor. Your last "
                        "action(s) likely did not produce a state change. Try a "
                        "different approach: different selector, alternate interaction "
                        "(click vs press_key vs evaluate), or check whether a dialog/"
                        "loading-spinner is blocking input.]"
                    )
                    b64, manifest = _payload_image_block(t1)
                    content = [
                        {"type": "image", "data": b64, "mimeType": "image/png"},
                        {"type": "text", "text": nudge},
                    ]
                    transition = "new_page"  # log it as a forced refresh
                    result_trigger = "no_change_hard_nudge"
                    self.dv_tokens += dv_cost
                    self.t0 = t1
                    savings_pct = round((self.ff_tokens - self.dv_tokens) / self.ff_tokens * 100, 1)
                    _log({
                        "step": self.step,
                        "transition": transition,
                        "trigger": result_trigger,
                        "diff_ratio": round(result.diff_ratio, 4),
                        "phash_distance": result.phash_distance,
                        "ff_tokens": ff_cost,
                        "dv_tokens": dv_cost,
                        "ff_cumulative": self.ff_tokens,
                        "dv_cumulative": self.dv_tokens,
                        "savings_pct_cumulative": savings_pct,
                        "no_change_streak_before": NO_CHANGE_HARD_NUDGE,
                    })
                    _emit_trace_step(
                        step_idx=self.step, full_frame=t1,
                        payload_images_manifest=[manifest],
                        payload_text_blocks=[nudge],
                        obs_type="full_frame", trigger="no_change_hard_nudge",
                        model_facing_tokens=dv_cost,
                        crop_bboxes_px=[],
                    )
                    return content, transition, result_trigger
                else:
                    # Soft nudge: still minimal tokens, but tell the agent nothing happened.
                    dv_cost = CROP_BASE_TOKENS
                    soft_nudge = (
                        "[DeltaVision: no visible change detected. If you just "
                        "issued an action, it may not have landed — consider a "
                        "different selector or approach before taking another "
                        "screenshot.]"
                    )
                    content = [{"type": "text", "text": soft_nudge}]
                    # Soft nudge ships zero images, only the text hint. Trace
                    # payload_images stays empty; the manifest hash is the hash
                    # of an empty list (a stable, well-defined value).
                    trace_payload_text = [soft_nudge]
                    trace_obs_type = "delta"

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
        _emit_trace_step(
            step_idx=self.step, full_frame=t1,
            payload_images_manifest=trace_payload_images,
            payload_text_blocks=trace_payload_text,
            obs_type=trace_obs_type, trigger=result.trigger,
            model_facing_tokens=dv_cost,
            crop_bboxes_px=trace_bboxes,
        )

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

    # Start Playwright MCP as a subprocess.
    # user-data-dir must be unique per MCP instance or Chrome refuses to launch
    # a second browser ("profile already in use"). FF-mode auto-suffixes so the
    # two proxies can run concurrently without collision; override with
    # DV_PLAYWRIGHT_PROFILE_DIR if you want something else.
    default_profile_suffix = "_ff" if FORCE_FULL_FRAME else ""
    default_profile = str(Path.home() / f".playwright-mcp-profile{default_profile_suffix}")
    profile_dir = os.environ.get("DV_PLAYWRIGHT_PROFILE_DIR", default_profile)

    # Viewport size is fixed so screenshots from DV and FF proxies are directly
    # comparable for side-by-side video composition. Override with DV_VIEWPORT_SIZE
    # env like "1280,800" if you need a different size.
    viewport = os.environ.get("DV_VIEWPORT_SIZE", "1280,800")

    pw_args = os.environ.get("DV_PLAYWRIGHT_ARGS", "").split()
    pw_cmd = ["npx", "@playwright/mcp@latest",
               "--browser", "chrome",
               "--user-data-dir", profile_dir,
               "--viewport-size", viewport,
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
                  "log_file": str(LOG_FILE), "n_tools": len(_pw_tools),
                  "mode": "ff_full_frame" if FORCE_FULL_FRAME else "dv_classified"})

            # Run our proxy server on stdio
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream,
                    server.create_initialization_options()
                )


if __name__ == "__main__":
    asyncio.run(main())
