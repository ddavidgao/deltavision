"""
DeltaVisionObserver — the public middleware API.

Drop-in for any computer-use agent that wants delta-gated observations. Owns
the CV pipeline state (t0 anchor frame, anchor template) so the caller doesn't
have to. One method to call per step, plus format adapters for each major
CU framework.

Usage (format-agnostic):

    from observer import DeltaVisionObserver
    observer = DeltaVisionObserver()
    obs = observer.observe(screenshot=pil_or_bytes_or_b64, url=..., last_action=...)
    # obs is a DeltaObservation — adapt to your framework:
    obs.to_anthropic_tool_result_content()  # Anthropic computer-use
    obs.to_openai_computer_call_output(call_id)  # OpenAI CUA
    obs.to_browser_use_screenshot_b64()  # Browser Use / Hermes Agent
    obs.to_skyvern_screenshots_list()  # Skyvern
    obs.to_stagehand_middleware_parts()  # Stagehand / Vercel AI SDK

The observer is stateful per-session. Instantiate one per agent run.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Union

from PIL import Image, ImageDraw

from config import DeltaVisionConfig
from vision.classifier import (
    ClassificationResult,
    TransitionType,
    classify_transition,
    extract_anchor,
)
from vision.diff import compute_diff, extract_crops

ScreenshotInput = Union[Image.Image, bytes, str]


def _load_screenshot(screenshot: ScreenshotInput) -> Image.Image:
    """Accept PIL, raw PNG bytes, or base64 string. Return a PIL Image."""
    if isinstance(screenshot, Image.Image):
        return screenshot.convert("RGB") if screenshot.mode != "RGB" else screenshot
    if isinstance(screenshot, bytes):
        return Image.open(io.BytesIO(screenshot)).convert("RGB")
    if isinstance(screenshot, str):
        # Accept both raw base64 and data URLs (data:image/png;base64,...)
        if screenshot.startswith("data:"):
            _, _, b64 = screenshot.partition(",")
        else:
            b64 = screenshot
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    raise TypeError(
        f"screenshot must be PIL Image, bytes, or base64 string; got {type(screenshot)}"
    )


def _img_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _img_to_b64(img: Image.Image) -> str:
    return base64.standard_b64encode(_img_to_bytes(img)).decode()


# ============================================================= observation

@dataclass
class DVObservation:
    """
    Packaged observation produced by DeltaVisionObserver. Format-agnostic —
    downstream adapters pick the exact wire format for each framework.

    Two categories:
      - obs_type == "full_frame" — caller should send `frame` (full PNG)
      - obs_type == "delta"      — caller should send `thumbnail` + `crops`

    All image fields are PIL Images. Use the adapters (to_*) to serialize.
    """

    obs_type: str  # "full_frame" | "delta"
    step: int

    # Classifier output (always populated)
    trigger: str
    diff_ratio: float
    phash_distance: int
    anchor_score: float
    action_had_effect: bool

    # Full-frame payload (populated when obs_type == "full_frame")
    frame: Image.Image | None = None

    # Delta payload (populated when obs_type == "delta")
    thumbnail: Image.Image | None = None   # 320x225, green boxes on changed regions
    crops: list[Image.Image] = field(default_factory=list)  # detail crops, up to N
    crop_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    # Extras
    url: str | None = None
    last_action: str | None = None

    # Cost-split bookkeeping. Always populated by the observer regardless of
    # obs_type. (width, height) of the screenshot DV consumed this step. Used
    # by dv_internal_tokens() to report what DV processed *internally* (always
    # a full frame), separately from what was actually shipped to the model.
    # See `model_facing_tokens()` and `dv_internal_tokens()` below.
    consumed_frame_size: tuple[int, int] | None = None

    # -------------------------------------------------------------- accessors

    def is_new_page(self) -> bool:
        return self.obs_type == "full_frame"

    # ---- Cost split (added v1.0.7-dev) ---------------------------------
    #
    # DV has two distinct token-cost numbers, which the codebase used to
    # conflate under a single "estimated_image_tokens()" call:
    #
    #   model_facing_tokens()  — tokens DV actually put in front of the
    #                            model on this step. Smaller than a full
    #                            frame on delta observations. THIS is the
    #                            number that drives savings claims.
    #   dv_internal_tokens()   — tokens DV consumed internally to do its
    #                            job. Always full-frame size, every step,
    #                            regardless of what the model sees. DV
    #                            needs the full frame to compute the next
    #                            diff. This is the infrastructure cost.
    #
    # When a paper/README says "DV saved X% tokens," X% MUST be derived
    # from sum(model_facing_tokens) / sum(dv_internal_tokens) — that's
    # the model-cost claim. The infrastructure cost did NOT go down.
    # See bugs/SCHEMA.md and results/trace.py for the trace-level mirror.

    def model_facing_tokens(self) -> int:
        """Tokens DV actually put in front of the model on this step.

        On full_frame observations this is the cost of `frame`. On delta
        observations it's the sum of `thumbnail` + `crops` (the actual
        payload the adapter ships). This is the number that drives
        savings claims.
        """
        if self.is_new_page() and self.frame is not None:
            return self._image_tokens(self.frame)
        total = 0
        if self.thumbnail is not None:
            total += self._image_tokens(self.thumbnail)
        for c in self.crops:
            total += self._image_tokens(c)
        return total

    def dv_internal_tokens(self) -> int:
        """Tokens DV consumed internally on this step. Always equals the
        cost of a full frame at the consumed viewport size. DV processes
        every screenshot at full resolution to compute its diff; the
        savings only show up at the model-facing layer.

        Resolution order:
          1. If `consumed_frame_size` is set, use it.
          2. Otherwise, if `frame` is present (full_frame observations
             always have it), derive size from `frame.size`. This makes
             a hand-built full-frame DVObservation work even without
             consumed_frame_size — the data is right there.
          3. Otherwise raise ValueError. Hitting this path means a delta
             observation was constructed without consumed_frame_size,
             and there's no way to recover the consumed-frame cost from
             the observation alone (delta observations carry only the
             thumbnail + crops, not the full screenshot). Loud failure
             beats silent zero — see BUG-0008.
        """
        if self.consumed_frame_size is not None:
            w, h = self.consumed_frame_size
            return self._image_tokens_from_size(w, h)
        if self.frame is not None:
            return self._image_tokens(self.frame)
        raise ValueError(
            "DVObservation.dv_internal_tokens() cannot be computed: "
            "consumed_frame_size is None and frame is None (this is a "
            "delta observation built outside DeltaVisionObserver). Set "
            "consumed_frame_size=(width, height) to record the size of "
            "the screenshot DV consumed. See BUG-0008."
        )

    def estimated_image_tokens(self) -> int:
        """DEPRECATED ALIAS: returns the same value as `model_facing_tokens()`.

        Kept for backward compatibility — pre-cost-split benchmark scripts and
        user code call this method. Will be removed in v1.1.0. New code should
        call `model_facing_tokens()` (or `dv_internal_tokens()` for the
        complementary number).
        """
        return self.model_facing_tokens()

    @staticmethod
    def _image_tokens(img: Image.Image) -> int:
        # Anthropic: (width * height) / 750 (rough formula for their scaling).
        # Works well as a cross-model estimate for image-based pricing.
        return max(75, int((img.width * img.height) / 750))

    @staticmethod
    def _image_tokens_from_size(width: int, height: int) -> int:
        # Same formula as _image_tokens() but takes raw dimensions, used by
        # dv_internal_tokens() which doesn't keep the consumed PIL image
        # around on delta observations (would inflate memory). Output must
        # match _image_tokens(img) exactly when img has these dimensions.
        return max(75, int((width * height) / 750))

    # ------------------------------------------------------- format adapters

    def to_anthropic_tool_result_content(self) -> list[dict]:
        """
        Returns the 'content' list for a tool_result block.
        Integration point: anthropic-quickstarts computer-use-demo
            loop.py::_make_api_tool_result
        Replace:
            tool_result_content.append({"type":"image", "source": {...}})
        With:
            tool_result_content.extend(dv_obs.to_anthropic_tool_result_content())
        """
        if self.is_new_page():
            return [self._anthropic_image_block(self.frame)]

        blocks = [
            {"type": "text", "text": self._delta_header_text()},
        ]
        if self.thumbnail is not None:
            blocks.append({"type": "text", "text":
                "Page overview (thumbnail, green = changed regions):"})
            blocks.append(self._anthropic_image_block(self.thumbnail))
        for i, crop in enumerate(self.crops):
            blocks.append({"type": "text", "text": f"Changed region {i+1} detail:"})
            blocks.append(self._anthropic_image_block(crop))
        return blocks

    def to_openai_computer_call_output(self, call_id: str) -> dict:
        """
        Returns a fully formed computer_call_output item for OpenAI CUA.
        Integration point: openai-cua-sample-app runner-core/responses-loop.ts
            buildComputerCallOutput
        Replace:
            { type: "computer_call_output", call_id, output: { type: "computer_screenshot", image_url: data_url } }
        With:
            dv_obs.to_openai_computer_call_output(call_id)

        OpenAI CUA's `computer_screenshot` output expects a single image. On
        DELTA steps we still need to hand it one image, so we composite the
        thumbnail with crops into a single annotated frame.
        """
        img = self._single_composite_image()
        data_url = f"data:image/png;base64,{_img_to_b64(img)}"
        return {
            "type": "computer_call_output",
            "call_id": call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": data_url,
            },
        }

    def to_openai_vision_content(self) -> list[dict]:
        """
        Returns content parts for a message-style OpenAI call (not Responses
        API). For Anthropic-style adapters prefer `to_anthropic_*`; this is
        useful for non-CUA OpenAI agents that use chat/completions with vision.
        """
        parts: list[dict] = []
        if self.is_new_page():
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_img_to_b64(self.frame)}",
                    "detail": "high",
                },
            })
            return parts

        parts.append({"type": "text", "text": self._delta_header_text()})
        if self.thumbnail is not None:
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{_img_to_b64(self.thumbnail)}",
                "detail": "low",
            }})
        for i, crop in enumerate(self.crops):
            parts.append({"type": "text", "text": f"Changed region {i+1}:"})
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{_img_to_b64(crop)}",
                "detail": "high",
            }})
        return parts

    def to_browser_use_screenshot_b64(self) -> str:
        """
        Returns a single base64 PNG string compatible with browser-use's
        BrowserStateSummary.screenshot field.

        Integration point: browser_use/agent/service.py
            browser_state_summary = await self.browser_session.get_browser_state_summary(...)
        Monkey-patch or subclass so .screenshot is replaced with:
            observer.observe(orig_screenshot).to_browser_use_screenshot_b64()

        Browser Use expects ONE image, so we return either the full frame
        (NEW_PAGE) or a composited thumbnail+crops image (DELTA). Caller's
        MessageManager resize/detail settings still apply unchanged.
        """
        return _img_to_b64(self._single_composite_image())

    def to_skyvern_screenshots_list(self) -> list[bytes]:
        """
        Returns list[bytes] of PNG-encoded screenshots for Skyvern's
        `screenshots=` kwarg.
        Integration point: Skyvern llm_messages_builder() consumes raw PNG bytes.
        Wrap via LLMAPIHandlerFactory.get_override_llm_api_handler().

        DELTA: returns thumbnail + crops as separate PNGs (Skyvern already
        handles multi-image prompts well).
        NEW_PAGE: returns [full_frame_bytes].
        """
        if self.is_new_page():
            return [_img_to_bytes(self.frame)]
        out: list[bytes] = []
        if self.thumbnail is not None:
            out.append(_img_to_bytes(self.thumbnail))
        for c in self.crops:
            out.append(_img_to_bytes(c))
        return out

    def to_stagehand_middleware_parts(self) -> list[dict]:
        """
        Returns Vercel AI SDK V2 message content parts. Use inside
        LanguageModelV2Middleware.wrapGenerate by replacing any existing
        media/image parts in params.prompt.

        Example middleware (TypeScript):
            { wrapGenerate: async ({ doGenerate, params }) => {
                // ... fetch DV observation from a sidecar server ...
                params.prompt = replaceImagesWith(params.prompt, dvParts);
                return doGenerate();
              }
            }
        """
        parts: list[dict] = []
        if self.is_new_page():
            parts.append({
                "type": "media",
                "mediaType": "image/png",
                "data": _img_to_b64(self.frame),
            })
            return parts

        parts.append({"type": "text", "text": self._delta_header_text()})
        if self.thumbnail is not None:
            parts.append({
                "type": "media",
                "mediaType": "image/png",
                "data": _img_to_b64(self.thumbnail),
            })
        for c in self.crops:
            parts.append({
                "type": "media",
                "mediaType": "image/png",
                "data": _img_to_b64(c),
            })
        return parts

    def to_raw(self) -> dict:
        """
        Raw PIL images + metadata for DIY integrations or debugging.
        Safe to json-ify after dropping the image fields.

        Cost keys (since v1.0.7-dev):
            "model_facing_tokens" — preferred. What DV shipped to the model.
            "dv_internal_tokens"  — what DV consumed internally (always
                                    full-frame size).
            "estimated_image_tokens" — DEPRECATED alias for
                                    model_facing_tokens. Will be removed
                                    in v1.1.0. Kept here so existing user
                                    code doesn't break.
        Compute savings as `1 - model_facing / dv_internal`.
        """
        mft = self.model_facing_tokens()
        return {
            "obs_type": self.obs_type,
            "step": self.step,
            "trigger": self.trigger,
            "diff_ratio": self.diff_ratio,
            "phash_distance": self.phash_distance,
            "anchor_score": self.anchor_score,
            "action_had_effect": self.action_had_effect,
            "frame": self.frame,
            "thumbnail": self.thumbnail,
            "crops": list(self.crops),
            "crop_bboxes": list(self.crop_bboxes),
            "url": self.url,
            "last_action": self.last_action,
            # Cost split (preferred names)
            "model_facing_tokens": mft,
            "dv_internal_tokens": self.dv_internal_tokens(),
            # Back-compat alias (deprecated; remove in v1.1.0)
            "estimated_image_tokens": mft,
        }

    # --------------------------------------------------------- private helpers

    def _delta_header_text(self) -> str:
        parts = [
            "DELTA observation (same page, partial change).",
            f"Trigger: {self.trigger}",
            f"Diff ratio: {self.diff_ratio:.3f}   pHash: {self.phash_distance}",
            f"Action had effect: {self.action_had_effect}",
        ]
        if self.last_action:
            parts.append(f"Last action: {self.last_action}")
        return "\n".join(parts)

    def _anthropic_image_block(self, img: Image.Image) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _img_to_b64(img),
            },
        }

    def _single_composite_image(self) -> Image.Image:
        """Combine thumbnail + crops into one image for frameworks that
        expect exactly one screenshot per step (OpenAI CUA, Browser Use)."""
        if self.is_new_page():
            return self.frame
        if self.thumbnail is None and not self.crops:
            # edge case — should not happen, but don't crash
            return Image.new("RGB", (320, 225), (0, 0, 0))
        # Layout: thumbnail on top, crops in a row below
        thumb = self.thumbnail
        if not self.crops:
            return thumb
        # Constrain each crop to 200px tall so composite stays small
        crops_resized = []
        for c in self.crops[:3]:
            ratio = 200 / c.height if c.height > 0 else 1
            nw = max(1, int(c.width * ratio))
            crops_resized.append(c.resize((nw, 200), Image.LANCZOS))
        crop_row_w = sum(c.width for c in crops_resized) + 20 * (len(crops_resized) - 1)
        canvas_w = max(thumb.width, crop_row_w) + 20
        canvas_h = thumb.height + 220 + 20
        canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
        canvas.paste(thumb, ((canvas_w - thumb.width) // 2, 10))
        # Lay out crops in a row, centered
        row_y = thumb.height + 20
        x = (canvas_w - crop_row_w) // 2
        for c in crops_resized:
            canvas.paste(c, (x, row_y))
            x += c.width + 20
        return canvas


# ============================================================= observer

class DeltaVisionObserver:
    """
    Stateful delta-observation middleware.

    One instance per agent session. Call `observe(screenshot, ...)` after each
    step; the observer internally tracks the anchor frame (t0) and re-anchors
    on NEW_PAGE transitions.

    Accepts screenshots as PIL, raw PNG bytes, or base64 string — whatever
    your framework produces natively.
    """

    def __init__(self, config: DeltaVisionConfig | None = None):
        self.config = config or DeltaVisionConfig()
        self._t0: Image.Image | None = None
        self._url_t0: str | None = None
        self._anchor: np.ndarray | None = None  # noqa: F821
        self._step = 0
        self._last_classification: ClassificationResult | None = None
        self._no_change_streak = 0

    # ----------------------------------------------------------- properties

    @property
    def last_classification(self) -> ClassificationResult | None:
        return self._last_classification

    @property
    def step(self) -> int:
        return self._step

    @property
    def no_change_streak(self) -> int:
        return self._no_change_streak

    def reset(self) -> None:
        """Drop all state. Call between independent runs."""
        self._t0 = None
        self._url_t0 = None
        self._anchor = None
        self._step = 0
        self._last_classification = None
        self._no_change_streak = 0

    # ----------------------------------------------------------- the API

    def observe(
        self,
        screenshot: ScreenshotInput,
        *,
        url: str | None = None,
        last_action: str | None = None,
    ) -> DVObservation:
        """
        Classify the transition since the last call; return the packaged
        observation. The first call always returns a FULL_FRAME (initial
        state — no prior frame to diff against).

        Arguments:
            screenshot: PIL.Image | bytes | base64 str | data URL
            url:        current URL (None if not a browser / unknown)
            last_action: string repr of the action that produced this frame
                         (optional, used only for Layer 1-scroll bypass and
                         logging)

        Returns DVObservation. Caller picks the adapter.
        """
        frame = _load_screenshot(screenshot)

        # First call: bootstrap t0, return FULL_FRAME
        if self._t0 is None:
            self._t0 = frame
            self._url_t0 = url
            self._anchor = extract_anchor(frame, self.config)
            self._step = 0
            return self._build_full_frame_obs(
                frame=frame,
                trigger="initial",
                url=url,
                last_action=last_action,
                diff_ratio=0.0,
                phash_distance=0,
                anchor_score=1.0,
                action_had_effect=False,
            )

        # Classify transition
        diff = compute_diff(self._t0, frame, self.config)
        cls = classify_transition(
            t0=self._t0,
            t1=frame,
            url_before=self._url_t0 or "",
            url_after=url or "",
            anchor_template=self._anchor,
            config=self.config,
            diff_result=diff,
            last_action_type=self._infer_action_type(last_action),
        )
        self._last_classification = cls
        self._step += 1

        if not diff.action_had_effect:
            self._no_change_streak += 1
        else:
            self._no_change_streak = 0

        if cls.transition == TransitionType.NEW_PAGE:
            # Re-anchor on NEW_PAGE (same invariant as the V1 agent loop)
            self._t0 = frame
            self._url_t0 = url
            self._anchor = extract_anchor(frame, self.config)
            self._no_change_streak = 0
            return self._build_full_frame_obs(
                frame=frame,
                trigger=cls.trigger,
                url=url,
                last_action=last_action,
                diff_ratio=cls.diff_ratio,
                phash_distance=cls.phash_distance,
                anchor_score=cls.anchor_score,
                action_had_effect=diff.action_had_effect,
            )

        # DELTA path — build thumbnail + crops
        crops = extract_crops(
            self._t0, frame, diff.changed_bboxes, self.config.CROP_PADDING
        )

        # Guard: if the crops together cover most of the frame (common on
        # large scrolls), sending thumbnail + near-full-frame crop costs MORE
        # than the full frame alone. Fall back to full_frame in that case.
        # Observed regression case: a 600 px scroll produces a crop bbox
        # covering 100% of the viewport — DV: 1536 tok, FF: 1365 tok (+13%).
        # This guard turns those into a clean 1:1 with FF.
        frame_area = frame.width * frame.height
        crop_area = sum((c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1])
                        for c in crops[:2])
        CROP_COVERAGE_MAX = getattr(self.config, "CROP_COVERAGE_MAX", 0.75)
        if crops and frame_area > 0 and crop_area / frame_area >= CROP_COVERAGE_MAX:
            # Re-anchor (we're essentially on a new visual context) and emit full frame
            self._t0 = frame
            self._anchor = extract_anchor(frame, self.config)
            self._no_change_streak = 0
            return self._build_full_frame_obs(
                frame=frame,
                trigger="crop_covers_frame",
                url=url,
                last_action=last_action,
                diff_ratio=cls.diff_ratio,
                phash_distance=cls.phash_distance,
                anchor_score=cls.anchor_score,
                action_had_effect=diff.action_had_effect,
            )

        thumb = self._make_thumbnail_with_boxes(frame, crops)

        # Cap crops to 2 for token efficiency
        crop_imgs = [c["crop_after"] for c in crops[:2]]
        crop_bboxes = [c["bbox"] for c in crops[:2]]

        # Re-anchor after scroll since viewport position changed
        if self._infer_action_type(last_action) == "scroll":
            self._t0 = frame
            self._anchor = extract_anchor(frame, self.config)

        return DVObservation(
            obs_type="delta",
            step=self._step,
            trigger=cls.trigger,
            diff_ratio=cls.diff_ratio,
            phash_distance=cls.phash_distance,
            anchor_score=cls.anchor_score,
            action_had_effect=diff.action_had_effect,
            frame=None,
            thumbnail=thumb,
            crops=crop_imgs,
            crop_bboxes=crop_bboxes,
            url=url,
            last_action=last_action,
            # consumed_frame_size: drives dv_internal_tokens() so the cost
            # split survives the fact that we drop `frame` on delta steps.
            consumed_frame_size=(frame.width, frame.height),
        )

    # ---------------------------------------------------- internal helpers

    def _build_full_frame_obs(self, *, frame, trigger, url, last_action,
                              diff_ratio, phash_distance, anchor_score,
                              action_had_effect) -> DVObservation:
        return DVObservation(
            obs_type="full_frame",
            step=self._step,
            trigger=trigger,
            diff_ratio=diff_ratio,
            phash_distance=phash_distance,
            anchor_score=anchor_score,
            action_had_effect=action_had_effect,
            frame=frame,
            thumbnail=None,
            crops=[],
            crop_bboxes=[],
            url=url,
            last_action=last_action,
            consumed_frame_size=(frame.width, frame.height),
        )

    @staticmethod
    def _infer_action_type(last_action: str | None) -> str:
        if not last_action:
            return ""
        s = last_action.lower().strip()
        for kind in ("scroll", "click", "type", "key", "drag", "wait"):
            if s.startswith(kind) or f" {kind}" in s:
                return kind
        return ""

    @staticmethod
    def _make_thumbnail_with_boxes(frame: Image.Image, crops: list) -> Image.Image:
        thumb = frame.resize((320, 225), Image.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        sx = 320 / frame.width
        sy = 225 / frame.height
        for c in crops:
            x, y, w, h = c["bbox"]
            draw.rectangle(
                [
                    (int(x * sx) - 1, int(y * sy) - 1),
                    (int((x + w) * sx) + 1, int((y + h) * sy) + 1),
                ],
                outline=(0, 255, 0),
                width=2,
            )
        return thumb
