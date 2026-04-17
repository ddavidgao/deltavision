"""
Claude backend. Uses vision API with structured JSON output.

Key decisions:
- System prompt distinguishes delta vs full_frame observation types
- Model is explicitly told it does NOT decide transition types
- Crop images sent in order of change_magnitude (largest first)
- Text deltas (Level 1) sent as pure text, no images
- Transient API errors (429, 503, connection reset) retry with exponential backoff
"""

import json
import base64
import logging
import os
import time
from io import BytesIO

import anthropic
from PIL import Image

from .base import BaseModel, ModelResponse
from ._response_parser import extract_json, normalize_response, get_confidence
from agent.actions import parse_action

log = logging.getLogger(__name__)

# Anthropic API errors that are worth retrying vs permanent failures
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

SYSTEM_PROMPT = """You are a GUI automation agent operating in DeltaVision mode.

Your observation type determines what you receive:

FULL_FRAME observations: You see the entire screen. This happens on initial load,
after navigation, or when the system forces a refresh. Use this to understand
the full page context.

DELTA observations: You see only what CHANGED since your last action. You receive:
- A diff heatmap showing where changes occurred
- Before/after crops of each changed region (sorted by size, largest first)
- OR text deltas if the change was small enough for OCR (fastest path)
- Whether your last action had a visible effect

CRITICAL RULES:
- You do NOT decide whether you're on a new page. The system handles that.
- If action_had_effect is False: your last action did nothing. Try a different
  approach — different element, different coordinates, scroll first, or wait.
- If no_change_count >= 2: you are stuck. Think creatively. Do NOT repeat the
  same failed action.
- For DELTA observations: reason about what changed and what it means for your
  task. Do not re-describe the full page.

Respond ONLY with valid JSON:
{
  "reasoning": "brief explanation of what you observe and why you chose this action",
  "action": {
    "type": "click|type|scroll|key|wait|done",
    "x": int,
    "y": int,
    "text": str,
    "direction": str,
    "amount": int,
    "key": str,
    "duration_ms": int
  },
  "done": false,
  "confidence": 0.85
}

If the task is complete, set done=true and action.type="done".
If you cannot proceed, set action=null and done=true with reasoning."""


class ClaudeModel(BaseModel):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def predict(self, observation, state) -> ModelResponse:
        messages = self._build_messages(observation, state)

        # Retry transient errors. Permanent errors (auth, bad request) bubble up.
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                last_err = None
                break
            except _RETRYABLE as e:
                last_err = e
                log.warning(
                    "Claude API transient error (attempt %d/3): %s — %s",
                    attempt + 1, type(e).__name__, str(e)[:200],
                )
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s
        if last_err is not None:
            raise last_err

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

    def _build_messages(self, observation, state) -> list:
        messages = []

        # Include recent history so the model remembers what it already tried.
        # Keep last N turns (observation + response pairs) to stay within context.
        # This is where delta observations pay off: delta history messages are
        # text-only or single-image, cheaper than full-frame history.
        max_history = 5
        obs_history = state.observations[-(max_history + 1):-1]  # exclude current
        resp_history = state.responses[-max_history:]

        for prev_obs, prev_resp in zip(obs_history, resp_history):
            prev_content = []

            if prev_obs.obs_type == "full_frame" and prev_obs.frame:
                # Include the actual screenshot in history (matches real Anthropic CU behavior)
                prev_content.append({"type": "text", "text": (
                    f"[Step {prev_obs.step}] Full screen after: {prev_obs.last_action}"
                )})
                prev_content.append(self._img_block(prev_obs.frame))
            elif prev_obs.obs_type == "delta":
                # Delta history: TEXT ONLY — no screenshot needed.
                # The model already has the full page from the last NEW_PAGE.
                # Sending text-only delta history is where DeltaVision saves tokens.
                parts = [f"[Step {prev_obs.step}] After: {prev_obs.last_action}"]
                if hasattr(prev_obs, 'action_had_effect'):
                    parts.append(f"Effect: {prev_obs.action_had_effect}")
                if hasattr(prev_obs, 'diff_result') and prev_obs.diff_result:
                    parts.append(f"Change: {prev_obs.diff_result.diff_ratio:.1%}")
                prev_content.append({"type": "text", "text": " | ".join(parts)})
            else:
                prev_content.append({"type": "text", "text": (
                    f"[Step {prev_obs.step}] After: {prev_obs.last_action}"
                )})

            messages.append({"role": "user", "content": prev_content})

            # Previous response (assistant turn)
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(prev_resp.raw_response) if prev_resp.raw_response else "{}"}
            ]})

        # Current observation (full content with image)
        content = []

        if observation.obs_type == "full_frame":
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"FULL_FRAME observation. Task: {observation.task}\n"
                        f"Step: {observation.step}\n"
                        f"URL: {observation.url}\n"
                        f"Trigger: {observation.trigger_reason}\n"
                        f"Last action: {observation.last_action}\n\n"
                        f"Full screen:"
                    ),
                }
            )
            content.append(self._img_block(observation.frame))

        else:  # delta
            # Delta observation: send the CURRENT screenshot (one image)
            # plus text metadata about what changed. The CV pipeline's value
            # is in the classification and metadata, not in fancy visual diffs.
            # One image = consistent token cost, no per-image overhead bloat.
            parts = [
                f"DELTA observation (same page, partial change). Task: {observation.task}",
                f"Step: {observation.step}",
                f"Last action: {observation.last_action}",
                f"Action had effect: {observation.action_had_effect}",
            ]

            if observation.no_change_count > 0:
                parts.append(f"WARNING: {observation.no_change_count} consecutive actions had no effect. Try something different.")

            if observation.text_deltas:
                parts.append("\nDetected text changes:")
                for td in observation.text_deltas:
                    parts.append(f"  Region {td['bbox']}: \"{td['before']}\" -> \"{td['after']}\"")

            elif observation.diff_result:
                parts.append(f"\nPixel change: {observation.diff_result.diff_ratio:.1%} of screen")
                if observation.crops:
                    parts.append(f"Changed regions ({len(observation.crops)}):")
                    for i, crop in enumerate(observation.crops):
                        x, y, w, h = crop["bbox"]
                        parts.append(f"  Region {i+1}: ({x},{y}) {w}x{h}px, magnitude={crop['change_magnitude']:.3f}")

            content.append({"type": "text", "text": "\n".join(parts)})

            # Video-compression-style delta observation:
            # 1. Low-res thumbnail of current page (spatial context, ~100 tokens)
            # 2. High-res crop(s) of what changed (detail, ~100 tokens each)
            # Total: ~200-300 tokens vs ~500 for a full screenshot
            if observation.current_frame:
                from PIL import ImageDraw

                # Thumbnail: 320x225 with green box showing where the change is
                thumb = observation.current_frame.resize((320, 225), Image.LANCZOS)
                draw = ImageDraw.Draw(thumb)
                scale_x = 320 / observation.current_frame.width
                scale_y = 225 / observation.current_frame.height
                for crop in observation.crops:
                    x, y, w, h = crop["bbox"]
                    draw.rectangle([
                        (int(x * scale_x) - 1, int(y * scale_y) - 1),
                        (int((x + w) * scale_x) + 1, int((y + h) * scale_y) + 1)
                    ], outline=(0, 255, 0), width=2)

                content.append({"type": "text", "text": "Page overview (low-res, green box = changed area):"})
                content.append(self._img_block(thumb))

                # High-res crops of changed regions
                if observation.crops:
                    crops = observation.crops[:2]  # max 2 crops
                    for i, c in enumerate(crops):
                        after = c["crop_after"]
                        # Cap crop size to keep tokens low
                        max_dim = 400
                        if after.width > max_dim or after.height > max_dim:
                            ratio = min(max_dim / after.width, max_dim / after.height)
                            after = after.resize((int(after.width * ratio), int(after.height * ratio)), Image.LANCZOS)
                        x, y, w, h = c["bbox"]
                        content.append({"type": "text", "text": f"Changed region {i+1} at ({x},{y}) — detail:"})
                        content.append(self._img_block(after))

            elif observation.diff_result and observation.diff_result.diff_image:
                content.append(self._img_block(observation.diff_result.diff_image))

        messages.append({"role": "user", "content": content})

        # Save the exact images sent to the model (for demo/debugging)
        save_dir = os.environ.get("DELTAVISION_SAVE_OBS")
        if save_dir:
            from pathlib import Path
            d = Path(save_dir)
            d.mkdir(parents=True, exist_ok=True)
            step = observation.step
            # Extract ALL images from content blocks
            img_idx = 0
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    import base64 as _b64
                    img_data = _b64.standard_b64decode(block["source"]["data"])
                    img = Image.open(BytesIO(img_data))
                    suffix = "" if img_idx == 0 else f"_{img_idx}"
                    img.save(d / f"step_{step:03d}_sent_to_model{suffix}.png")
                    img_idx += 1
            # Also save the text prompt
            text_parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            with open(d / f"step_{step:03d}_prompt.txt", "w") as f:
                f.write("\n".join(text_parts))

        return messages

    @staticmethod
    def _img_block(img: Image.Image) -> dict:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        }
