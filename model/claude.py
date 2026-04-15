"""
Claude backend. Uses vision API with structured JSON output.

Key decisions:
- System prompt distinguishes delta vs full_frame observation types
- Model is explicitly told it does NOT decide transition types
- Crop images sent in order of change_magnitude (largest first)
- Text deltas (Level 1) sent as pure text, no images
"""

import json
import base64
from io import BytesIO

import anthropic
from PIL import Image

from .base import BaseModel, ModelResponse
from agent.actions import parse_action

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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        raw_text = response.content[0].text
        parsed = json.loads(raw_text)

        action = parse_action(parsed.get("action")) if not parsed.get("done") else None

        return ModelResponse(
            action=action,
            done=parsed.get("done", False),
            reasoning=parsed.get("reasoning", ""),
            confidence=parsed.get("confidence", 0.0),
            raw_response=parsed,
        )

    def _build_messages(self, observation, state) -> list:
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
            header = (
                f"DELTA observation. Task: {observation.task}\n"
                f"Step: {observation.step}\n"
                f"Last action: {observation.last_action}\n"
                f"Action had effect: {observation.action_had_effect}\n"
                f"Consecutive no-effect steps: {observation.no_change_count}\n"
            )

            # Level 1: text deltas — cheapest path, no images needed
            if observation.text_deltas:
                header += "\nText changes detected (OCR):\n"
                for td in observation.text_deltas:
                    header += f"  [{td['bbox']}] \"{td['before']}\" -> \"{td['after']}\"\n"
                content.append({"type": "text", "text": header})

            else:
                # Level 2: image crops
                header += (
                    f"Diff ratio: {observation.diff_result.diff_ratio:.3f}\n"
                    f"Changed regions: {len(observation.crops)}\n\n"
                    f"Diff heatmap:"
                )
                content.append({"type": "text", "text": header})
                content.append(self._img_block(observation.diff_result.diff_image))

                for i, crop in enumerate(observation.crops):
                    content.append(
                        {
                            "type": "text",
                            "text": f"\nRegion {i + 1} (magnitude={crop['change_magnitude']:.3f}) — BEFORE:",
                        }
                    )
                    content.append(self._img_block(crop["crop_before"]))
                    content.append({"type": "text", "text": "AFTER:"})
                    content.append(self._img_block(crop["crop_after"]))

        return [{"role": "user", "content": content}]

    @staticmethod
    def _img_block(img: Image.Image) -> dict:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        }
