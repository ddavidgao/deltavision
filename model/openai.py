"""
OpenAI GPT-4o / GPT-4V backend.
Same observation format as Claude — just different API.
"""

import json
import base64
from io import BytesIO

from PIL import Image

from .base import BaseModel, ModelResponse
from agent.actions import parse_action
from model.claude import SYSTEM_PROMPT  # same prompt works


class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    async def predict(self, observation, state) -> ModelResponse:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_content(observation)},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=messages,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content
        parsed = json.loads(raw_text)

        action = parse_action(parsed.get("action")) if not parsed.get("done") else None

        return ModelResponse(
            action=action,
            done=parsed.get("done", False),
            reasoning=parsed.get("reasoning", ""),
            confidence=parsed.get("confidence", 0.0),
            raw_response=parsed,
        )

    def _build_content(self, observation) -> list:
        content = []

        if observation.obs_type == "full_frame":
            content.append({
                "type": "text",
                "text": (
                    f"FULL_FRAME observation. Task: {observation.task}\n"
                    f"Step: {observation.step}\n"
                    f"URL: {observation.url}\n"
                    f"Trigger: {observation.trigger_reason}\n"
                    f"Last action: {observation.last_action}\n\nFull screen:"
                ),
            })
            content.append(self._img_block(observation.frame))
        else:
            header = (
                f"DELTA observation. Task: {observation.task}\n"
                f"Step: {observation.step}\n"
                f"Last action: {observation.last_action}\n"
                f"Action had effect: {observation.action_had_effect}\n"
                f"Consecutive no-effect steps: {observation.no_change_count}\n"
            )

            if observation.text_deltas:
                header += "\nText changes (OCR):\n"
                for td in observation.text_deltas:
                    header += f"  \"{td['before']}\" -> \"{td['after']}\"\n"
                content.append({"type": "text", "text": header})
            else:
                header += f"Diff ratio: {observation.diff_result.diff_ratio:.3f}\nDiff heatmap:"
                content.append({"type": "text", "text": header})
                content.append(self._img_block(observation.diff_result.diff_image))
                for i, crop in enumerate(observation.crops):
                    content.append({"type": "text", "text": f"Region {i+1} BEFORE:"})
                    content.append(self._img_block(crop["crop_before"]))
                    content.append({"type": "text", "text": "AFTER:"})
                    content.append(self._img_block(crop["crop_after"]))

        return content

    @staticmethod
    def _img_block(img: Image.Image) -> dict:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
