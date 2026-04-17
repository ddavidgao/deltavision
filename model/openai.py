"""
OpenAI GPT-4o / GPT-4V backend.
Same observation format as Claude — just different API.

Also covers any OpenAI-compatible endpoint (llama-server, vLLM, SGLang) via
the base_url parameter — same retry behavior applies.
"""

import base64
import logging
import time
from io import BytesIO

from PIL import Image

from .base import BaseModel, ModelResponse
from ._response_parser import extract_json, normalize_response, get_confidence
from agent.actions import parse_action
from model.claude import SYSTEM_PROMPT  # same prompt works

log = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    def __init__(
        self,
        api_key: str = "sk-no-key-required",
        model: str = "gpt-4o",
        base_url: str = None,
    ):
        """OpenAI-compatible backend.

        base_url: optional override for OpenAI-compatible servers
            (e.g. llama-server, vLLM, SGLang).
            Default None → OpenAI official API.
            Example local: base_url="http://localhost:8080/v1"
        """
        import openai
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)
        self.model = model
        self.is_local = base_url is not None

    async def predict(self, observation, state) -> ModelResponse:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_content(observation)},
        ]

        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": messages,
        }

        if self.is_local:
            # Strict JSON schema for local VLMs (llama.cpp supports this via b4xxx+).
            # Forces schema compliance — prevents common failures like dropping key names
            # or emitting two values under one key (observed with MAI-UI-8B).
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "deltavision_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "action": {
                                "type": ["object", "null"],
                                "properties": {
                                    "type": {"type": "string", "enum": ["click", "type", "scroll", "key", "wait", "done"]},
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                    "text": {"type": "string"},
                                    "direction": {"type": "string"},
                                    "amount": {"type": "integer"},
                                    "key": {"type": "string"},
                                    "duration_ms": {"type": "integer"},
                                },
                            },
                            "done": {"type": "boolean"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["reasoning", "action", "done"],
                    },
                },
            }
        else:
            # Cloud OpenAI — standard json_object mode is reliable.
            kwargs["response_format"] = {"type": "json_object"}

        # Retryable errors across both cloud OpenAI and local OpenAI-compat servers.
        import openai
        retryable = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )

        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**kwargs)
                last_err = None
                break
            except retryable as e:
                last_err = e
                log.warning(
                    "OpenAI-compat API transient error (attempt %d/3, is_local=%s): %s — %s",
                    attempt + 1, self.is_local, type(e).__name__, str(e)[:200],
                )
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if last_err is not None:
            raise last_err

        raw_text = response.choices[0].message.content
        parsed = normalize_response(extract_json(raw_text))

        action = parse_action(parsed.get("action")) if not parsed.get("done") else None

        return ModelResponse(
            action=action,
            done=parsed.get("done", False),
            reasoning=parsed.get("reasoning", ""),
            confidence=get_confidence(parsed),
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
