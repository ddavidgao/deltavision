"""
Ollama backend — run ANY local model (Hermes, Qwen2.5-VL, LLaVA, etc.)
via Ollama's HTTP API. Zero Python dependencies beyond requests.

Supports two modes:
1. VLM mode (Qwen2.5-VL, LLaVA, etc.) — sends images directly
2. Text-only mode (Hermes, Llama, etc.) — converts observations to text
   descriptions, no images. Relies on DeltaVision's text deltas and
   structured diff metadata instead of raw pixels.

Usage:
  # VLM — sees images
  model = OllamaModel("qwen2.5-vl:7b")

  # Text-only — gets structured text descriptions
  model = OllamaModel("hermes3:8b", vision=False)

Requires: ollama running locally (ollama serve)
"""

import base64
from io import BytesIO
from typing import Optional

import requests
from PIL import Image

from .base import BaseModel, ModelResponse
from ._response_parser import extract_json, normalize_response, get_confidence
from agent.actions import parse_action
from model.claude import SYSTEM_PROMPT


class OllamaModel(BaseModel):
    def __init__(
        self,
        model: str = "qwen2.5-vl:7b",
        host: str = "http://localhost:11434",
        vision: bool = True,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.vision = vision

    async def predict(self, observation, state) -> ModelResponse:
        if self.vision:
            prompt, images = self._build_vision_prompt(observation)
        else:
            prompt = self._build_text_prompt(observation)
            images = []

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 512, "temperature": 0.1},
        }
        if images:
            payload["images"] = images

        # Retry on ANY transient failure: HTTP 5xx, connection reset, timeout.
        # This layer is what keeps a run alive when Ollama GCs a model or
        # the VRAM allocator briefly blocks.
        import logging
        import time as _time
        log = logging.getLogger(__name__)

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
                resp.raise_for_status()
                last_err = None
                break
            except (requests.exceptions.HTTPError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", "no-status")
                body = getattr(getattr(e, "response", None), "text", "") or ""
                log.warning(
                    "Ollama error (attempt %d/3): %s %s — %s",
                    attempt + 1, type(e).__name__, status, body[:300],
                )
                if attempt < 2:
                    _time.sleep(3 * (attempt + 1))  # backoff: 3s, 6s
                    continue
        if last_err is not None:
            raise last_err

        raw_text = resp.json()["response"]
        parsed = normalize_response(extract_json(raw_text))

        action = parse_action(parsed.get("action")) if not parsed.get("done") else None

        return ModelResponse(
            action=action,
            done=parsed.get("done", False),
            reasoning=parsed.get("reasoning", ""),
            confidence=get_confidence(parsed),
            raw_response=parsed,
        )

    def _build_vision_prompt(self, observation) -> tuple:
        """Build prompt + base64 images for VLM models."""
        images = []

        if observation.obs_type == "full_frame":
            prompt = (
                f"FULL_FRAME observation. Task: {observation.task}\n"
                f"Step: {observation.step}\n"
                f"URL: {observation.url}\n"
                f"Last action: {observation.last_action}\n"
                f"Full screen shown in image 1."
            )
            images.append(self._img_b64(observation.frame))
        else:
            prompt = (
                f"DELTA observation. Task: {observation.task}\n"
                f"Step: {observation.step}\n"
                f"Last action: {observation.last_action}\n"
                f"Action had effect: {observation.action_had_effect}\n"
                f"No-effect streak: {observation.no_change_count}\n"
            )
            if observation.text_deltas:
                prompt += "\nText changes:\n"
                for td in observation.text_deltas:
                    prompt += f"  \"{td['before']}\" -> \"{td['after']}\"\n"
            elif observation.diff_result:
                prompt += f"Diff ratio: {observation.diff_result.diff_ratio:.3f}\n"
                images.append(self._img_b64(observation.diff_result.diff_image))
                for i, crop in enumerate(observation.crops[:3]):  # fewer for local
                    prompt += f"\nImage {len(images)+1}: Region {i+1} BEFORE, Image {len(images)+2}: AFTER"
                    images.append(self._img_b64(crop["crop_before"]))
                    images.append(self._img_b64(crop["crop_after"]))

        return prompt, images

    def _build_text_prompt(self, observation) -> str:
        """Build text-only prompt for non-VLM models (Hermes, Llama, etc.).
        Converts visual observations into structured text descriptions.
        This is where DeltaVision's text deltas shine — the model gets
        actionable info without needing to see pixels."""

        if observation.obs_type == "full_frame":
            return (
                f"FULL_FRAME observation (you cannot see images — working from metadata).\n"
                f"Task: {observation.task}\n"
                f"Step: {observation.step}\n"
                f"URL: {observation.url}\n"
                f"Trigger: {observation.trigger_reason}\n"
                f"Last action: {observation.last_action}\n\n"
                f"A new page or context has loaded. You need to decide your next action "
                f"based on the task description and URL. If you need to see the page, "
                f"consider a 'wait' action to let the page settle."
            )

        parts = [
            f"DELTA observation (text-only mode).",
            f"Task: {observation.task}",
            f"Step: {observation.step}",
            f"Last action: {observation.last_action}",
            f"Action had effect: {observation.action_had_effect}",
            f"No-effect streak: {observation.no_change_count}",
        ]

        if observation.text_deltas:
            parts.append("\nDetected text changes:")
            for td in observation.text_deltas:
                parts.append(f'  Region {td["bbox"]}: "{td["before"]}" -> "{td["after"]}"')
        elif observation.diff_result:
            parts.append(f"\nDiff ratio: {observation.diff_result.diff_ratio:.3f}")
            parts.append(f"Changed regions: {len(observation.crops)}")
            for i, crop in enumerate(observation.crops):
                parts.append(
                    f"  Region {i+1}: bbox={crop['bbox']}, "
                    f"magnitude={crop['change_magnitude']:.3f}"
                )

        return "\n".join(parts)

    @staticmethod
    def _img_b64(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode()
