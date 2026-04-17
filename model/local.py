"""
Local VLM backend — Qwen2.5-VL via transformers.

Target hardware: RTX 5080 laptop (16GB VRAM)
  - Qwen2.5-VL-7B-Instruct fp16 ≈ 14GB VRAM
  - Qwen2.5-VL-3B-Instruct fp16 ≈ 6GB VRAM (faster, good for iteration)
  - 4-bit quantization cuts VRAM roughly in half

Install:
  pip install torch torchvision transformers accelerate qwen-vl-utils bitsandbytes
"""

import base64
from io import BytesIO
from typing import Optional

from PIL import Image

from .base import BaseModel, ModelResponse
from ._response_parser import extract_json, normalize_response, get_confidence
from agent.actions import parse_action

# Same structured output contract as Claude backend
SYSTEM_PROMPT = """You are a GUI automation agent operating in DeltaVision mode.

Your observation type determines what you receive:

FULL_FRAME observations: You see the entire screen. Understand the full page context.

DELTA observations: You see only what CHANGED since your last action:
- A diff heatmap showing where changes occurred
- Before/after crops of each changed region
- OR text deltas if the change was small enough for OCR
- Whether your last action had a visible effect

RULES:
- If action_had_effect is False: try a different approach.
- If no_change_count >= 2: you are stuck. Do NOT repeat the same action.

Respond ONLY with valid JSON:
{
  "reasoning": "brief explanation",
  "action": {
    "type": "click|type|scroll|key|wait|done",
    "x": int, "y": int,
    "text": str,
    "direction": str, "amount": int,
    "key": str, "duration_ms": int
  },
  "done": false,
  "confidence": 0.85
}"""


class LocalModel(BaseModel):
    """
    Qwen2.5-VL (or compatible) running locally via transformers.
    Lazy-loads on first predict() call to avoid slow imports at startup.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", quantization: Optional[str] = None):
        self.model_name = model_name
        self.quantization = quantization
        self._model = None
        self._processor = None

    def _load(self):
        """Lazy load model + processor."""
        if self._model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        load_kwargs = {"device_map": "auto", "torch_dtype": "auto"}

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )

    async def predict(self, observation, state) -> ModelResponse:
        self._load()

        messages = self._build_messages(observation)

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Collect all images from the message content
        images = self._extract_images(messages)

        inputs = self._processor(
            text=[text_prompt],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        output_ids = self._model.generate(**inputs, max_new_tokens=512)

        # Trim input tokens from output
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        raw_text = self._processor.decode(generated, skip_special_tokens=True)

        parsed = normalize_response(extract_json(raw_text))
        action = parse_action(parsed.get("action")) if not parsed.get("done") else None

        return ModelResponse(
            action=action,
            done=parsed.get("done", False),
            reasoning=parsed.get("reasoning", ""),
            confidence=get_confidence(parsed),
            raw_response=parsed,
        )

    def _build_messages(self, observation) -> list:
        """Build Qwen2.5-VL chat messages with interleaved images."""
        content = []

        if observation.obs_type == "full_frame":
            content.append({"type": "text", "text": (
                f"FULL_FRAME observation. Task: {observation.task}\n"
                f"Step: {observation.step}\n"
                f"URL: {observation.url}\n"
                f"Last action: {observation.last_action}\n\n"
                f"Full screen:"
            )})
            content.append({"type": "image", "image": observation.frame})

        else:
            header = (
                f"DELTA observation. Task: {observation.task}\n"
                f"Step: {observation.step}\n"
                f"Last action: {observation.last_action}\n"
                f"Action had effect: {observation.action_had_effect}\n"
                f"No-effect streak: {observation.no_change_count}\n"
            )

            if observation.text_deltas:
                header += "\nText changes (OCR):\n"
                for td in observation.text_deltas:
                    header += f"  \"{td['before']}\" -> \"{td['after']}\"\n"
                content.append({"type": "text", "text": header})
            else:
                header += f"Diff ratio: {observation.diff_result.diff_ratio:.3f}\nDiff heatmap:"
                content.append({"type": "text", "text": header})
                content.append({"type": "image", "image": observation.diff_result.diff_image})

                for i, crop in enumerate(observation.crops[:4]):  # Fewer crops for local model token budget
                    content.append({"type": "text", "text": f"Region {i+1} BEFORE:"})
                    content.append({"type": "image", "image": crop["crop_before"]})
                    content.append({"type": "text", "text": "AFTER:"})
                    content.append({"type": "image", "image": crop["crop_after"]})

        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _extract_images(messages: list) -> list:
        """Pull PIL images from message content for processor."""
        images = []
        for msg in messages:
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image")
                    if isinstance(img, Image.Image):
                        images.append(img)
        return images
