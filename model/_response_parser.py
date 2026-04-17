"""
Shared response parsing for model backends.

Local VLMs (MAI-UI-8B, Qwen-VL, Hermes, etc.) frequently emit non-strict JSON:
markdown code fences, prose preambles, nested confidence fields, etc.
Centralizing the parsing here keeps behavior consistent across backends
and makes it unit-testable without spinning up an API client.
"""

import json
import re
from typing import Any, Optional


# Markdown code fences: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw_text: str) -> dict:
    """Robustly extract a JSON object from model output.

    Handles (in order):
      1. Pure JSON (fast path)
      2. Markdown code fences: ```json {...} ```
      3. Prose preamble + JSON: "Here's my response: {...}"
      4. JSON embedded in prose with trailing text: "{...} Hope this helps!"
      5. Total garbage → returns a safe done=True fallback dict with the raw
         text as reasoning so the caller can log it.

    The fallback dict has the same shape as a valid response so callers
    don't need special-case handling — they can treat malformed responses
    as "model gave up, stop the run."
    """
    if raw_text is None:
        return _safe_fallback("")

    text = raw_text.strip()
    if not text:
        return _safe_fallback("")

    # 1. Pure JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fence
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # 3-4. Brace-extraction: find the outermost {...} block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # 5. Fallback
    return _safe_fallback(text)


def _safe_fallback(raw_text: str) -> dict:
    return {
        "reasoning": raw_text or "[empty response]",
        "action": None,
        "done": True,
        "confidence": 0.0,
    }


def normalize_response(parsed: dict) -> dict:
    """Smooth out VLM-specific quirks so callers see a consistent shape.

    Observed quirks:
      - MAI-UI-8B nests confidence inside action
      - Some Qwen variants emit "finish" instead of "done"
      - Some models return action as a top-level field, not nested
    """
    if not isinstance(parsed, dict):
        return _safe_fallback(str(parsed))

    out = dict(parsed)

    # Hoist confidence from action if it's nested there.
    if out.get("confidence") is None and isinstance(out.get("action"), dict):
        nested = out["action"].get("confidence")
        if nested is not None:
            out["confidence"] = nested

    # Normalize alternate "done" field names.
    for alt in ("finish", "finished", "complete", "is_done"):
        if alt in out and "done" not in out:
            out["done"] = out[alt]
            break

    # Defaults for missing fields.
    out.setdefault("reasoning", "")
    out.setdefault("done", False)
    out.setdefault("confidence", 0.0)
    out.setdefault("action", None)

    return out


def get_confidence(parsed: dict) -> float:
    """Coerce confidence to float in [0, 1], handling strings and missing values."""
    v = parsed.get("confidence", 0.0)
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))
