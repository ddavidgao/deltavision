"""
Tests for model/_response_parser.py — the shared JSON extraction + normalization
that protects every model backend from local-VLM output quirks.
"""

import pytest

from model._response_parser import (
    extract_json,
    normalize_response,
    get_confidence,
)


# ------------------------------------------------------------------ extract_json

class TestPureJSON:
    def test_simple_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_nested(self):
        assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}

    def test_with_whitespace(self):
        assert extract_json('  \n  {"a": 1}  \n  ') == {"a": 1}


class TestCodeFences:
    def test_json_fence(self):
        text = '```json\n{"reasoning": "hi", "done": false}\n```'
        assert extract_json(text) == {"reasoning": "hi", "done": False}

    def test_plain_fence(self):
        text = '```\n{"action": null}\n```'
        assert extract_json(text) == {"action": None}

    def test_fence_with_preamble(self):
        text = "Here is my response:\n```json\n{\"x\": 1}\n```"
        assert extract_json(text) == {"x": 1}

    def test_fence_priority(self):
        """When the outer text also has braces, the fence content wins."""
        text = 'Before { not json } ```json\n{"real": true}\n```'
        assert extract_json(text) == {"real": True}


class TestBraceExtraction:
    def test_preamble(self):
        text = 'I will click on the button: {"action": {"type": "click"}, "done": false}'
        result = extract_json(text)
        assert result["action"]["type"] == "click"

    def test_postamble(self):
        text = '{"action": {"type": "done"}, "done": true} -- finished!'
        result = extract_json(text)
        assert result["done"] is True

    def test_multiple_braces_takes_outermost(self):
        text = '{"outer": {"inner": 1}, "other": 2}'
        result = extract_json(text)
        assert result["other"] == 2
        assert result["outer"]["inner"] == 1


class TestFallback:
    def test_empty_string(self):
        result = extract_json("")
        assert result["done"] is True
        assert result["action"] is None
        assert result["confidence"] == 0.0

    def test_none_input(self):
        result = extract_json(None)
        assert result["done"] is True

    def test_garbage_text(self):
        result = extract_json("Just some random text without any braces")
        assert result["done"] is True
        assert "random text" in result["reasoning"]

    def test_unclosed_json(self):
        """Unterminated JSON falls through to fallback since brace-extraction fails."""
        result = extract_json('{"action": {"type":')
        assert result["done"] is True

    def test_partially_parseable_returns_fallback(self):
        """If outer braces don't form valid JSON, fallback wins."""
        result = extract_json("random {not json at all} more text")
        assert result["done"] is True


# --------------------------------------------------------------- normalize_response

class TestNormalizeConfidenceHoisting:
    def test_hoists_from_action(self):
        """MAI-UI-8B nests confidence inside action."""
        raw = {"reasoning": "x", "action": {"type": "click", "confidence": 0.9}, "done": False}
        norm = normalize_response(raw)
        assert norm["confidence"] == 0.9

    def test_top_level_confidence_preferred(self):
        raw = {"confidence": 0.7, "action": {"type": "click", "confidence": 0.9}}
        norm = normalize_response(raw)
        assert norm["confidence"] == 0.7

    def test_no_confidence_defaults_to_zero(self):
        raw = {"reasoning": "x", "action": {"type": "click"}}
        norm = normalize_response(raw)
        assert norm["confidence"] == 0.0


class TestNormalizeAltDoneFields:
    @pytest.mark.parametrize("alt_name", ["finish", "finished", "complete", "is_done"])
    def test_alt_done_field_normalized(self, alt_name):
        raw = {"reasoning": "x", "action": None, alt_name: True}
        norm = normalize_response(raw)
        assert norm["done"] is True

    def test_existing_done_wins(self):
        raw = {"done": False, "finish": True}
        norm = normalize_response(raw)
        assert norm["done"] is False


class TestNormalizeDefaults:
    def test_fills_missing_fields(self):
        norm = normalize_response({})
        assert norm["reasoning"] == ""
        assert norm["done"] is False
        assert norm["confidence"] == 0.0
        assert norm["action"] is None

    def test_non_dict_input(self):
        """If we got e.g. a list or string, fall back safely."""
        norm = normalize_response(["not", "a", "dict"])
        assert norm["done"] is True
        assert norm["action"] is None


# ------------------------------------------------------------------- get_confidence

class TestGetConfidence:
    def test_float(self):
        assert get_confidence({"confidence": 0.85}) == 0.85

    def test_int(self):
        assert get_confidence({"confidence": 1}) == 1.0

    def test_string_numeric(self):
        """Some local VLMs emit numbers as strings."""
        assert get_confidence({"confidence": "0.75"}) == 0.75

    def test_none(self):
        assert get_confidence({"confidence": None}) == 0.0

    def test_missing(self):
        assert get_confidence({}) == 0.0

    def test_garbage_string(self):
        assert get_confidence({"confidence": "high"}) == 0.0

    def test_clamped_above(self):
        """Some models emit 1.5 — clamp to 1.0."""
        assert get_confidence({"confidence": 1.5}) == 1.0

    def test_clamped_below(self):
        assert get_confidence({"confidence": -0.3}) == 0.0
