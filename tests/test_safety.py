"""
Safety layer tests. Critical because safety.py runs regardless of backend —
these checks are what protect against uncensored local VLMs.

Untested code in this layer = a bug here can expose credentials or navigate
to malicious URLs. Every branch in safety.py needs coverage.
"""

import pytest

from agent.actions import Action, ActionType
from safety import SafetyLayer, SafetyResult, PERMISSIVE, STRICT, EDUCATIONAL


# ------------------------------------------------------------------- helpers

def click(x=100, y=100):
    return Action(type=ActionType.CLICK, x=x, y=y)


def type_(text):
    return Action(type=ActionType.TYPE, text=text)


def scroll(direction="down", amount=300):
    return Action(type=ActionType.SCROLL, direction=direction, amount=amount)


# ------------------------------------------------------------------- URL safety

class TestURLShorteners:
    def test_bitly_blocked(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://bit.ly/abc123")
        assert not result.allowed
        assert "shortener" in result.reason.lower()
        assert result.severity == "block"

    def test_tinyurl_blocked(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://tinyurl.com/xyz")
        assert not result.allowed

    def test_tco_blocked(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://t.co/short")
        assert not result.allowed

    def test_shortener_disabled(self):
        s = SafetyLayer(block_url_shorteners=False)
        result = s.check_action(click(), "https://bit.ly/abc123")
        assert result.allowed

    def test_normal_url_allowed(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://en.wikipedia.org/wiki/Test")
        assert result.allowed


class TestSuspiciousPatterns:
    def test_russian_domain_blocked(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://example.ru/something")
        assert not result.allowed

    def test_password_reset_blocked(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://example.com/password-reset")
        assert not result.allowed

    def test_account_verify_blocked(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://example.com/account-verify-now")
        assert not result.allowed


class TestDomainAllowlist:
    def test_allowed_domain_passes(self):
        s = SafetyLayer(allowed_domains={"wikipedia.org"})
        result = s.check_action(click(), "https://en.wikipedia.org/wiki/Test")
        assert result.allowed

    def test_disallowed_domain_blocked(self):
        s = SafetyLayer(allowed_domains={"wikipedia.org"})
        result = s.check_action(click(), "https://evil.com/stuff")
        assert not result.allowed
        assert "allowlist" in result.reason.lower()

    def test_no_allowlist_permissive(self):
        """With allowed_domains=None, all non-suspicious domains pass."""
        s = SafetyLayer(allowed_domains=None)
        result = s.check_action(click(), "https://random-new-site.com")
        assert result.allowed

    def test_empty_url_passes(self):
        s = SafetyLayer()
        result = s.check_action(click(), "")
        assert result.allowed


# ------------------------------------------------------------------- type safety

class TestCredentialDetection:
    def test_ssn_blocked(self):
        s = SafetyLayer()
        result = s.check_action(type_("123456789"), "https://example.com")
        assert not result.allowed

    def test_ssn_with_dashes_blocked(self):
        s = SafetyLayer()
        result = s.check_action(type_("123-45-6789"), "https://example.com")
        assert not result.allowed

    def test_credit_card_blocked(self):
        s = SafetyLayer()
        result = s.check_action(type_("4532123456789012"), "https://example.com")
        assert not result.allowed

    def test_credit_card_with_spaces_blocked(self):
        s = SafetyLayer()
        result = s.check_action(type_("4532 1234 5678 9012"), "https://example.com")
        assert not result.allowed

    def test_cvv_blocked(self):
        s = SafetyLayer()
        result = s.check_action(type_("123"), "https://example.com")
        assert not result.allowed

    def test_normal_text_allowed(self):
        s = SafetyLayer()
        result = s.check_action(type_("hello world this is fine"), "https://example.com")
        assert result.allowed

    def test_search_query_allowed(self):
        s = SafetyLayer()
        result = s.check_action(type_("neural networks explained"), "https://google.com")
        assert result.allowed

    def test_credential_detection_disabled(self):
        s = SafetyLayer(block_credential_entry=False)
        result = s.check_action(type_("123456789"), "https://example.com")
        assert result.allowed


class TestSensitiveFieldContext:
    def test_password_field_context_blocked(self):
        s = SafetyLayer()
        result = s.check_action(
            type_("hello"),
            "https://example.com",
            page_context="<input type='password' name='login-password'>",
        )
        assert not result.allowed
        assert "credential" in result.reason.lower()

    def test_ssn_field_context_blocked(self):
        s = SafetyLayer()
        result = s.check_action(
            type_("hello"),
            "https://example.com",
            page_context="Please enter your SSN:",
        )
        assert not result.allowed

    def test_benign_context_allowed(self):
        s = SafetyLayer()
        result = s.check_action(
            type_("search query"),
            "https://example.com",
            page_context="<input name='search' placeholder='search'>",
        )
        assert result.allowed


# ------------------------------------------------------------------- action limits

class TestActionLimits:
    def test_oversized_type_blocked(self):
        s = SafetyLayer(max_type_length=100)
        result = s.check_action(type_("x" * 200), "https://example.com")
        assert not result.allowed
        assert "too long" in result.reason.lower()

    def test_exact_limit_allowed(self):
        s = SafetyLayer(max_type_length=100)
        result = s.check_action(type_("x" * 100), "https://example.com")
        assert result.allowed

    def test_negative_click_blocked(self):
        s = SafetyLayer()
        result = s.check_action(click(x=-5, y=100), "https://example.com")
        assert not result.allowed
        assert "negative" in result.reason.lower()

    def test_positive_click_allowed(self):
        s = SafetyLayer()
        result = s.check_action(click(x=500, y=500), "https://example.com")
        assert result.allowed

    def test_click_without_coords_allowed(self):
        """Some actions (keyboard-like) have no coords — don't crash."""
        a = Action(type=ActionType.CLICK, x=None, y=None)
        s = SafetyLayer()
        result = s.check_action(a, "https://example.com")
        assert result.allowed


# ------------------------------------------------------------------- presets

class TestPresets:
    def test_permissive_allows_long_search(self):
        result = PERMISSIVE.check_action(
            type_("a long search query with many words " * 10),
            "https://google.com",
        )
        # 10 * 37 = 370 chars, under PERMISSIVE's 1000
        assert result.allowed

    def test_strict_blocks_long_type(self):
        result = STRICT.check_action(
            type_("a" * 300),
            "https://google.com",
        )
        # STRICT has max_type_length=200
        assert not result.allowed

    def test_educational_blocks_non_edu(self):
        result = EDUCATIONAL.check_action(
            click(),
            "https://random.com/thing",
        )
        assert not result.allowed

    def test_educational_allows_mheducation(self):
        result = EDUCATIONAL.check_action(
            click(),
            "https://connect.mheducation.com/assessment",
        )
        assert result.allowed


# ------------------------------------------------------------------- non-typed actions pass through

class TestNonTypeActions:
    def test_click_not_subject_to_type_checks(self):
        s = SafetyLayer()
        result = s.check_action(click(), "https://example.com")
        assert result.allowed

    def test_scroll_passes(self):
        s = SafetyLayer()
        result = s.check_action(scroll(), "https://example.com")
        assert result.allowed

    def test_key_passes(self):
        s = SafetyLayer()
        result = s.check_action(
            Action(type=ActionType.KEY, key="Enter"),
            "https://example.com",
        )
        assert result.allowed


# ------------------------------------------------------------------- result shape

class TestSafetyResult:
    def test_allowed_default_fields(self):
        r = SafetyResult(allowed=True)
        assert r.allowed
        assert r.reason == ""
        assert r.severity == "info"

    def test_blocked_fields(self):
        r = SafetyResult(allowed=False, reason="test", severity="block")
        assert not r.allowed
        assert r.reason == "test"
        assert r.severity == "block"
