"""
DeltaVision safety layer.

Model-agnostic safety checks that run BEFORE actions are executed,
regardless of which backend (Claude, OpenAI, Hermes, Qwen) generated them.

Critical for uncensored models (Hermes, etc.) that won't refuse dangerous
actions on their own. DeltaVision enforces safety at the framework level.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Set
from urllib.parse import urlparse

from agent.actions import Action, ActionType

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    allowed: bool
    reason: str = ""
    severity: str = "info"  # "info", "warn", "block"


class SafetyLayer:
    """
    Validates actions before execution.
    Plugged into the agent loop between model response and execute_action.
    """

    # Domains/URL patterns known to be credential/phishing targets.
    # URL shorteners are checked separately in _check_url_safety so that
    # block_url_shorteners=False can actually disable the check — including
    # them here would defeat that flag.
    SUSPICIOUS_PATTERNS = [
        r".*login.*\.(?!edu|gov)",  # login pages outside .edu/.gov
        r".*signin.*",
        r".*account.*verify.*",
        r".*password.*reset.*",
        r".*\.ru/.*",
        r".*\.cn/.*login.*",
    ]

    # Input fields that should never be auto-filled
    SENSITIVE_FIELD_PATTERNS = [
        r"password",
        r"ssn",
        r"social.security",
        r"credit.card",
        r"card.number",
        r"cvv",
        r"cvc",
        r"bank.account",
        r"routing.number",
    ]

    def __init__(
        self,
        allowed_domains: Optional[Set[str]] = None,
        block_credential_entry: bool = True,
        block_url_shorteners: bool = True,
        max_type_length: int = 500,
    ):
        self.allowed_domains = allowed_domains
        self.block_credential_entry = block_credential_entry
        self.block_url_shorteners = block_url_shorteners
        self.max_type_length = max_type_length

    def check_action(
        self, action: Action, current_url: str, page_context: str = ""
    ) -> SafetyResult:
        """
        Run all safety checks on a proposed action.
        Returns SafetyResult with allowed=False if action should be blocked.
        """
        checks = [
            self._check_type_safety(action, page_context),
            self._check_url_safety(action, current_url),
            self._check_action_limits(action),
        ]

        for result in checks:
            if not result.allowed:
                logger.warning(
                    "Action BLOCKED: %s — %s (severity=%s)",
                    action, result.reason, result.severity,
                )
                return result

        return SafetyResult(allowed=True)

    def _check_type_safety(self, action: Action, page_context: str) -> SafetyResult:
        """Block typing sensitive data into credential fields."""
        if action.type != ActionType.TYPE:
            return SafetyResult(allowed=True)

        if not self.block_credential_entry:
            return SafetyResult(allowed=True)

        # Check if the page context suggests a credential field
        context_lower = page_context.lower()
        for pattern in self.SENSITIVE_FIELD_PATTERNS:
            if re.search(pattern, context_lower):
                return SafetyResult(
                    allowed=False,
                    reason=f"Blocked typing into potential credential field (matched: {pattern})",
                    severity="block",
                )

        # Check if the text being typed looks like a credential
        if action.text and self._looks_like_credential(action.text):
            return SafetyResult(
                allowed=False,
                reason="Blocked: text looks like a credential (SSN, card number, etc.)",
                severity="block",
            )

        return SafetyResult(allowed=True)

    def _check_url_safety(self, action: Action, current_url: str) -> SafetyResult:
        """Check current page URL against suspicious patterns."""
        if not current_url:
            return SafetyResult(allowed=True)

        parsed = urlparse(current_url)
        domain = parsed.netloc.lower()

        # Block URL shorteners
        if self.block_url_shorteners:
            for shortener in ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"]:
                if shortener in domain:
                    return SafetyResult(
                        allowed=False,
                        reason=f"Blocked: URL shortener detected ({shortener}). Cannot verify destination.",
                        severity="block",
                    )

        # Check domain allowlist if configured
        if self.allowed_domains is not None:
            base_domain = ".".join(domain.split(".")[-2:])
            if base_domain not in self.allowed_domains and domain not in self.allowed_domains:
                return SafetyResult(
                    allowed=False,
                    reason=f"Blocked: domain {domain} not in allowlist",
                    severity="warn",
                )

        # Check suspicious URL patterns
        url_lower = current_url.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url_lower):
                return SafetyResult(
                    allowed=False,
                    reason=f"Blocked: suspicious URL pattern ({pattern})",
                    severity="warn",
                )

        return SafetyResult(allowed=True)

    def _check_action_limits(self, action: Action) -> SafetyResult:
        """Sanity checks on action parameters."""
        if action.type == ActionType.TYPE and action.text:
            if len(action.text) > self.max_type_length:
                return SafetyResult(
                    allowed=False,
                    reason=f"Blocked: type action too long ({len(action.text)} chars > {self.max_type_length})",
                    severity="warn",
                )

        if action.type == ActionType.CLICK:
            if action.x is not None and action.y is not None:
                if action.x < 0 or action.y < 0:
                    return SafetyResult(
                        allowed=False,
                        reason="Blocked: negative click coordinates",
                        severity="warn",
                    )

        return SafetyResult(allowed=True)

    @staticmethod
    def _looks_like_credential(text: str) -> bool:
        """Heuristic check if text looks like a sensitive credential."""
        stripped = text.replace("-", "").replace(" ", "")

        # SSN pattern: 9 digits
        if re.match(r"^\d{9}$", stripped):
            return True

        # Credit card: 13-19 digits
        if re.match(r"^\d{13,19}$", stripped):
            return True

        # CVV: 3-4 digits alone
        if re.match(r"^\d{3,4}$", stripped) and len(text) <= 4:
            return True

        return False


# Pre-built safety configs

PERMISSIVE = SafetyLayer(
    block_credential_entry=True,
    block_url_shorteners=True,
    max_type_length=1000,
)

STRICT = SafetyLayer(
    block_credential_entry=True,
    block_url_shorteners=True,
    max_type_length=200,
)

EDUCATIONAL = SafetyLayer(
    allowed_domains={
        "mheducation.com",
        "learning.mheducation.com",
        "connect.mheducation.com",
        "purdue.brightspace.com",
        "humanbenchmark.com",
    },
    block_credential_entry=True,
    block_url_shorteners=True,
)
