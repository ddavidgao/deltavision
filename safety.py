"""
DeltaVision safety layer.

Model-agnostic safety checks that run BEFORE actions are executed,
regardless of which backend (Claude, OpenAI, Hermes, Qwen) generated them.

Critical for uncensored models (Hermes, etc.) that won't refuse dangerous
actions on their own. DeltaVision enforces safety at the framework level.
"""

import importlib.util
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agent.actions import Action

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_angl_safety_boundary():
    path = Path(__file__).resolve().parent / "angl_build" / "safety" / "evaluate_action_safety.py"
    if not path.exists():
        raise RuntimeError(
            "Angl safety artifact is missing. Compile "
            "specs/evaluate_action_safety.angl into angl_build/safety first."
        )
    spec = importlib.util.spec_from_file_location("_angl_evaluate_action_safety", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Angl safety artifact at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_action_safety


def _action_payload(action: Action) -> dict:
    return {
        "type": action.type.value,
        "x": action.x,
        "y": action.y,
        "text": action.text,
        "direction": action.direction,
        "amount": action.amount,
        "key": action.key,
        "duration_ms": action.duration_ms,
    }


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

    def __init__(
        self,
        allowed_domains: set[str] | None = None,
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
        policy = {
            "allowed_domains": (
                sorted(self.allowed_domains) if self.allowed_domains is not None else None
            ),
            "block_credential_entry": self.block_credential_entry,
            "block_url_shorteners": self.block_url_shorteners,
            "max_type_length": self.max_type_length,
        }
        raw = _load_angl_safety_boundary()(
            _action_payload(action),
            policy,
            current_url,
            page_context,
        )
        if not isinstance(raw, dict) or not isinstance(raw.get("allowed"), bool):
            raise RuntimeError(f"Angl safety artifact returned invalid result: {raw!r}")

        result = SafetyResult(
            allowed=raw["allowed"],
            reason=str(raw.get("reason") or ""),
            severity=str(raw.get("severity") or "info"),
        )
        if not result.allowed:
            logger.warning(
                "Action BLOCKED: %s - %s (severity=%s)",
                action, result.reason, result.severity,
            )
        return result


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
