"""
Scripted model backend — follows a pre-defined action sequence.
No API key needed. Tests the full pipeline without LLM costs.

Usage:
    model = ScriptedModel([
        Action(type=ActionType.CLICK, x=640, y=300),
        Action(type=ActionType.WAIT, duration_ms=1000),
        Action(type=ActionType.TYPE, text="hello"),
        Action(type=ActionType.KEY, key="Enter"),
    ])
"""

from typing import List, Optional

from .base import BaseModel, ModelResponse
from agent.actions import Action, ActionType


class ScriptedModel(BaseModel):
    """
    Executes a fixed sequence of actions. For testing the full pipeline
    without burning API credits. The CV pipeline still runs — this only
    replaces the model reasoning step.
    """

    def __init__(self, actions: List[Action], log_observations: bool = True):
        self.actions = actions
        self.current = 0
        self.log_observations = log_observations
        self.observation_log = []

    async def predict(self, observation, state) -> ModelResponse:
        # Log what the pipeline sent us (for analysis)
        if self.log_observations:
            self.observation_log.append({
                "step": observation.step,
                "type": observation.obs_type,
                "had_effect": getattr(observation, "action_had_effect", None),
                "diff_ratio": getattr(
                    getattr(observation, "diff_result", None), "diff_ratio", None
                ),
                "num_crops": len(getattr(observation, "crops", [])),
                "text_deltas": getattr(observation, "text_deltas", []),
            })

        if self.current >= len(self.actions):
            return ModelResponse(
                action=None,
                done=True,
                reasoning="Script complete",
                confidence=1.0,
                raw_response={"scripted": True},
            )

        action = self.actions[self.current]
        self.current += 1

        return ModelResponse(
            action=action,
            done=False,
            reasoning=f"Scripted action {self.current}/{len(self.actions)}: {action}",
            confidence=1.0,
            raw_response={"scripted": True, "step": self.current},
        )
