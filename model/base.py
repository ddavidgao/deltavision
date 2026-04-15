"""
Abstract model interface. Swap backends without touching the agent loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from agent.actions import Action


@dataclass
class ModelResponse:
    action: Optional[Action]
    done: bool
    reasoning: str
    confidence: float  # 0-1 self-reported
    raw_response: dict


class BaseModel(ABC):
    @abstractmethod
    async def predict(self, observation, state) -> ModelResponse:
        ...
