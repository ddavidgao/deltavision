"""
Builds typed Observation objects from raw pipeline outputs.
"""

from typing import Optional, List
from PIL import Image

from agent.actions import Action
from vision.diff import DiffResult
from .types import FullFrameObservation, DeltaObservation, Observation


def build_observation(
    obs_type: str,
    task: str,
    step: int,
    last_action: Optional[Action],
    # Full frame args
    frame: Optional[Image.Image] = None,
    url: str = "",
    trigger_reason: str = "",
    # Delta args
    diff_result: Optional[DiffResult] = None,
    crops: Optional[List[dict]] = None,
    action_had_effect: bool = False,
    no_change_count: int = 0,
    text_deltas: Optional[List[dict]] = None,
    current_frame: Optional[Image.Image] = None,
) -> Observation:
    """
    Factory for building the right observation type.
    The agent loop calls this — model backends consume the result.
    """
    if obs_type == "full_frame":
        return FullFrameObservation(
            obs_type="full_frame",
            task=task,
            step=step,
            last_action=last_action,
            frame=frame,
            url=url,
            trigger_reason=trigger_reason,
        )

    return DeltaObservation(
        obs_type="delta",
        task=task,
        step=step,
        last_action=last_action,
        diff_result=diff_result,
        crops=crops or [],
        action_had_effect=action_had_effect,
        no_change_count=no_change_count,
        text_deltas=text_deltas or [],
        current_frame=current_frame,
    )
