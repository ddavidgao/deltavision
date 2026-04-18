"""
Builds typed Observation objects from raw pipeline outputs.
"""


from PIL import Image

from agent.actions import Action
from vision.diff import DiffResult

from .types import DeltaObservation, FullFrameObservation, Observation


def build_observation(
    obs_type: str,
    task: str,
    step: int,
    last_action: Action | None,
    # Full frame args
    frame: Image.Image | None = None,
    url: str = "",
    trigger_reason: str = "",
    # Delta args
    diff_result: DiffResult | None = None,
    crops: list[dict] | None = None,
    action_had_effect: bool = False,
    no_change_count: int = 0,
    text_deltas: list[dict] | None = None,
    current_frame: Image.Image | None = None,
    # Common: DOM-extracted clickable elements (fixes small-UI targeting)
    clickable_elements: list[dict] | None = None,
    # Common: DOM-extracted focus state (fixes "click on input didn't register" failures)
    focus: dict | None = None,
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
            clickable_elements=clickable_elements or [],
            focus=focus,
            frame=frame,
            url=url,
            trigger_reason=trigger_reason,
        )

    return DeltaObservation(
        obs_type="delta",
        task=task,
        step=step,
        last_action=last_action,
        clickable_elements=clickable_elements or [],
        focus=focus,
        diff_result=diff_result,
        crops=crops or [],
        action_had_effect=action_had_effect,
        no_change_count=no_change_count,
        text_deltas=text_deltas or [],
        current_frame=current_frame,
    )
