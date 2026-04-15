"""
Evaluator for McGraw-Hill benchmark runs.
"""

from agent.state import AgentState


def evaluate_run(state: AgentState) -> dict:
    """
    Compute evaluation metrics from a completed run.
    """
    total_steps = state.step
    transition_log = state.transition_log

    # Trigger breakdown
    triggers = {}
    for t in transition_log:
        tr = t["trigger"]
        triggers[tr] = triggers.get(tr, 0) + 1

    # Efficiency
    delta_steps = sum(1 for t in transition_log if t["transition"] == "delta")
    new_page_steps = sum(1 for t in transition_log if t["transition"] == "new_page")

    # No-effect actions (proxy for wasted steps)
    no_effect = sum(
        1
        for t in transition_log
        if t["transition"] == "delta" and t["diff_ratio"] < 0.005
    )

    return {
        "total_steps": total_steps,
        "delta_steps": delta_steps,
        "new_page_steps": new_page_steps,
        "delta_ratio": state.delta_ratio,
        "trigger_breakdown": triggers,
        "no_effect_actions": no_effect,
        "efficiency": (total_steps - no_effect) / max(total_steps, 1),
        "completed": state.done,
    }
