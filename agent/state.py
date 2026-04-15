from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class AgentState:
    task: str
    step: int = 0
    done: bool = False
    no_change_streak: int = 0
    new_page_count: int = 0
    observations: List[Any] = field(default_factory=list)
    responses: List[Any] = field(default_factory=list)
    transition_log: List[dict] = field(default_factory=list)

    def add_observation(self, obs):
        self.observations.append(obs)

    def add_response(self, r):
        self.responses.append(r)

    def increment_no_change_streak(self):
        self.no_change_streak += 1

    def reset_no_change_streak(self):
        self.no_change_streak = 0

    def increment_new_page_count(self):
        self.new_page_count += 1

    def log_transition(self, classification, action, step):
        self.transition_log.append(
            {
                "step": step,
                "action": str(action),
                "transition": classification.transition.value,
                "trigger": classification.trigger,
                "diff_ratio": classification.diff_ratio,
                "phash_distance": classification.phash_distance,
                "anchor_score": classification.anchor_score,
            }
        )

    @property
    def delta_ratio(self) -> float:
        """Fraction of steps that used DELTA. Higher = more efficient."""
        if not self.transition_log:
            return 0.0
        deltas = sum(1 for t in self.transition_log if t["transition"] == "delta")
        return deltas / len(self.transition_log)
