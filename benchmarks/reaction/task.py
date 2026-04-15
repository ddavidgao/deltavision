"""
Reaction time benchmark for DeltaVision.

Tests how quickly the agent can detect and respond to a visual state change
(e.g. red light → green light). This measures the full pipeline latency:
  screenshot capture + diff computation + classification + model inference + action execution

Run against: https://humanbenchmark.com/tests/reactiontime
Alternative: https://www.arealme.com/reaction-test/en/

Metrics:
  - reaction_time_ms: Time from visual change to action (click)
  - detection_latency_ms: Time for diff pipeline to detect the change
  - model_latency_ms: Time for model to decide to click
  - false_start_rate: Clicks before the change occurred
  - delta_efficiency: What % of steps used DELTA vs full frame

This benchmark is fundamentally about DeltaVision's speed advantage:
a standard agent re-processes the full screen on every observation,
while DeltaVision can detect "green appeared" from a tiny diff region
and skip the full-frame model call entirely.
"""

REACTION_TASKS = [
    {
        "id": "human_benchmark_reaction",
        "description": (
            "Complete the reaction time test. Wait for the screen to turn green, "
            "then click as fast as possible. Repeat 5 times."
        ),
        "start_url": "https://humanbenchmark.com/tests/reactiontime",
        "success_criteria": "Results page visible with average reaction time",
        "notes": [
            "Screen goes from blue (instructions) → red (wait) → green (click!)",
            "The red→green transition is a MASSIVE diff (entire viewport changes color)",
            "DeltaVision should detect this at Level 0 (diff_ratio near 1.0)",
            "The key metric is time from green appearing to click being sent",
            "No model inference needed — the diff itself IS the signal",
        ],
        "deltavision_optimizations": [
            "Skip model call entirely on color-change transitions",
            "Pre-compute the expected 'green screen' state for instant matching",
            "Use the diff_ratio spike as a direct trigger: if diff_ratio > 0.9 AND "
            "the dominant color of t1 is green → click immediately",
            "This bypasses the full observation→model→action pipeline",
        ],
    },
    {
        "id": "aim_trainer",
        "description": (
            "Hit the targets as quickly as possible. Targets appear at random "
            "positions on screen."
        ),
        "start_url": "https://humanbenchmark.com/tests/aim",
        "success_criteria": "Results page with average time per target",
        "notes": [
            "Small circular target appears at random position",
            "This is a DELTA transition — only the target region changes",
            "DeltaVision should detect the target via bbox extraction",
            "The bbox center gives the click coordinates directly",
            "No model needed: diff → find new bbox → click center",
        ],
    },
]

# These benchmarks test the thesis that DeltaVision enables
# reactive agents that can bypass the model for simple state changes.
# The model is for understanding; the CV pipeline is for reacting.
