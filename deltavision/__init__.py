"""
deltavision — umbrella import for the CV+DOM observation framework.

This module re-exports the public API so that `import deltavision` works
out of the box alongside the flat module layout (`from observer import ...`).
Both styles are equivalent; flat imports are kept for backwards compatibility.

Example:

    from deltavision import DeltaVisionObserver, DeltaVisionConfig
    obs = DeltaVisionObserver()
    result = obs.observe(screenshot_bytes)
    print(result.obs_type, result.estimated_image_tokens())
"""
from __future__ import annotations

# Core observer + observation + config
from observer import (  # noqa: F401
    DeltaVisionObserver,
    DeltaVisionConfig,
    DVObservation,
    ScreenshotInput,
)

# Vision primitives — exported so power users can build their own pipelines
from vision.diff import (  # noqa: F401
    DiffResult,
    compute_diff,
    extract_crops,
)
from vision.phash import (  # noqa: F401
    compute_phash,
    hamming_distance,
)
from vision.classifier import (  # noqa: F401
    ClassificationResult,
    TransitionType,
    classify_transition,
    extract_anchor,
)

# DOM layer (v1.0.2+: CV + DOM hybrid)
from vision.elements import (  # noqa: F401
    extract_page_state,
    extract_clickables,
    format_page_state_for_prompt,
    format_clickables_for_prompt,
)

__version__ = "1.0.7"

__all__ = [
    # core
    "DeltaVisionObserver",
    "DeltaVisionConfig",
    "DVObservation",
    "ScreenshotInput",
    # vision
    "DiffResult",
    "compute_diff",
    "extract_crops",
    "compute_phash",
    "hamming_distance",
    "ClassificationResult",
    "TransitionType",
    "classify_transition",
    "extract_anchor",
    # DOM layer
    "extract_page_state",
    "extract_clickables",
    "format_page_state_for_prompt",
    "format_clickables_for_prompt",
    # metadata
    "__version__",
]
