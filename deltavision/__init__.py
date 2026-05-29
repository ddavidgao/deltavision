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

import sys
from importlib import import_module
from pathlib import Path


_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


def _import_packaged_module(module_name: str):
    original_path = list(sys.path)
    try:
        if sys.path[:1] != [_PACKAGE_PARENT]:
            sys.path.insert(0, _PACKAGE_PARENT)
        return import_module(module_name)
    finally:
        sys.path[:] = original_path


# Core observer + observation + config
_observer = _import_packaged_module("observer")

DeltaVisionObserver = _observer.DeltaVisionObserver
DeltaVisionConfig = _observer.DeltaVisionConfig
DVObservation = _observer.DVObservation
ScreenshotInput = _observer.ScreenshotInput

# Vision primitives — exported so power users can build their own pipelines
_diff = _import_packaged_module("vision.diff")
_phash = _import_packaged_module("vision.phash")
_classifier = _import_packaged_module("vision.classifier")

DiffResult = _diff.DiffResult
compute_diff = _diff.compute_diff
extract_crops = _diff.extract_crops
compute_phash = _phash.compute_phash
hamming_distance = _phash.hamming_distance
ClassificationResult = _classifier.ClassificationResult
TransitionType = _classifier.TransitionType
classify_transition = _classifier.classify_transition
extract_anchor = _classifier.extract_anchor

# DOM layer (v1.0.2+: CV + DOM hybrid)
_elements = _import_packaged_module("vision.elements")

extract_page_state = _elements.extract_page_state
extract_clickables = _elements.extract_clickables
format_page_state_for_prompt = _elements.format_page_state_for_prompt
format_clickables_for_prompt = _elements.format_clickables_for_prompt

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
