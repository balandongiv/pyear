"""Stub for blink refinement.

This module provides ``refine_blink_with_ear_extrema_and_threshold_stub`` which
mimics the behaviour of a more sophisticated blink refinement algorithm. It is
intended solely for unit testing within this repository.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def refine_ear_extrema_and_threshold_stub(
    signal_segment: np.ndarray,
    start_rel: int,
    end_rel: int,
    peak_rel_cvat: int | None = None,
    *,
    local_max_prominence: float = 0.01,
    search_expansion_frames: int = 5,
    value_threshold: float | None = None,
) -> Tuple[int, int, int]:
    """Return a crude ear refinement.
    currently the logic below is for trough refinement only.The trough is lowest point in the eye aspect ratio segment between the
    start and end relative indices. If ``peak_rel_cvat`` is provided, it is used

    Parameters mirror those of the real refinement routine but the
    implementation merely validates that indices are within bounds and
    estimates a trough location if ``peak_rel_cvat`` is not supplied.
    """
    valid_trough = peak_rel_cvat
    if not (peak_rel_cvat is not None and 0 <= peak_rel_cvat < len(signal_segment)):
        if end_rel >= start_rel and len(signal_segment) > 0:
            valid_trough = (start_rel + end_rel) // 2
        else:
            valid_trough = 0

    rs_stub = max(0, min(start_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    re_stub = max(0, min(end_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    if rs_stub > re_stub:
        rs_stub = re_stub
    return rs_stub, valid_trough, re_stub
