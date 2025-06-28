"""Utilities for epoch handling and annotation."""

from __future__ import annotations

from typing import Iterable, List, Dict, Any
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def slice_raw_to_epochs(
    signal: np.ndarray,
    annotations: Iterable[Dict[str, int]],
    sfreq: float,
    epoch_len: float = 30.0,
) -> List[Dict[str, Any]]:
    """Slice a continuous recording into annotated epochs.

    Parameters
    ----------
    signal : numpy.ndarray
        Full eyelid aperture recording.
    annotations : Iterable[dict]
        Blink annotations with ``start``, ``trough`` and ``end`` indices.
    sfreq : float
        Sampling frequency in Hertz.
    epoch_len : float, optional
        Length of each epoch in seconds, by default ``30.0``.

    Returns
    -------
    list of dict
        Blink annotations including ``epoch_index`` and ``epoch_signal``.
    """

    samples_per_epoch = int(epoch_len * sfreq)
    n_epochs = int(np.ceil(len(signal) / samples_per_epoch))
    results: List[Dict[str, Any]] = []

    for epoch_idx in tqdm(range(n_epochs), desc="Slicing epochs"):
        start = epoch_idx * samples_per_epoch
        end = min(len(signal), start + samples_per_epoch)
        epoch_signal = signal[start:end]
        epoch_annotations = [
            ann for ann in annotations if start <= ann["trough"] < end
        ]
        for ann in epoch_annotations:
            entry = ann.copy()
            entry["epoch_index"] = epoch_idx
            entry["epoch_signal"] = epoch_signal
            entry["refined_start_frame"] = ann["start"] - start
            entry["refined_peak_frame"] = ann["trough"] - start
            entry["refined_end_frame"] = ann["end"] - start
            results.append(entry)

    logger.info("Sliced raw signal into %d epochs", n_epochs)
    return results


__all__ = ["slice_raw_to_epochs"]
