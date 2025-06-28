"""Histogram of inter-blink intervals for an entire session."""
from __future__ import annotations

from typing import Iterable, Dict, Sequence

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def blink_interval_distribution(
    blinks: Iterable[Dict[str, int]],
    sfreq: float,
    bins: int | Sequence[float] = 10,
) -> pd.DataFrame:
    """Return a histogram of inter-blink intervals.

    Parameters
    ----------
    blinks : iterable of dict
        Blink annotations with ``refined_start_frame`` and
        ``refined_end_frame`` fields as integers and an ``epoch_index``
        identifying the epoch.
    sfreq : float
        Sampling frequency of the recording in Hertz.
    bins : int | sequence of float, optional
        Number of histogram bins or the explicit bin edges. Defaults to ``10``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``bin_start``, ``bin_end`` and ``count``.
    """
    logger.info("Computing blink interval distribution with %s bins", bins)

    starts = []
    ends = []
    for b in blinks:
        epoch_offset = b.get("epoch_index", 0) * sfreq
        starts.append(epoch_offset + b["refined_start_frame"])
        ends.append(epoch_offset + b["refined_end_frame"])
    if len(starts) < 2:
        return pd.DataFrame({"bin_start": [], "bin_end": [], "count": []})
    order = np.argsort(starts)
    starts = np.asarray(starts)[order]
    ends = np.asarray(ends)[order]
    ibis = (starts[1:] - ends[:-1]) / sfreq

    counts, edges = np.histogram(ibis, bins=bins)
    df = pd.DataFrame({
        "bin_start": edges[:-1],
        "bin_end": edges[1:],
        "count": counts,
    })
    logger.debug("IBI histogram:\n%s", df)
    return df
