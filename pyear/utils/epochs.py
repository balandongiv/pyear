"""Epoching utilities for time-series data."""
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BLINK_LABEL = "blink"        # Annotation label for blink events
EPOCH_LEN = 30.0              # Epoch duration in seconds

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core utility functions
# -----------------------------------------------------------------------------

def slice_into_mini_raws(
        raw: mne.io.BaseRaw,
        out_dir: Path,
        epoch_len: float = EPOCH_LEN,
        save: bool = True,
        report: Optional[mne.Report] = None,
        blink_label: Optional[str] = BLINK_LABEL,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """Slice a raw recording into epochs and count blinks.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording with blink annotations.
    out_dir : Path
        Directory to save epoch files, if enabled.
    epoch_len : float
        Length of each epoch in seconds.
    save : bool
        Whether to write each epoch raw file to disk.
    report : mne.Report | None
        Report object to which figures will be added.
    blink_label : str | None
        Description label to filter blink annotations.

    Returns
    -------
    df : pandas.DataFrame
        Epoch blink counts (columns: "epoch_id", "blink_count").
    boundary_pairs : list of tuple
        Pairs of epoch indices where a blink spans a boundary.
    """
    logger.info("Entering slice_into_mini_raws")
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = raw.annotations
    mask = np.ones(len(ann), dtype=bool)
    if blink_label is not None:
        mask &= ann.description == blink_label
    onsets = ann.onset[mask]
    durations = ann.duration[mask]

    total_time = raw.times[-1]
    n_epochs = int(np.ceil(total_time / epoch_len))
    counts = [0] * n_epochs
    boundary_pairs: List[Tuple[int, int]] = []

    for i in tqdm(range(n_epochs), desc="Cropping epochs", unit="epoch"):
        start = i * epoch_len
        stop = min(start + epoch_len, total_time)

        in_epoch = (onsets >= start) & (onsets < stop)
        counts[i] = int(np.sum(in_epoch))
        spans = in_epoch & ((onsets + durations) > stop)
        for _ in np.where(spans)[0]:
            if i + 1 < n_epochs:
                boundary_pairs.append((i, i + 1))

        mini = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)
        ann_epoch = mini.annotations
        shifted = mne.Annotations(
            onset=ann_epoch.onset - start,
            duration=ann_epoch.duration,
            description=ann_epoch.description,
        )
        mini.set_annotations(shifted)
        # mini.plot()
        fig = mini.plot(
            n_channels=min(10, len(mini.ch_names)),
            scalings="auto",
            title=f"Epoch {i} ({start:.2f}-{stop:.2f}s)",
            show=False,
        )
        if report is not None:
            report.add_figure(fig, title=f"Epoch {i}", section="epochs")
        if save:
            fname = out_dir / f"epoch_{i:04d}_{start:07.2f}s-{stop:07.2f}s_raw.fif"
            mini.save(fname, overwrite=True)
        plt.close(fig)

    df = pd.DataFrame({"epoch_id": range(n_epochs), "blink_count": counts})
    logger.debug("Blink counts per epoch: %s", counts)
    logger.debug("Cross-boundary pairs: %s", boundary_pairs)
    logger.info("Exiting slice_into_mini_raws")
    return df, boundary_pairs