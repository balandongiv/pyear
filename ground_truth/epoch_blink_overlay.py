#!/usr/bin/env python3
"""Utilities for inspecting blink annotations over fixed-length epochs.

This module provides helpers to slice a continuous :class:`mne.io.Raw` object
into 30-second segments and optionally save them to disk.  It also generates a
summary CSV with the number of blinks per epoch, detects blinks crossing epoch
boundaries and produces an HTML report containing images of each epoch with the
annotation overlays.

Example
-------
Run the module directly to process ``ear_eog.fif`` used in unit tests::

    python epoch_blink_overlay.py --file ../unitest/ear_eog.fif --save-segments
"""
import logging
from pathlib import Path

from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BLINK_LABEL = "blink"        # Description of blink annotations in the Raw
EPOCH_LEN = 30.0              # Epoch duration in seconds

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def slice_into_mini_raws(
    raw: mne.io.BaseRaw,
    out_dir: Path,
    epoch_len: float = EPOCH_LEN,
    save: bool = True,
    report: mne.Report | None = None,
    blink_label: str | None = BLINK_LABEL,
) -> tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """Slice ``raw`` into individual epochs and count blinks.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording containing blink annotations.
    out_dir : Path
        Directory where epoch files will be saved if ``save`` is ``True``.
    epoch_len : float, optional
        Epoch duration in seconds.
    save : bool, optional
        Whether to write each mini raw to ``out_dir``.
    report : mne.Report | None, optional
        If provided, figures of each epoch are added to this report.
    
    Returns
    -------
    tuple[pandas.DataFrame, list[tuple[int, int]]]
        DataFrame with blink counts per epoch and list of cross-boundary blink
        epoch pairs.
    """
    logger.info("Entering slice_into_mini_raws")
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = raw.annotations
    mask = np.ones(len(ann), dtype=bool)
    if blink_label is not None:
        mask &= ann.description == blink_label
    blink_onsets = ann.onset[mask]
    blink_durations = ann.duration[mask]

    total_time = raw.times[-1]
    n_epochs = int(np.ceil(total_time / epoch_len))
    counts = [0 for _ in range(n_epochs)]
    boundary_pairs: List[Tuple[int, int]] = []

    for i in tqdm(range(n_epochs), desc="Cropping epochs", unit="epoch"):
        start = i * epoch_len
        stop = min(start + epoch_len, total_time)

        mask_epoch = (blink_onsets >= start) & (blink_onsets < stop)
        counts[i] = int(mask_epoch.sum())
        cross = mask_epoch & ((blink_onsets + blink_durations) > stop)
        for _ in np.where(cross)[0]:
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

        fig = mini.plot(
            n_channels=min(10, len(mini.ch_names)),
            scalings="auto",
            title=f"Epoch {i} ({start:.2f}-{stop:.2f}s)",
            show=False,
        )
        if report is not None:
            report.add_figs_to_section(fig, title=f"Epoch {i}", section="epochs")
        if save:
            fname = out_dir / f"epoch_{i:04d}_{start:07.2f}s-{stop:07.2f}s_raw.fif"
            mini.save(fname, overwrite=True)
        plt.close(fig)
    df = pd.DataFrame({"epoch_id": range(n_epochs), "blink_count": counts})
    logger.debug("Blink counts per epoch: %s", counts)
    logger.debug("Cross-boundary pairs: %s", boundary_pairs)
    logger.info("Exiting slice_into_mini_raws")
    return df, boundary_pairs


def summarize_blink_counts(
    raw: mne.io.BaseRaw,
    epoch_len: float = EPOCH_LEN,
    blink_label: str | None = BLINK_LABEL,
) -> tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """Compute blink count per epoch and detect cross-boundary blinks.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with blink annotations.
    epoch_len : float, optional
        Epoch length in seconds.
    blink_label : str | None, optional
        Annotation label used to mark blinks. If ``None`` all annotations are
        treated as blinks.

    Returns
    -------
    tuple[pandas.DataFrame, list[tuple[int, int]]]
        DataFrame with columns ``"epoch_id"`` and ``"blink_count"`` and a list
        of epoch index pairs where a blink crosses the boundary.
    """
    logger.info("Summarizing blink counts per epoch")
    onsets = raw.annotations.onset
    durations = raw.annotations.duration
    descriptions = raw.annotations.description

    total_time = raw.times[-1]
    n_epochs = int(np.ceil(total_time / epoch_len))
    counts = [0 for _ in range(n_epochs)]
    boundary_pairs: List[Tuple[int, int]] = []

    for onset, duration, desc in zip(onsets, durations, descriptions):
        if blink_label is not None and desc != blink_label:
            continue

        start_epoch = int(onset // epoch_len)
        end_epoch = int((onset + duration) // epoch_len)
        if start_epoch < n_epochs:
            counts[start_epoch] += 1
        if end_epoch != start_epoch and end_epoch < n_epochs:
            boundary_pairs.append((start_epoch, end_epoch))

    df = pd.DataFrame({"epoch_id": range(n_epochs), "blink_count": counts})
    logger.debug("Blink counts: %s", counts)
    logger.debug("Cross-boundary pairs: %s", boundary_pairs)
    return df, boundary_pairs



# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Execute blink overlay workflow using command-line arguments."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--file",
        type=Path,
        default=Path("unitest/ear_eog.fif"),
        help="Path to FIF file",
    )


    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_epochs"),
        help="Directory for saved epochs",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("unitest/ear_eog_blink_count_epoch.csv"),
        help="CSV file for blink counts",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("epoch_report.html"),
        help="Output HTML report",
    )
 
 
    parser.add_argument(
        "--save-segments",
        action="store_true",
        help="Persist segmented raw files",
    )
    parser.add_argument(
        "--blink-label",
        default=None,
        help="Annotation label identifying blinks (default: all annotations)",
    )
    args = parser.parse_args()

    logger.info("Reading raw FIF from %s", args.file)
    raw = mne.io.read_raw_fif(str(args.file), preload=True)

    report = mne.Report(title="Epoch Blink Overlay")
    df, boundaries = slice_into_mini_raws(
        raw,
        args.out_dir,
        epoch_len=EPOCH_LEN,
        save=args.save_segments,
        report=report,
        blink_label=args.blink_label,
    )

    df.to_csv(args.summary, index=False)
    if boundaries:
        logger.info("Cross-boundary blinks detected between epochs: %s", boundaries)

    summary_html = "<h2>Blink Counts</h2>" + df.to_html(index=False)
    if boundaries:
        btxt = "<br/>".join(f"{s}→{e}" for s, e in boundaries)
        summary_html += f"<h2>Cross-boundary Blinks</h2><p>{btxt}</p>"
    report.add_html(summary_html, title="Statistics", section="summary")

    report.save(args.report, overwrite=True, open_browser=False)




if __name__ == "__main__":
    main()
