#!/usr/bin/env python3
"""Utilities for inspecting blink annotations over fixed-length epochs.

This module provides functions to slice a continuous :class:`mne.io.Raw` object
into 30-second segments, generate blink-count summaries, and produce an HTML
report with annotation overlays for each epoch.
"""
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


def summarize_blink_counts(
        raw: mne.io.BaseRaw,
        epoch_len: float = EPOCH_LEN,
        blink_label: Optional[str] = BLINK_LABEL,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """Compute blink counts per epoch and detect boundary-spanning blinks."""
    logger.info("Summarizing blink counts per epoch")
    onsets = raw.annotations.onset
    durations = raw.annotations.duration
    descriptions = raw.annotations.description

    total_time = raw.times[-1]
    n_epochs = int(np.ceil(total_time / epoch_len))
    counts = [0] * n_epochs
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
# Workflow orchestration
# -----------------------------------------------------------------------------

def run_epoch_blink_overlay(
        fif_file: Path,
        out_dir: Path,
        summary_csv: Path,
        report_html: Path,
        save_segments: bool = False,
        blink_label: Optional[str] = BLINK_LABEL,
) -> None:
    """Run the blink overlay pipeline end-to-end with provided paths."""
    # Resolve paths
    fif_file = fif_file.resolve()
    out_dir = out_dir.resolve()
    summary_csv = summary_csv.resolve()
    report_html = report_html.resolve()

    logger.info("Reading raw FIF from %s", fif_file)
    raw = mne.io.read_raw_fif(str(fif_file), preload=True)

    report = mne.Report(title="Epoch Blink Overlay")
    df, boundaries = slice_into_mini_raws(
        raw=raw,
        out_dir=out_dir,
        epoch_len=EPOCH_LEN,
        save=save_segments,
        report=report,
        blink_label=blink_label,
    )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)

    if boundaries:
        logger.info("Cross-boundary blinks detected between epochs: %s", boundaries)

    summary_html = "<h2>Blink Counts</h2>" + df.to_html(index=False)
    if boundaries:
        btxt = "<br/>".join(f"{s}\u2192{e}" for s, e in boundaries)
        summary_html += f"<h2>Cross-boundary Blinks</h2><p>{btxt}</p>"
    report.add_html(summary_html, title="Statistics", section="summary")
    report.save(report_html, overwrite=True, open_browser=False)


def create_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", type=Path,
        default=Path("../unitest/ear_eog.fif"),
        help="Path to input FIF file",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("debug_epochs"),
        help="Directory to save epoch FIF files",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("../unitest/output_ear_eog_blink_count_epoch.csv"),
        help="CSV file path for blink counts",
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("epoch_report.html"),
        help="HTML report output path",
    )
    parser.add_argument(
        "--save-segments", action="store_true",
        help="Save segmented FIF files",
    )
    parser.add_argument(
        "--blink-label", default=None,
        help="Annotation label identifying blinks",
    )
    return parser


def main() -> None:
    """Entry point for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = create_argument_parser()
    args = parser.parse_args()

    run_epoch_blink_overlay(
        fif_file=args.file,
        out_dir=args.out_dir,
        summary_csv=args.summary,
        report_html=args.report,
        save_segments=args.save_segments,
        blink_label=args.blink_label,
    )


if __name__ == "__main__":
    main()
