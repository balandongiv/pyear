"""
Blink count validation for individual raw epochs.

Overview
--------
This module validates blink detection within EEG/EOG segments using refined annotations.
The process follows these main steps:

1. Load a raw EOG signal file (in .fif format).
2. Slice the full raw into 30-second non-overlapping mini-raw segments.
3. Run blink refinement to correct start/end frames per blink annotation.
4. Update each segment’s annotations in-place with refined blink timings.
5. Optionally, visualize raw data and updated blink annotations for inspection.
6. Count blinks from the updated annotations and validate against ground truth.

Use
---
This test ensures refined annotations produce consistent blink counts
compared to the original ground-truth blink CSV for selected segments.
"""

import logging
from pathlib import Path
import tempfile
import unittest
from typing import List, Dict

import mne
import pandas as pd
from tqdm import tqdm

from pyear.utils.epochs import slice_into_mini_raws
from pyear.utils.refinement import refine_blinks_from_epochs, plot_refined_blinks

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def update_segment_annotations(
        segments: List[mne.io.BaseRaw],
        refined: List[Dict[str, int]],
) -> None:
    """
    Update annotations on each Raw segment using refined blink start/end frames.

    Parameters
    ----------
    segments : List[mne.io.BaseRaw]
        List of MNE Raw segments with existing blink annotations.
    refined : List[Dict[str, int]]
        List of dicts, one per original annotation, each containing:
        - 'refined_start_frame': int
        - 'refined_end_frame': int

    Notes
    -----
    This modifies each Raw in-place, replacing its annotations with new
    ones whose onsets and durations are computed from the refined frames.
    """
    logger.info("Entering update_segment_annotations")
    idx = 0
    for seg_idx, seg in enumerate(tqdm(segments, desc="Segments")):
        sfreq = seg.info["sfreq"]
        logger.debug("Segment %d: sfreq=%s Hz", seg_idx, sfreq)

        orig_anns = seg.annotations
        n_anns = len(orig_anns)
        logger.debug("Segment %d: %d original annotations", seg_idx, n_anns)

        new_onsets: List[float] = []
        new_durations: List[float] = []
        new_descriptions: List[str] = []

        for ann_i in tqdm(range(n_anns), desc=f"Seg {seg_idx} annotations", leave=False):
            blink_info = refined[idx]
            start_frame = blink_info["refined_start_frame"]
            end_frame = blink_info["refined_end_frame"]

            onset = start_frame / sfreq
            duration = (end_frame - start_frame) / sfreq
            desc = orig_anns.description[ann_i]

            logger.debug(
                "Seg %d Ann %d: start_frame=%d, end_frame=%d → onset=%.3f s, duration=%.3f s, desc=%s",
                seg_idx, ann_i, start_frame, end_frame, onset, duration, desc,
            )

            new_onsets.append(onset)
            new_durations.append(duration)
            new_descriptions.append(desc)
            idx += 1

        seg.set_annotations(
            mne.Annotations(onset=new_onsets, duration=new_durations, description=new_descriptions)
        )

    logger.info("Exiting update_segment_annotations")


class TestRawBlinkCount(unittest.TestCase):
    """Compare blink counts from sliced raw epochs to expected values."""

    def setUp(self) -> None:
        """
        Set up test fixture: load raw, slice into segments, refine blinks,
        update annotations, and prepare expected counts.
        """
        logger.info("Entering TestRawBlinkCount.setUp")

        raw_path = PROJECT_ROOT / "unitest" / "ear_eog.fif"
        expected_csv = PROJECT_ROOT / "unitest" / "ear_eog_blink_count_epoch.csv"

        # load raw data without preloading
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        logger.debug("Loaded raw file: %s", raw_path)

        # slice into 30s mini-raw segments
        (
            self.segments,
            self.df,
            _,
            _,
        ) = slice_into_mini_raws(
            raw,
            Path(tempfile.mkdtemp()),
            epoch_len=30.0,
            blink_label=None,
            save=False,
            overwrite=False,
            report=False,
        )
        logger.debug("Sliced into %d segments", len(self.segments))

        # load expected blink counts
        self.expected = pd.read_csv(expected_csv)
        logger.debug("Loaded expected CSV: %s", expected_csv)

        # refine blink start/end frames
        self.channel = "EOG-EEG-eog_vert_left"
        self.refined = refine_blinks_from_epochs(self.segments, self.channel)
        logger.debug("Refined blink info for %d annotations", len(self.refined))

        # update each segment's annotations in-place
        update_segment_annotations(self.segments, self.refined)
        plot=False
        if plot:
            # Plot the first segment of the mne object for sanity check
            self.segments[0].plot(block=True)
        logger.info("Exiting TestRawBlinkCount.setUp")

    @staticmethod
    def _count_blinks(raw: mne.io.BaseRaw, label: str | None = "blink") -> int:
        """
        Count blinks in a Raw segment using blink_count_epoch.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw segment to count blinks from.
        label : str | None
            The annotation label to count as blinks (None for default).

        Returns
        -------
        int
            Number of blink annotations.
        """
        from pyear.blink_events.event_features.blink_count import blink_count_epoch

        return blink_count_epoch(raw, label=label)

    def test_total_blink_count(self) -> None:
        """
        Validate blink counts for selected raw indices against:
        - computed counts
        - DataFrame results
        - expected CSV values
        """
        logger.info("Entering TestRawBlinkCount.test_total_blink_count")
        checks = {0: 2, 13: 4, 49: 13}
        for idx, expected_count in checks.items():
            count = self._count_blinks(self.segments[idx], label=None)
            logger.debug("Segment %d: counted %d blinks", idx, count)

            # compare against DataFrame
            df_count = int(self.df.loc[idx, "blink_count"])
            csv_count = int(self.expected.loc[idx, "blink_count"])

            self.assertEqual(count, expected_count)
            self.assertEqual(count, df_count)
            self.assertEqual(count, csv_count)
        logger.info("Exiting TestRawBlinkCount.test_total_blink_count")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
