"""Blink count validation for individual raw epochs."""
import logging
from pathlib import Path
import tempfile
import unittest

import mne
import pandas as pd
import numpy as np

from pyear.utils.epochs import slice_into_mini_raws
from pyear.utils.refinement import refine_blinks_from_epochs, plot_refined_blinks

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestRawBlinkCount(unittest.TestCase):
    """Compare blink counts from sliced raw epochs to expected values."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unitest" / "ear_eog.fif"
        expected_csv_path = PROJECT_ROOT / "unitest" / "ear_eog_blink_count_epoch.csv"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
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
        self.expected = pd.read_csv(expected_csv_path)

        # refine blink start/end frames and update segment annotations
        self.channel = "EOG-EEG-eog_vert_left"
        self.refined = refine_blinks_from_epochs(self.segments, self.channel)

        idx = 0
        for seg in self.segments:
            sfreq = seg.info["sfreq"]
            new_onsets = []
            new_durs = []
            new_desc = []
            for ann_i in range(len(seg.annotations)):
                blink = self.refined[idx]
                new_onsets.append(blink["refined_start_frame"] / sfreq)
                new_durs.append(
                    (blink["refined_end_frame"] - blink["refined_start_frame"]) / sfreq
                )
                new_desc.append(seg.annotations.description[ann_i])
                idx += 1
            seg.set_annotations(mne.Annotations(new_onsets, new_durs, new_desc))

        # create sanity check plots for selected epochs without displaying them
        self.plots = plot_refined_blinks(
            self.refined,
            self.segments[0].info["sfreq"],
            30.0,
            epoch_indices=[0, 13, 49],
            show=False,
        )

    @staticmethod
    def _count_blinks(raw: mne.io.BaseRaw, label: str | None = "blink") -> int:
        from pyear.blink_events.event_features.blink_count import blink_count_epoch
        return blink_count_epoch(raw, label=label)

    def test_total_blink_count(self) -> None:
        """Validate blink counts for selected raw indices."""
        checks = {0: 2, 13: 4, 49: 13}
        for idx, expected in checks.items():
            count = self._count_blinks(self.segments[idx], label=None)
            self.assertEqual(count, expected)
            self.assertEqual(count, int(self.df.loc[idx, "blink_count"]))
            self.assertEqual(count, int(self.expected.loc[idx, "blink_count"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
