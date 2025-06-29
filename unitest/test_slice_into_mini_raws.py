"""Blink count validation for raw segmentation.

This test ensures that ``slice_into_mini_raws`` correctly counts blinks in
30-second segments of the ``ear_eog.fif`` recording.  The expected blink counts
for the first ten segments are stored in
``ear_eog_blink_count_epoch.csv``.
"""

import logging
import tempfile
from pathlib import Path
import unittest

import mne
import pandas as pd

from pyear.utils.epochs import slice_into_mini_raws

logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSliceIntoMiniRaws(unittest.TestCase):
    """Verify blink counts from 30-second raw segments."""

    def setUp(self) -> None:
        """Run ``slice_into_mini_raws`` once for use in all tests."""
        raw_path = PROJECT_ROOT / "unitest" / "ear_eog.fif"
        expected_csv_path = PROJECT_ROOT / "unitest" / "ear_eog_blink_count_epoch.csv"

        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.df, _ = slice_into_mini_raws(
                raw=raw,
                out_dir=Path(tmpdir),
                epoch_len=30.0,
                save=False,
                report=None,
                blink_label=None,
            )
        self.expected = pd.read_csv(expected_csv_path)

    def test_each_segment(self) -> None:
        """Check blink count for each of the first ten segments individually."""
        for idx, expected_count in enumerate(self.expected["blink_count"]):
            with self.subTest(segment=idx):
                result = int(self.df.loc[idx, "blink_count"])
                self.assertEqual(result, expected_count)

    def test_total_blink_count(self) -> None:
        """Total blink count across the first ten segments should match."""
        result_total = int(self.df.loc[: len(self.expected) - 1, "blink_count"].sum())
        expected_total = int(self.expected["blink_count"].sum())
        self.assertEqual(result_total, expected_total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
