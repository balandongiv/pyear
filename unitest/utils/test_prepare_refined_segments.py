"""Tests for prepare_refined_segments utility."""
import logging
from pathlib import Path
import unittest

import pandas as pd

from pyear.utils import prepare_refined_segments

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestPrepareRefinedSegments(unittest.TestCase):
    """Validate the end-to-end preprocessing helper."""

    def setUp(self) -> None:
        self.raw_path = PROJECT_ROOT / "unitest" / "ear_eog.fif"
        self.expected_csv = PROJECT_ROOT / "unitest" / "ear_eog_blink_count_epoch.csv"

    def test_segments_count_no_epoch_signal(self) -> None:
        """Default call returns expected number of segments without epoch_signal."""
        segments, refined = prepare_refined_segments(
            self.raw_path,
            "EOG-EEG-eog_vert_left",
            keep_epoch_signal=False,
        )
        expected_len = len(pd.read_csv(self.expected_csv))
        self.assertEqual(len(segments), expected_len)
        self.assertNotIn("epoch_signal", refined[0])

    def test_segments_count_keep_epoch_signal(self) -> None:
        """Segments retain epoch_signal field when requested."""
        segments, refined = prepare_refined_segments(
            self.raw_path,
            "EOG-EEG-eog_vert_left",
            keep_epoch_signal=True,
        )
        expected_len = len(pd.read_csv(self.expected_csv))
        self.assertEqual(len(segments), expected_len)
        self.assertIn("epoch_signal", refined[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
