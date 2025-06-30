"""Blink count validation for individual raw epochs."""
import logging
from pathlib import Path
import tempfile
import unittest

import mne
import pandas as pd
import numpy as np

from pyear.utils.epochs import slice_into_mini_raws

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestRawBlinkCount(unittest.TestCase):
    """Compare blink counts from sliced raw epochs to expected values."""

    def setUp(self) -> None:
        raw_path = PROJECT_ROOT / "unitest" / "ear_eog.fif"
        expected_csv_path = PROJECT_ROOT / "unitest" / "ear_eog_blink_count_epoch.csv"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        (
            self.epochs,
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

    @staticmethod
    def _count_blinks(raw: mne.io.BaseRaw, label: str | None = "blink") -> int:
        mask = np.ones(len(raw.annotations), dtype=bool)
        if label is not None:
            mask &= raw.annotations.description == label
        return int(mask.sum())

    def test_total_blink_count(self) -> None:
        """Validate blink counts for selected raw indices."""
        checks = {0: 2, 13: 4, 49: 13}
        for idx, expected in checks.items():
            count = self._count_blinks(self.epochs[idx], label=None)
            self.assertEqual(count, expected)
            self.assertEqual(count, int(self.df.loc[idx, "blink_count"]))
            self.assertEqual(count, int(self.expected.loc[idx, "blink_count"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
