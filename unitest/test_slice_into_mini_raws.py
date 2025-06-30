"""Integration tests for ``slice_into_mini_raws``.

This test creates a synthetic raw recording with blink annotations, slices it
into epochs, saves them to disk and verifies that the saved files match the
in-memory epochs. Blink counts per epoch are also validated.
"""
import logging
import tempfile
from pathlib import Path
import unittest

import mne
import numpy as np

from pyear.utils.epochs import slice_into_mini_raws

logger = logging.getLogger(__name__)


class TestSliceIntoMiniRaws(unittest.TestCase):
    """Validate slicing and saving of raw epochs."""

    def setUp(self) -> None:
        self.sfreq = 50.0
        self.epoch_len = 10.0
        self.n_epochs = 3
        n_samples = int(self.sfreq * self.epoch_len * self.n_epochs)
        rng = np.random.default_rng(0)
        data = rng.normal(scale=1e-6, size=(1, n_samples))
        info = mne.create_info(["EOG"], self.sfreq, ["misc"])
        raw = mne.io.RawArray(data, info, verbose=False)
        onsets = np.array([2.0, 5.0, 12.0, 18.0, 22.0])
        durations = np.repeat(0.1, len(onsets))
        raw.set_annotations(mne.Annotations(onsets, durations, ["blink"] * len(onsets)))
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp_dir.name)
        (
            self.segments,
            self.df,
            _,
            _,
        ) = slice_into_mini_raws(
            raw,
            self.out_dir,
            epoch_len=self.epoch_len,
            blink_label="blink",
            save=True,
            overwrite=True,
            report=False,
        )
        self.saved_segments = [
            mne.io.read_raw_fif(p, preload=True, verbose=False)
            for p in sorted(self.out_dir.glob("epoch_*_raw.fif"))
        ]
        self.expected_counts = [2, 2, 1]

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @staticmethod
    def _count_blinks(raw: mne.io.BaseRaw, label: str = "blink") -> int:
        mask = np.ones(len(raw.annotations), dtype=bool)
        if label is not None:
            mask &= raw.annotations.description == label
        return int(mask.sum())

    def test_saved_equals_memory(self) -> None:
        """Saved segments should be identical to in-memory segments."""
        self.assertEqual(len(self.segments), len(self.saved_segments))
        for mem, disk in zip(self.segments, self.saved_segments):
            np.testing.assert_allclose(mem.get_data(), disk.get_data())
            np.testing.assert_array_equal(mem.annotations.onset, disk.annotations.onset)
            self.assertListEqual(
                list(mem.annotations.description),
                list(disk.annotations.description),
            )

    def test_blink_counts(self) -> None:
        """Blink counts match expectation for each segment."""
        for idx, raw in enumerate(self.segments):
            count = self._count_blinks(raw)
            self.assertEqual(count, self.expected_counts[idx])
            self.assertEqual(count, int(self.df.loc[idx, "blink_count"]))
        for idx, raw in enumerate(self.saved_segments):
            count = self._count_blinks(raw)
            self.assertEqual(count, self.expected_counts[idx])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
