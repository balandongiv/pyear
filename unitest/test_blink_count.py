"""Unit tests for :mod:`blink_count` feature extraction."""

import unittest

from pyear.blink_events.event_features.blink_count import blink_count_epoch
from unitest.fixtures.mock_ear_generation import _generate_refined_ear


class TestBlinkCount(unittest.TestCase):
    """Tests for ``blink_count_epoch``."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.per_epoch = [list() for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_counts(self) -> None:
        """Blink counts should match expected values."""
        self.assertEqual(blink_count_epoch(self.per_epoch[0]), 3)
        self.assertEqual(blink_count_epoch(self.per_epoch[3]), 0)


if __name__ == "__main__":
    unittest.main()
