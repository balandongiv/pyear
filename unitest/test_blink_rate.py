"""Unit tests for :mod:`blink_rate` feature extraction."""

import unittest

from pyear.blink_events.event_features.blink_rate import blink_rate_epoch
from unitest.fixtures.mock_ear_generation import _generate_refined_ear


class TestBlinkRate(unittest.TestCase):
    """Tests for ``blink_rate_epoch``."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.epoch_len = epoch_len
        self.per_epoch = [list() for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_rates(self) -> None:
        """Blink rate should be calculated in blinks per minute."""
        self.assertEqual(blink_rate_epoch(self.per_epoch[0], self.epoch_len), 18)
        self.assertEqual(blink_rate_epoch(self.per_epoch[3], self.epoch_len), 0)


if __name__ == "__main__":
    unittest.main()
