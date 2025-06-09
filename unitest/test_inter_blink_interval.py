"""Unit tests for inter-blink interval feature extraction."""

import unittest
import math

from pyear.blink_events.event_features.inter_blink_interval import compute_ibi_features
from unitest.fixtures.mock_ear_generation import _generate_refined_ear


class TestInterBlinkInterval(unittest.TestCase):
    """Tests for ``compute_ibi_features``."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.per_epoch = [list() for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_ibi_mean(self) -> None:
        """Mean IBI should equal the expected value."""
        feats = compute_ibi_features(self.per_epoch[0], self.sfreq)
        self.assertTrue(math.isclose(feats["ibi_mean"], 2.9))

    def test_no_ibis(self) -> None:
        """IBI metrics should be NaN when less than two blinks are present."""
        feats = compute_ibi_features(self.per_epoch[3], self.sfreq)
        self.assertTrue(math.isnan(feats["ibi_mean"]))


if __name__ == "__main__":
    unittest.main()
