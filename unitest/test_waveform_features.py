"""Unit tests for EAR waveform-based metrics."""
import unittest
import math
import logging

from pyear.waveform_features import (
    duration_base,
    duration_zero,
    neg_amp_vel_ratio_zero,
    aggregate_waveform_features,
)
from unitest.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestWaveformFeatures(unittest.TestCase):
    """Verify waveform-derived feature calculations."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.blinks = blinks
        self.n_epochs = n_epochs
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_individual_functions(self) -> None:
        """Check simple computations for a single blink."""
        blink = self.per_epoch[0][0]
        self.assertTrue(duration_base(blink, self.sfreq) > 0)
        self.assertTrue(duration_zero(blink, self.sfreq) > 0)
        ratio = neg_amp_vel_ratio_zero(blink, self.sfreq)
        self.assertFalse(math.isnan(ratio))

    def test_aggregate_shape(self) -> None:
        """Aggregated DataFrame should contain expected columns."""
        df = aggregate_waveform_features(self.blinks, self.sfreq, self.n_epochs)
        logger.debug("Waveform feature columns: %s", df.columns)
        self.assertIn("duration_base_mean", df.columns)
        self.assertEqual(len(df), self.n_epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
