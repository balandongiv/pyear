"""Unit tests for frequency-domain feature extraction."""
import unittest
import logging

from pyear.frequency_domain.frequency_features import compute_frequency_domain_features
from pyear.frequency_domain import aggregate_frequency_domain_features
from unitest.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestFrequencyDomainFeatures(unittest.TestCase):
    """Verify frequency-domain metrics."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.n_epochs = n_epochs
        self.blinks = blinks
        self.signal0 = blinks[0]["epoch_signal"]

    def test_single_epoch(self) -> None:
        """Check computation on one epoch."""
        feats = compute_frequency_domain_features(self.signal0, self.sfreq)
        logger.debug("Frequency features: %s", feats)
        self.assertIn("peak_frequency", feats)
        self.assertFalse(feats["peak_power"] is None)

    def test_aggregate_shape(self) -> None:
        """Aggregated DataFrame should contain expected columns."""
        df = aggregate_frequency_domain_features(self.blinks, self.sfreq, self.n_epochs)
        self.assertIn("wavelet_energy_d1", df.columns)
        self.assertEqual(len(df), self.n_epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
