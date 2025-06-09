"""Unit tests for blink kinematic feature extraction."""
import unittest
import math
import logging

from pyear.kinematics.kinematic_features import compute_kinematic_features
from unitest.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestKinematicFeatures(unittest.TestCase):
    """Tests for kinematic feature calculations."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.per_epoch = [[] for _ in range(n_epochs)]
        for blink in blinks:
            self.per_epoch[blink["epoch_index"]].append(blink)

    def test_first_epoch_features(self) -> None:
        """Verify kinematic metrics for the first epoch."""
        feats = compute_kinematic_features(self.per_epoch[0], self.sfreq)
        logger.debug(f"Kinematic features epoch 0: {feats}")
        self.assertTrue(math.isclose(feats["blink_velocity_mean"], 9.5))
        self.assertTrue(math.isclose(feats["blink_acceleration_mean"], 950.0))
        self.assertTrue(math.isclose(feats["blink_jerk_mean"], 71250.0))
        self.assertTrue(math.isclose(feats["blink_avr_mean"], 0.02))

    def test_nan_with_no_blinks(self) -> None:
        """Epoch without blinks should yield NaNs."""
        feats = compute_kinematic_features(self.per_epoch[3], self.sfreq)
        logger.debug(f"Kinematic features epoch 3: {feats}")
        self.assertTrue(math.isnan(feats["blink_velocity_mean"]))
        self.assertTrue(math.isnan(feats["blink_avr_mean"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
