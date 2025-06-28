"""Unit tests for blink interval distribution calculation."""

import unittest
import logging

from pyear.blink_events.event_features.blink_interval_distribution import (
    blink_interval_distribution,
)
from unitest.fixtures.mock_ear_generation import _generate_refined_ear

logger = logging.getLogger(__name__)


class TestBlinkIntervalDistribution(unittest.TestCase):
    """Tests for ``blink_interval_distribution`` utility."""

    def setUp(self) -> None:
        logger.info("Preparing synthetic blink data for distribution tests...")
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.sfreq = sfreq
        self.blinks = blinks
        self.n_intervals = len(blinks) - 1

    def test_histogram_counts(self) -> None:
        """Histogram counts should sum to the number of intervals."""
        df = blink_interval_distribution(self.blinks, self.sfreq, bins=4)
        total = int(df["count"].sum())
        logger.debug(
            "Histogram counts: %s (total %s)",
            df["count"].tolist(),
            total,
        )
        self.assertEqual(total, self.n_intervals)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
