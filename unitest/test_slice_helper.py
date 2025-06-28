"""Tests for the slice_raw_to_epochs helper."""
import unittest
import logging

import numpy as np

from pyear.utils.epoch_aggregator import slice_raw_to_epochs
from unitest.fixtures.mock_ear_generation import _generate_signal_with_blinks

logger = logging.getLogger(__name__)


class TestSliceHelper(unittest.TestCase):
    """Ensure raw slicing works correctly."""

    def setUp(self) -> None:
        self.signal, self.annotations = _generate_signal_with_blinks(100.0, 30.0, 2)

    def test_output_structure(self) -> None:
        """Returned annotations include epoch info."""
        sliced = slice_raw_to_epochs(self.signal, self.annotations, 100.0, 30.0)
        self.assertTrue(all("epoch_index" in b for b in sliced))
        self.assertTrue(all("epoch_signal" in b for b in sliced))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
