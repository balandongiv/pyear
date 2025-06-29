"""Blink count feature."""
from typing import List, Dict

import logging

logger = logging.getLogger(__name__)


def blink_count_epoch(blinks: List[Dict[str, int]]) -> int:
    """Return the number of blinks for a single epoch.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to one epoch. Each annotation must contain
        at least ``refined_start_frame`` and ``refined_end_frame`` keys.

    Returns
    -------
    int
        Total count of blinks detected in the epoch.
    """
    logger.debug("Counting %s blinks", len(blinks))
    return len(blinks)
