"""Utility functions for pyear."""
from .segments import slice_raw_to_segments
from .epochs import (
    slice_raw_into_epochs,
    save_epoch_raws,
    generate_epoch_report,
    slice_into_mini_raws,
)

__all__ = [
    "slice_raw_to_segments",
    "slice_raw_into_epochs",
    "save_epoch_raws",
    "generate_epoch_report",
    "slice_into_mini_raws",
]
