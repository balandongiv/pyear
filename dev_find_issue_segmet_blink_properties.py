"""
Debugging script to inspect compute_segment_blink_properties on annotated segments.
"""

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyear.utils.epochs import slice_raw_into_epochs
from pyear.blink_events import generate_blink_dataframe
from pyear.pyblinkers.segment_blink_properties import compute_segment_blink_properties

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = r"C:\Users\balan\IdeaProjects\pyear\unitest\ear_eog.fif"
CHANNEL = "EEG-E8"
EPOCH_LEN = 30.0
PARAMS = {
    "base_fraction": 0.5,
    "shut_amp_fraction": 0.9,  # ⚠️ Legacy value from MATLAB — may require tuning
    "p_avr_threshold": 3,      # ⚠️ Legacy, currently unused
    "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),  # ⚠️ Legacy, currently unused
}

# --- Load Data ---
logger.info("Loading raw data from %s", RAW_PATH)
raw = mne.io.read_raw_fif(RAW_PATH, preload=False, verbose=False)

logger.info("Slicing into 30-second segments")
segments, _, _, _ = slice_raw_into_epochs(
    raw, epoch_len=EPOCH_LEN, blink_label=None
)

logger.info("Generating blink metadata for channel: %s", CHANNEL)
blink_df = generate_blink_dataframe(
    segments, channel=CHANNEL, blink_label=None
)

# --- Compute Features ---
logger.info("Computing segment blink properties")
df = compute_segment_blink_properties(
    segments, blink_df, PARAMS, channel=CHANNEL
)

# --- Display & Basic Checks ---
print("\n🔍 First few rows of result:")
print(df.head())

print("\n📊 Summary:")
print(f"Total blinks in blink_df: {len(blink_df)}")
print(f"Total entries in result: {len(df)}")
print(f"Unique segments in result: {df['seg_id'].nunique()}")

expected_cols = {
    "duration_base",
    "pos_amp_vel_ratio_zero",
    "closing_time_zero",
}
missing_cols = expected_cols - set(df.columns)
if missing_cols:
    print(f"❌ Missing expected columns: {missing_cols}")
else:
    print("✅ All expected columns are present.")

if not set(df["seg_id"].unique()).issubset(set(blink_df["seg_id"].unique())):
    print("⚠️ Some segment IDs in result are not in blink_df")

