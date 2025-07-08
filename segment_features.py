"""Extract blink-related features for 30-second segments of a raw FIF file."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict

import mne
import pandas as pd
from tqdm import tqdm

from pyear.utils.epochs import slice_raw_into_epochs
from pyear.energy_complexity import compute_time_domain_features
from pyear.frequency_domain.segment_features import compute_frequency_domain_features

logger = logging.getLogger(__name__)


def process_file(raw_path: Path, channel: str) -> pd.DataFrame:
    """Process one raw file and return per-segment features.

    Parameters
    ----------
    raw_path : pathlib.Path
        Path to the raw FIF recording.
    channel : str
        Name of the data channel used for feature extraction.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a row for each 30-second segment.
    """
    logger.info("Loading raw file: %s", raw_path)
    raw = mne.io.read_raw_fif(str(raw_path), preload=False, verbose=False)
    segments, _, _, _ = slice_raw_into_epochs(raw, epoch_len=30.0, blink_label=None)
    sfreq = raw.info["sfreq"]

    records: List[Dict[str, float]] = []
    for seg_idx, segment in enumerate(tqdm(segments, desc="Segments", unit="seg")):
        signal = segment.get_data(picks=channel)[0]
        time_feats = compute_time_domain_features(signal, sfreq)
        freq_feats = compute_frequency_domain_features([], signal, sfreq)
        record = {"segment_index": seg_idx}
        record.update(time_feats)
        record.update(freq_feats)
        records.append(record)

    df = pd.DataFrame(records)
    logger.info("Computed features for %d segments", len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_file", type=Path, help="Path to raw FIF file")
    parser.add_argument(
        "--channel",
        default="EOG-EEG-eog_vert_left",
        help="Channel name used for feature extraction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = process_file(args.raw_file, args.channel)
    if args.output:
        df.to_csv(args.output, index=False)
        logger.info("Saved features to %s", args.output)
    else:
        print(df)


if __name__ == "__main__":
    main()

