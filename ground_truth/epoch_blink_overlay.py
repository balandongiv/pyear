#!/usr/bin/env python3
"""
1. **Strategy 1 – Crop & Save Mini-Raws:**
   Splits the continuous Raw into multiple 30-second Raw objects using `.crop()`,
   automatically retaining blink annotations. Each segment is saved to disk for visual
   inspection using `raw.plot()`.

Usage:
    python blink_dev_workarounds.py --file ear_eog.fif
"""
import logging
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BLINK_LABEL = "blink"        # Description of blink annotations in the Raw
EPOCH_LEN = 30.0              # Epoch duration in seconds

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Strategy 1: Crop & Save mini-raws
# -----------------------------------------------------------------------------

def slice_into_mini_raws(
        raw: mne.io.BaseRaw,
        out_dir: Path,
        epoch_len: float = EPOCH_LEN
) -> None:
    """
    Save each `epoch_len`-second segment as its own Raw FIF, preserving blink annotations.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording with blink annotations.
    out_dir : Path
        Directory where epoch files will be saved (will be created if missing).
    epoch_len : float, optional
        Duration of each segment in seconds.

    Returns
    -------
    None
    """
    logger.info("Entering slice_into_mini_raws")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_time = raw.times[-1]
    n_epochs = int(np.ceil(total_time / epoch_len))

    for i in tqdm(range(n_epochs), desc="Cropping epochs", unit="epoch"):
        start = i * epoch_len
        stop = min(start + epoch_len, total_time)
        mini = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)

        # Shift annotation onsets so each mini-raw starts at 0
        ann = mini.annotations
        shifted = mne.Annotations(
            onset=ann.onset - start,
            duration=ann.duration,
            description=ann.description
        )
        mini.set_annotations(shifted)
        mini.plot(
            n_channels=10,
            scalings='auto',
            title=f"Mini Raw {i+1}/{n_epochs} ({start:.2f}s - {stop:.2f}s)",
            show=True, block=False
        )
        fname = out_dir / f"epoch_{i:04d}_{start:07.2f}s-{stop:07.2f}s_raw.fif"
        mini.save(fname, overwrite=True)
    logger.info("Exiting slice_into_mini_raws")



# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Load the continuous FIF, then run both dev workarounds:
      1) slice_into_mini_raws → saves per-epoch FIFs in `debug_epochs/`
      2) epoch_to_rawarray_with_blinks → creates RawArrays and saves in `debug_rawarrays/`
    """
    # Root logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.DEBUG)

    fif_path = Path("../unitest/ear_eog.fif")
    logger.info(f"Reading raw FIF from {fif_path}")
    raw = mne.io.read_raw_fif(str(fif_path), preload=True)

    # Strategy 1: Crop & Save
    slice_into_mini_raws(raw, Path("../debug_epochs"))




if __name__ == "__main__":
    main()
