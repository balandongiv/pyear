from __future__ import annotations

from typing import Iterable, Dict, Any, List
import logging
import pandas as pd
import numpy as np

from .frequency_features import compute_frequency_domain_features

logger = logging.getLogger(__name__)


def aggregate_frequency_domain_features(
    blinks: Iterable[Dict[str, Any]],
    sfreq: float,
    n_epochs: int,
) -> pd.DataFrame:
    """Aggregate spectral and wavelet metrics across epochs.

    Parameters
    ----------
    blinks : Iterable[dict]
        Blink annotations containing ``epoch_index`` and ``epoch_signal``.
    sfreq : float
        Sampling frequency in Hertz.
    n_epochs : int
        Number of epochs to aggregate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by epoch with frequency-domain features.
    """
    logger.info("Aggregating frequency-domain features over %d epochs", n_epochs)

    per_epoch_signals: List[np.ndarray | None] = [None for _ in range(n_epochs)]
    for blink in blinks:
        idx = blink["epoch_index"]
        if 0 <= idx < n_epochs and per_epoch_signals[idx] is None:
            per_epoch_signals[idx] = np.asarray(blink["epoch_signal"], dtype=float)

    records = []
    for idx in range(n_epochs):
        signal = per_epoch_signals[idx]
        record = {"epoch": idx}
        if signal is not None:
            feats = compute_frequency_domain_features(signal, sfreq)
            record.update(feats)
        else:
            record.update(
                {
                    "peak_frequency": float("nan"),
                    "peak_power": float("nan"),
                    "band_power_ratio": float("nan"),
                    "one_over_f_slope": float("nan"),
                    "wavelet_energy_d1": float("nan"),
                    "wavelet_energy_d2": float("nan"),
                    "wavelet_energy_d3": float("nan"),
                    "wavelet_energy_d4": float("nan"),
                }
            )
        records.append(record)

    df = pd.DataFrame.from_records(records).set_index("epoch")
    logger.debug("Aggregated frequency-domain DataFrame shape: %s", df.shape)
    return df
