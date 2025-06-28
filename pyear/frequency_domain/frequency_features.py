"""Power spectral and wavelet energy features."""
from __future__ import annotations

from typing import Dict
import logging

import numpy as np
from scipy.signal import welch
import pywt

logger = logging.getLogger(__name__)


def compute_frequency_domain_features(epoch_signal: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute spectral peak and wavelet energy metrics for an epoch.

    Parameters
    ----------
    epoch_signal : numpy.ndarray
        Eyelid aperture samples for one epoch.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    dict
        Dictionary containing power spectral and wavelet energy features.
    """
    if epoch_signal.size == 0:
        return {
            "peak_frequency": float("nan"),
            "peak_power": float("nan"),
            "band_power_ratio": float("nan"),
            "one_over_f_slope": float("nan"),
            "wavelet_energy_d1": float("nan"),
            "wavelet_energy_d2": float("nan"),
            "wavelet_energy_d3": float("nan"),
            "wavelet_energy_d4": float("nan"),
        }

    freqs, psd = welch(epoch_signal, fs=sfreq, nperseg=min(256, epoch_signal.size))
    peak_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_idx])
    peak_power = float(psd[peak_idx])

    low_mask = (freqs >= 0.5) & (freqs <= 2.0)
    high_mask = (freqs > 2.0) & (freqs <= 10.0)
    low_power = float(np.trapz(psd[low_mask], freqs[low_mask]))
    high_power = float(np.trapz(psd[high_mask], freqs[high_mask]))
    band_ratio = low_power / high_power if high_power != 0 else float("nan")

    if np.all(psd[1:] > 0):
        log_freqs = np.log10(freqs[1:])
        log_psd = np.log10(psd[1:])
        slope, _ = np.polyfit(log_freqs, log_psd, 1)
        one_over_f_slope = float(-slope)
    else:
        one_over_f_slope = float("nan")

    coeffs = pywt.wavedec(epoch_signal, "db4", level=4)
    energies = {
        f"wavelet_energy_d{idx}": float(np.sum(detail ** 2))
        for idx, detail in enumerate(coeffs[1:], start=1)
    }

    features = {
        "peak_frequency": peak_freq,
        "peak_power": peak_power,
        "band_power_ratio": band_ratio,
        "one_over_f_slope": one_over_f_slope,
    }
    features.update(energies)
    logger.debug("Frequency features: %s", features)
    return features
