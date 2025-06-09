"""Main pipeline entry for feature extraction."""

from __future__ import annotations

import logging
from typing import Iterable, Dict, Sequence

import pandas as pd

from .blink_events.event_features import aggregate_blink_event_features
from .morphology import aggregate_morphology_features
from .kinematics import aggregate_kinematic_features

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(
    blinks: Iterable[Dict[str, int]],
    sfreq: float,
    epoch_len: float,
    n_epochs: int,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract blink features using provided blink annotations.

    Parameters
    ----------
    blinks : Iterable[Dict[str, int]]
        Blink annotations with epoch indices and frame positions.
    sfreq : float
        Sampling frequency of the recording.
    epoch_len : float
        Length of each epoch in seconds.
    n_epochs : int
        Total number of epochs.
    features : Sequence[str] | None, optional
        Feature groups to compute. Values from
        :func:`aggregate_blink_event_features` (``"blink_count"``, ``"blink_rate"``,
        ``"ibi"``), ``"morphology`` and ``"kinematics"`` are recognized. ``None``
        computes all available features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with aggregated features per epoch.
    """
    logger.info("Starting feature extraction")

    df_events = aggregate_blink_event_features(
        blinks, sfreq, epoch_len, n_epochs, features
    )

    if features is None or "kinematics" in features:
        df_kin = aggregate_kinematic_features(blinks, sfreq, n_epochs)
        df_events = pd.concat([df_events, df_kin], axis=1)

    if features is None or "morphology" in features:
        df_morph = aggregate_morphology_features(blinks, sfreq, n_epochs)
        df = pd.concat([df_events, df_morph], axis=1)
    else:
        df = df_events

    logger.info("Finished feature extraction")
    return df
