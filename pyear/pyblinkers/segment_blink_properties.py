"""Segment-level blink property extraction utilities.

This module exposes :func:`compute_segment_blink_properties`, which iterates
over a sequence of ``mne.Raw`` segments and extracts detailed blink metrics for
each blink using :class:`FitBlinks` and :class:`BlinkProperties`.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any
import logging
import warnings

import pandas as pd
import mne
from tqdm import tqdm

from .fit_blink import FitBlinks
from .extract_blink_properties import BlinkProperties

logger = logging.getLogger(__name__)


def compute_segment_blink_properties(
    segments: Sequence[mne.io.BaseRaw],
    blink_df: pd.DataFrame,
    params: Dict[str, Any],
    *,
    channel: str = "EEG-E8",
    run_fit: bool = False,
) -> pd.DataFrame:
    """Calculate blink properties for each blink found within every segment.

    Parameters
    ----------
    segments : Sequence[mne.io.BaseRaw]
        Iterable of Raw segments containing blink annotations.
    blink_df : pandas.DataFrame
        DataFrame returned by :func:`pyear.blink_events.generate_blink_dataframe`.
        Expected columns include ``seg_id``, ``start_blink``, ``end_blink``,
        ``outer_start``, ``outer_end``, ``left_zero`` and optionally
        ``right_zero``.
    params : dict
        Parameter dictionary forwarded to :class:`FitBlinks` and
        :class:`BlinkProperties`. Required keys include ``"base_fraction"``,
        ``"shut_amp_fraction"``, ``"p_avr_threshold"`` and ``"z_thresholds"``.
    channel : str, optional
        Channel name used to extract the blink signal from ``segments``.
        Defaults to ``"EEG-E8"``.
    run_fit : bool, optional
        If ``True`` blink fits are computed via :meth:`FitBlinks.fit`.
        This mirrors the original Matlab workflow where only ``start_blink`` and
        ``end_blink`` were provided and fitting was always performed. This may
        drop blinks due to ``NaN`` values in the fit range. The default is
        ``False``.

    Returns
    -------
    pandas.DataFrame
        Concatenated blink-property table for all segments. The returned
        DataFrame contains all columns generated by :class:`BlinkProperties`
        along with ``seg_id`` identifying the source segment.
    """
    logger.info("Computing blink properties for %d segments", len(segments))

    if run_fit:
        warnings.warn(
            "run_fit=True may drop blinks due to NaNs in fit range",
            RuntimeWarning,
        )

    if blink_df.empty:
        logger.info("Blink DataFrame is empty; nothing to compute")
        return pd.DataFrame()

    sfreq = segments[0].info["sfreq"] if segments else 0.0
    all_props = []

    for seg_id, raw in enumerate(tqdm(segments, desc="Segments")):
        rows = blink_df[blink_df["seg_id"] == seg_id].copy()
        rows["start_blink"] = rows["start_blink"].astype(int)
        rows["end_blink"] = rows["end_blink"].astype(int)
        rows["outer_start"] = rows["outer_start"].astype(int)
        rows["outer_end"] = rows["outer_end"].astype(int)
        rows["left_zero"] = rows["left_zero"].astype(int)
        if "right_zero" in rows.columns:
            rows["right_zero"] = rows["right_zero"].fillna(-1).astype(int)
        if rows.empty:
            continue

        signal = raw.get_data(picks=channel)[0]

        fitter = FitBlinks(candidate_signal=signal, df=rows, params=params)
        try:
            fitter.dprocess_segment_raw(run_fit=run_fit)
        except Exception as exc:  # pragma: no cover - safeguard against bad data
            logger.warning("Skipping segment %d due to fit error: %s", seg_id, exc)
            continue

        props = BlinkProperties(
            signal,
            fitter.frame_blinks,
            sfreq,
            params,
            fitted=run_fit,
        ).df
        props["seg_id"] = seg_id
        all_props.append(props)

    if not all_props:
        return pd.DataFrame()

    result = pd.concat(all_props, ignore_index=True)
    logger.info("Computed blink properties for %d blinks", len(result))
    return result
