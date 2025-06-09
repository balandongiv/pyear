from typing import List, Dict, Any, Tuple

import mne
import numpy as np

from unitest.fixtures.ear_refinement_stub import refine_ear_extrema_and_threshold_stub


def _generate_signal_with_blinks(sfreq: float, epoch_len: float, n_epochs: int) -> Tuple[np.ndarray, List[Dict]]:
    """Generates a synthetic signal with embedded blink-like patterns.

    Args:
        sfreq (float): Sampling frequency.
        epoch_len (float): Length of one epoch in seconds.
        n_epochs (int): Number of epochs to generate.

    Returns:
        Tuple[np.ndarray, List[Dict]]: Simulated signal array and blink annotations.
    """
    n_samples = int(epoch_len * sfreq * n_epochs)
    rng = np.random.default_rng(42)
    signal = 0.32 + rng.normal(scale=0.005, size=n_samples)
    signal = np.clip(signal, 0.2, 0.38)

    def insert_blink(signal: np.ndarray, idx: int) -> Tuple[int, int, int]:
        shoulder = 0.28
        trough_val = 0.09
        span = int(0.05 * sfreq)
        s = max(0, idx - span)
        e = min(len(signal) - 1, idx + span)
        signal[s:idx] = shoulder
        signal[idx] = trough_val
        signal[idx + 1 : e + 1] = shoulder
        return s, idx, e

    annotations = []
    troughs = [200, 500, 800, 1300, 1700, 2500, 4200, 4600]
    for i, t in enumerate(troughs, start=1):
        s, tr, e = insert_blink(signal, t)
        annotations.append({"id": i, "start": s, "trough": tr, "end": e})

    return signal, annotations


def _create_epochs(signal: np.ndarray, sfreq: float, epoch_len: float) -> mne.Epochs:
    """Creates MNE epochs from the raw signal.

    Args:
        signal (np.ndarray): The signal data.
        sfreq (float): Sampling frequency.
        epoch_len (float): Length of one epoch.

    Returns:
        mne.Epochs: MNE Epochs object.
    """
    info = mne.create_info(["EAR"], sfreq, ["misc"])
    raw = mne.io.RawArray(signal[np.newaxis, :], info, verbose=False)
    events = mne.make_fixed_length_events(raw, id=1, duration=epoch_len)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=epoch_len - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs

def _refine_ear(
        epochs: mne.Epochs,
        annotations: List[Dict[str, int]],
        sfreq: float
) -> List[Dict[str, Any]]:
    """Refines blink annotations for each epoch by locating precise start, peak, and end frames
    using ear‐sensor signal extrema and thresholds.

    Args:
        epochs (mne.Epochs):
            An MNE Epochs object containing the segmented ear‐sensor data.
        annotations (List[Dict[str, int]]):
            A list of original blink annotations. Each dict must have:
              - 'id': unique blink identifier
              - 'start': absolute start sample index in the full signal
              - 'trough': absolute peak/trough sample index in the full signal
              - 'end': absolute end sample index in the full signal
        sfreq (float):
            Sampling frequency in Hz. Used to convert time‐based parameters into frames.

    Returns:
        List[Dict[str, Any]]:
            A list of refined blink entries. Each dict contains:
              - 'epoch_index' (int): 0‐based index of the epoch containing the blink
              - 'epoch_signal' (np.ndarray): 1D signal array for that epoch
              - 'original_annotation' (Dict[str, int]): the original annotation dict
              - 'refined_start_frame' (int): refined start frame index *within* the epoch
              - 'refined_peak_frame' (int): refined trough frame index *within* the epoch
              - 'refined_end_frame' (int): refined end frame index *within* the epoch

    """
    all_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    refined_results: List[Dict[str, Any]] = []

    for epoch_index in range(len(epochs)):
        epoch_signal = all_data[epoch_index, 0]
        epoch_start_sample = epochs.events[epoch_index, 0]

        # Filter annotations to those whose troughs fall in this epoch
        epoch_annotations = [
            ann for ann in annotations
            if epoch_start_sample <= ann['trough'] < epoch_start_sample + len(epoch_signal)
        ]

        for annotation in epoch_annotations:
            # Original absolute indices
            original_start = annotation['start']
            original_trough = annotation['trough']
            original_end = annotation['end']

            # Convert to indices relative to the start of this epoch
            rel_start_frame = original_start - epoch_start_sample
            rel_trough_frame = original_trough - epoch_start_sample
            rel_end_frame = original_end - epoch_start_sample

            # Call the refinement stub
            refined_start_frame, refined_peak_frame, refined_end_frame = refine_ear_extrema_and_threshold_stub(
                epoch_signal,
                rel_start_frame,
                rel_end_frame,
                rel_trough_frame,
                local_max_prominence=0.015,
                search_expansion_frames=int(0.1 * sfreq),
                value_threshold=0.23,
            )

            refined_results.append({
                'epoch_index': epoch_index,
                'epoch_signal': epoch_signal,
                'original_annotation': annotation,
                'refined_start_frame': refined_start_frame,
                'refined_peak_frame': refined_peak_frame,
                'refined_end_frame': refined_end_frame,
            })

    return refined_results

def _generate_refined_ear() -> Tuple[List, float, float, int]:
    """Top-level function to generate and refine synthetic blinks.

    Returns:
        Tuple[List, float, float, int]: Refined blink list, sampling frequency,
                                        epoch length, and number of epochs.
    """
    sfreq = 100.0
    epoch_len = 10.0
    n_epochs = 5
    signal, annotations = _generate_signal_with_blinks(sfreq, epoch_len, n_epochs)
    epochs = _create_epochs(signal, sfreq, epoch_len)
    refined = _refine_ear(epochs, annotations, sfreq)
    return refined, sfreq, epoch_len, n_epochs


if __name__ == "__main__":
    refined_blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
    print(f"Generated {len(refined_blinks)} refined blinks (sfreq={sfreq}, epoch_len={epoch_len}, n_epochs={n_epochs})")

    for blink in refined_blinks:
        # Exclude the signal for printing
        blink_info = {k: v for k, v in blink.items() if k != 'epoch_signal'}
        print(blink_info)
