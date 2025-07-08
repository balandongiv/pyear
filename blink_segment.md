## 🎯 High-level Objective

Build a **re-usable blink-event DataFrame generator** for unit-testing and visual sanity-checks, starting from a list of 30-s `mne.io.Raw` segments.
Focus first on the **`'EOG-EEG-eog_vert_left'`** channel; the design must remain extensible to other channels later.

---

## 📐 Functional Requirements

1. **Input**

   ```python
   segments: list[mne.io.Raw]          # exactly 60 items, 30 s each
   channel: str = "EOG-EEG-eog_vert_left"
   ```

   *Each `Raw` may or may not contain **blink** annotations (`annot.description == "blink"`)*.

2. **Per-blink processing (per segment)**

    1. Convert annotation `onset` / `duration` → **sample indices** (`start_idx`, `end_idx`) using `sfreq = raw.info['sfreq']` (30 Hz in your recordings).
    2. **Peak index** `peak_idx`: `np.argmax(np.abs(sig[start_idx:end_idx + 1])) + start_idx`
       (absolute amplitude to be robust against polarity).
    3. **Outer window** boundaries
        
       An **outer window** is a segment of the signal that surrounds a blink **peak**, defined per blink as:

       ```python
        outer_start ≤ peak_idx ≤ outer_end
       ```

Say we detect 3 blinks with these `peak_idx` (in sample indices):

```python
peak_0 = 450
peak_1 = 930
peak_2 = 1420
```

And the segment has length `n_samples = 1800` (30s × 30Hz).

We want:

* Blink 0's outer window to extend **until** Blink 1 begins.
* Blink 1's outer window to span **between** Blinks 0 and 2.
* Blink 2's outer window to go **until** the end of the segment.

So:

| Blink # | outer\_start | outer\_end      |
| ------- | ------------ | --------------- |
| 0       | 0            | `peak_1`        |
| 1       | `peak_0`     | `peak_2`        |
| 2       | `peak_1`     | `n_samples - 1` |

This ensures:

* **No overlap** of search windows,
* **Full coverage** of the signal span,
* Search for zero-crossings is **not truncated prematurely**.

---

## 🧮 Algorithm (pseudocode)

```python
n_blinks = len(peaks)
n_samples = len(signal)

for i, peak_idx in enumerate(peaks):
    if i == 0:
        outer_start = 0
    else:
        outer_start = peaks[i - 1]

    if i == n_blinks - 1:
        outer_end = n_samples - 1
    else:
        outer_end = peaks[i + 1]
```

    4. **Zero-crossings**

       ```python
       left_zero, right_zero = left_right_zero_crossing(
           sig, peak_idx, outer_start, outer_end
       )
       ```

3. **Output**
   `pd.DataFrame` with **one row per blink** and at least these columns:

   | col name         | dtype      | meaning (sample index)                   |
      | ---------------- | ---------- | ---------------------------------------- |
   | `seg_id`         | int        | 0-based position in `segments` list      |
   | `blink_id`       | int        | 0-based within its segment               |
   | `start_idx`      | int        | blink onset sample                       |
   | `peak_idx`       | int        | max-amplitude sample                     |
   | `end_idx`        | int        | blink offset sample                      |
   | `outer_start`    | int        | outer window start                       |
   | `outer_end`      | int        | outer window end                         |
   | `left_zero_idx`  | int        | nearest < 0 crossing left of `peak_idx`  |
   | `right_zero_idx` | int \|None | nearest < 0 crossing right of `peak_idx` |

   *Times in seconds* can be derived downstream (`idx / sfreq`)—keep raw indices in the master table.

---

## 🧮 Algorithmic Helper

```python
def left_right_zero_crossing(
    candidate_signal: np.ndarray,
    max_frame: int,
    outer_start: int,
    outer_end: int
) -> tuple[int, Optional[int]]:
    """
    Return indices of nearest negative-valued samples (zero-crossings proxy)
    on the left and right of `max_frame`, constrained to [outer_start, outer_end].

    Behaviour & edge-cases:
    - If no negative sample in left search window, extend to [0, max_frame).
    - If still none, raise ValueError (data likely DC-shifted).
    - If no negative sample in right window, extend to (max_frame, len(signal)).
    - If still none, return (left_zero, None).
    - Validate: left_zero ≤ max_frame ≤ right_zero (when right_zero found).
    """
```

---

## 🧪 Unit-test Requirements

*File:* `test_blink_dataframe.py`

2. **Assertions**

    * For sanity, the number of blinks in each segments is available in unitest/ear_eog_blink_count_epoch.csv
      
    * Row count equals total blinks injected.

3. **Progress bars & logging**

    * All loops wrapped with `tqdm(...)`.
    * `INFO` log on entry/exit of public functions; `DEBUG` for intermediate arrays.

---

## 📊 Visual Sanity Check (compulsory)

*Utility script:* `plot_blinks_report.py`

Generate an **MNE Report** page that consist all the blink in all the segments:

1. Raw trace of each of the blink with low-opacity line.
2. Clear label of each blink with:

    * blink `start_idx`, `peak_idx`, `end_idx` (different colours / labels).
    * `left_zero_idx`, `right_zero_idx`.
3. Save to HTML (one file for whole dataset).


---


