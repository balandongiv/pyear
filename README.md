# 👁️ pyear

**`pyear`** (Python Eye Aspect Ratio) is a comprehensive toolkit for extracting and analyzing **eye aspect ratio (EAR)** and eyelid aperture-based features from eye-tracking data. It’s built to support research and applications in **fatigue detection**, **cognitive state estimation**, and **drowsiness monitoring** through advanced blink dynamics and ocular behavior analysis.

---

## 🧠 Overview

This repository provides:

* Blink event detection based on **eye aspect ratio (EAR)**
* Feature extraction from **eyelid aperture** over 30-second epochs at 30Hz sampling
* Rich descriptors of **blink frequency**, **morphology**, **kinematics**, and **complexity**
* Signal processing tools for time-series, frequency, and entropy-based metrics
* Designed for real-time and batch analysis of **fatigue-related ocular markers**

---

## 🚀 Features

### 📈 Blink Event Features

Quantitative metrics per 30s epoch:

* **Blink count & rate**
* **Inter-blink interval (IBI)**: mean, std, CV, RMSSD, SD1/SD2
* **Burstiness Index**, **Poincaré plots**, **Hurst exponent**

### 👁️ Blink Morphology

Describes each blink’s shape:

* **Duration & asymmetry** (e.g. rise/fall time, time-to-peak)
* **Amplitude**, **area**, **inflection points**
* **FWHM**, **skewness**, **kurtosis**

### ⏱️ Blink Kinematics

Captures motion dynamics:

* **Velocity**, **acceleration**, **jerk**
* **Amplitude-Velocity Ratio (AVR)**
* **Kinematic smoothness and fatigue-linked deceleration**

### 🔋 Energy & Complexity

Quantifies blink intensity and signal unpredictability:

* **Signal energy**, **Teager-Kaiser energy**
* **Line length**, **velocity integral**
* **Entropy** (permutation, spectral), **zero-crossing rates**

### 🌙 Microsleep Detection

* **Prolonged blink count** (>400ms)
* **Micropause count** (100–300ms partial closures)
* **PERCLOS** and maximum closure durations

### 📊 Inter-Blink Features

Analyzes open-eye periods:

* **Baseline drift**, **baseline variability**
* **Low-frequency drift (<0.1Hz)**
* **Eye-opening RMS**, **inter-blink entropy**

### 🎚️ Frequency-Domain Metrics

Power spectral and rhythm descriptors:

* **Blink-rate peak frequency & power**
* **Broadband energy (0.5–2Hz)**, **high-frequency entropy**
* **1/f slope**, **band power ratios**, **wavelet energy (D1–D4)**

### 🔬 Advanced Signal Processing

* **FFT/Wavelet analysis** of blink shapes
* **State-modeling (HMM) and trend analysis**
* **Composite fatigue scores (e.g., JDS, AVR thresholds)**

---


## 🧰 How It Works

1. **Input**: Time-series of eye aspect ratio (EAR) or eyelid aperture data at 30Hz.

   * Supports both `mne.Epochs` objects (for evenly spaced, fixed-length epochs) and long continuous signals.

2. **Blink Detection**: Uses velocity- and threshold-based segmentation on EAR to detect blink onset, peak, and offset.

3. **Epoch Segmentation**: For `mne.Epochs` input, features are computed per 30-second **evenly spaced epoch**.

   * For continuous signals, features can also be computed **directly over the entire signal**, without windowing.

4. **Feature Extraction**: Hierarchical analysis pipeline:

   * **Per-blink** → **Per-epoch (if applicable)** → **Session-level** statistics and trends.

5. **Output**: Features are returned as a structured `pandas.DataFrame` or saved as CSV, ready for modeling, visualization, or temporal analysis.



---

## 📦 Installation

```bash
git clone [https://github.com/your-username/pyear.git](https://github.com/balandongiv/pyear.git)
cd pyear
pip install .
```

---

## 🛠️ Usage Example

```python
from pyear.pipeline import extract_features

# "blinks" should be a list of dictionaries with at least
# ``epoch_index`` and refined frame positions. Each entry typically
# includes ``refined_start_frame``, ``refined_peak_frame``,
# ``refined_end_frame`` and the epoch's signal array under
# ``epoch_signal``.

features = extract_features(
    blinks,
    sfreq=30,
    epoch_len=30,
    n_epochs=n_epochs,
    raw_segments=raw_segments,  # required when using blink_interval_dist
)

features.head()
```

The ``blinks`` list can be built with the provided segmentation helpers
(see ``unitest/fixtures/mock_ear_generation.py``).  When working with a
continuous ``mne.Raw`` recording, the new
``pyear.utils.slice_raw_to_segments`` helper slices the raw file into
30‑second annotated segments with a progress bar.  When requesting the
``blink_interval_dist`` feature, supply ``raw_segments`` as a list of
the original per‑epoch signals.

---

## 📚 Documentation

See [docs/FEATURES.md](docs/FEATURES.md) for detailed descriptions and formulas for:

* Blink waveform descriptors
* Frequency bands
* Complexity measures
* State modeling strategies

---

## 🧪 Applications

* **Driver fatigue detection**
* **Attention monitoring in aviation or surveillance**
* **Clinical drowsiness studies**
* **Cognitive load tracking**
* **Real-time wearable drowsiness systems**

---

## 📈 Planned Enhancements

Perfect — since the repository is just starting and your goal is to implement all the detailed features listed, we can update the **"📈 Planned Enhancements"** section to better reflect the roadmap.

Here’s the revised version:

---

## 📈 Planned Enhancements

This is an early-stage repository. The goal is to implement the **full set of blink and eyelid features** described above — spanning time-domain, frequency-domain, morphological, kinematic, and complexity metrics.

### Core Feature Roadmap

* [ ] Per-blink feature extraction (duration, amplitude, symmetry, etc.)
* [ ] Per-epoch feature extraction (IBI stats, blink rate, RMSSD, entropy, etc.)
* [ ] Inter-blink and open-eye period metrics (baseline drift, PERCLOS, micropause detection)
* [ ] Frequency-domain features (spectral power, 1/f slope, band ratios, wavelet energy)
* [ ] Advanced shape descriptors (skewness, kurtosis, inflection points, waveform energy)
* [ ] Complexity and non-linear features (Hurst exponent, sample entropy, etc.)

### Integration and Extension Plans

* [ ] Integration with EOG and EEG datasets (e.g., blink detection from frontal EEG)


---

