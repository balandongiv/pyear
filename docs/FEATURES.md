# Blink Feature Overview

This document summarizes the available blink metrics in `pyear`.

## Frequency-Domain Features

* **blink_rate_peak_freq** – dominant blink rhythm between 0.1‑0.5 Hz.
* **blink_rate_peak_power** – spectral power at that peak frequency.
* **broadband_power_0_5_2** – total power of the eyelid signal between 0.5‑2 Hz.
* **broadband_com_0_5_2** – center of mass of the 0.5‑2 Hz band.
* **high_freq_entropy_2_13** – spectral entropy in the 2‑13 Hz band.
* **one_over_f_slope** – slope of the 1/f fit from 0.5‑13 Hz.
* **band_power_ratio** – (0.5‑2 Hz) / (2‑13 Hz) power ratio.
* **wavelet_energy_d1..d4** – energies of detail levels from a four‑level DWT.
