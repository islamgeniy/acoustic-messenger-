# Controlled Experiment Protocol — Acoustic Messenger (Option A)

**Team:** DFT Squad  
**Baseline Configuration (held constant unless explicitly varied):**
- **Payload:** 32-bit sequence `10110010111001010110100110010110` (`data/transmitted/baseline_32bit.wav`)
- **Modulation:** Continuous-phase 2-FSK, $f_0 = 1{,}000\text{ Hz}$ (`0`), $f_1 = 2{,}000\text{ Hz}$ (`1`), $T_b = 0.10\text{ s}$
- **Hardware:** Laptop A (transmitter speaker at fixed 70% volume) $\rightarrow$ air $\rightarrow$ Laptop B (receiver mic at fixed input gain)
- **Baseline Acoustic Path:** `0.5 m` direct line-of-sight (LOS) in a quiet room
- **Trials per Setting:** 3 independent recordings per setting (**all trials, including failures, are retained in `results/trials.csv`**)

---

## Experiment 1: Bit Duration ($T_b$)
- **Leader:** Shuhratov Abdulloh
- **Investigation Question:** How fast can we transmit reliably?
- **Varied Factor:** $T_b \in \{100\text{ ms},\; 50\text{ ms},\; 20\text{ ms},\; 10\text{ ms}\}$ (`exp1_tb100ms` … `exp1_tb10ms`)
- **Fixed Factors:** $f_0 = 1{,}000\text{ Hz}$, $f_1 = 2{,}000\text{ Hz}$, quiet room, `0.5 m` LOS, fixed speaker volume.
- **Pre-Experiment Prediction:**
  - At $T_b = 100\text{ ms}$ ($\Delta f = 1/T_b = 10\text{ Hz}$) and $50\text{ ms}$ ($\Delta f = 20\text{ Hz}$), the spectral main lobes at $1{,}000\text{ Hz}$ and $2{,}000\text{ Hz}$ remain narrow and well separated, yielding $\text{BER} \approx 0\%$.
  - At $T_b = 20\text{ ms}$ and especially $10\text{ ms}$ ($\Delta f = 100\text{ Hz}$), two physical effects degrade decoding: (1) wider main lobes reduce SNR per bit (fewer samples integrated in the FFT), and (2) room reverberation ($\sim 20\text{–}50\text{ ms}$ decay) spills energy from the previous bit into the current bit interval (inter-symbol interference, ISI), causing BER to rise sharply.

---

## Experiment 2: Tone Separation ($\Delta f = f_1 - f_0$)
- **Leader:** Xojambetov Islam
- **Investigation Question:** How close can the frequencies be?
- **Varied Factor:** $f_0 = 1{,}000\text{ Hz}$ fixed, $f_1 \in \{2{,}000,\; 1{,}200,\; 1{,}050,\; 1{,}010\}\text{ Hz}$ ($\Delta f \in \{1000,\; 200,\; 50,\; 10\}\text{ Hz}$)
- **Fixed Factors:** $T_b = 0.10\text{ s}$ (nominal DFT bin spacing $1/T_b = 10\text{ Hz}$, Hann main-lobe half-width $\approx 20\text{ Hz}$), quiet room, `0.5 m` LOS.
- **Pre-Experiment Prediction:**
  - $\Delta f = 1{,}000\text{ Hz}$ and $200\text{ Hz}$ are far wider than the Hann main lobe ($\sim 20\text{ Hz}$), so BER should stay near $0\%$.
  - At $\Delta f = 50\text{ Hz}$, sidelobe leakage and minor acoustic/clock frequency shifts start reducing decision margin, though clean recordings should still mostly decode.
  - At $\Delta f = 10\text{ Hz}$ (equal to $1/T_b$), the Hann main lobes of $f_0$ and $f_1$ overlap directly; spectral leakage and minor room noise will produce substantial decoding errors (failed/difficult case).

---

## Experiment 3: Background Interference
- **Leader:** Mirobidov Mirakbar
- **Investigation Question:** Which interference causes decoding errors?
- **Varied Factor (3 settings $\times$ 3 trials):**
  1. `Quiet baseline` (no added noise)
  2. `Out-of-band interference` (e.g., low-frequency human speech / desk tapping / 400 Hz tone played from a phone near Laptop B)
  3. `In-band interference` (e.g., loud 1,000 Hz or 2,000 Hz tone, whistling, or loud music rich in 1–2 kHz energy played near Laptop B)
- **Fixed Factors:** `baseline_32bit.wav` ($1{,}000 / 2{,}000\text{ Hz}, T_b = 0.10\text{ s}$), `0.5 m` LOS, fixed speaker volume.
- **Pre-Experiment Prediction:**
  - Out-of-band interference will look noisy in the **time-domain waveform**, yet the receiver's FFT only compares spectral energy near $1{,}000\text{ Hz}$ and $2{,}000\text{ Hz}$ — so BER should remain near $0\%$!
  - In-band interference near $1{,}000\text{ Hz}$ or $2{,}000\text{ Hz}$ adds spectral magnitude directly into one of the decision bins, biasing decisions toward that frequency and causing high BER.

---

## Experiment 4: Distance & Position (Acoustic Path — Featured Analysis)
- **Leader:** Malikov Amirxon
- **Investigation Question:** How does the acoustic path affect reception?
- **Varied Factor (4 positions $\times$ 3 trials):**
  1. `0.5m LOS` — Direct line-of-sight at 0.5 m (baseline)
  2. `2.0m LOS` — Direct line-of-sight at 2.0 m
  3. `5.0m LOS` — Across a room or hallway at 5.0 m
  4. `2.0m NLOS (blocked)` — 2.0 m apart with Laptop A's speaker turned 180° away or obstructed behind a thick textbook/monitor so sound arrives via wall reflections & diffraction
- **Fixed Factors:** `baseline_32bit.wav` ($1{,}000 / 2{,}000\text{ Hz}, T_b = 0.10\text{ s}$), fixed speaker volume, quiet room.
- **Pre-Experiment Prediction:**
  - As distance increases ($0.5\text{ m} \rightarrow 2.0\text{ m} \rightarrow 5.0\text{ m}$), inverse-square spreading ($1/r$ amplitude) reduces received SNR smoothly, while the ratio of reflected room reverberation to direct path increases.
  - In `2.0m NLOS (blocked)`, shorter wavelengths ($2{,}000\text{ Hz}$, $\lambda \approx 17\text{ cm}$) diffract less easily around obstacles and are absorbed more strongly than $1{,}000\text{ Hz}$ ($\lambda \approx 34\text{ cm}$). We predict **frequency-dependent attenuation** (shown in `exp4_distance_position_freq_attenuation.png`), causing weaker $f_1$ peaks and more `1 -> 0` bit errors, alongside visible reverberation tails on the STFT spectrogram.

---

## How to Run Real Recordings & Generate All Figures

1. **Generate all TX files (already run once, safe to re-run):**
   ```bash
   .venv/bin/python scripts/prepare_experiment_audio.py
   ```
2. **Record each trial on Laptop B while playing the corresponding WAV on Laptop A:**
   ```bash
   # Example: Exp 4, 2.0m NLOS, Trial 1
   .venv/bin/python scripts/record_trial.py \
       -o data/recorded/exp4_distance_position/2p0m_nlos_t1.wav \
       --duration 6.0 \
       --decode-meta data/transmitted/baseline_32bit.json
   ```
3. **Copy `data/recorded/manifest_template.json` to `data/recorded/manifest.json`** (keep only recorded files, fill in `start_time` if manual alignment is needed for very noisy/far trials, and add brief `notes`).
4. **Run the batch evaluation & plotting pipeline:**
   ```bash
   .venv/bin/python scripts/run_experiment_analysis.py --manifest data/recorded/manifest.json
   ```
   This updates `results/trials.csv` and generates all labelled waveforms, spectrograms, bit-decision plots, pass-vs-fail bit spectra, BER/SNR curves, and the frequency-attenuation chart under `results/plots/`.
