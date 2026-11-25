# Lecture → Code Map (Acoustic Messenger, Option A)

Where each course concept lives in this project, which test proves it, and which plot to show.
Slide decks are named by their file in `lecture_infos/` (the decks' own lecture numbers are not consistent).
Equation numbers `(3-x)` are from Lyons, *Understanding Digital Signal Processing*, ch. 3, the source of the DFT lecture.

---

## 1. Concept map

| Lecture (deck) | Concept | Where in code | Proof / evidence |
|---|---|---|---|
| SaS-w1-1 §1.1 | Sampling: x[n] = x(nT_s), n is dimensionless | `audio_io.load_wav` / `save_wav` (48 kHz) | every WAV in `data/` |
| SaS-w1-1 §1.1.2 | Average power (1/N)Σ\|x[n]\|² | `sync.sliding_rms` (used by `sync.detect_start`): √(average power) | `test_*auto_sync*` |
| SaS-w1-2 §1.3.2 | Discrete-time sinusoid A·sin(ω0·n), ω0 = 2πf/fs | `tx.modulate_fsk` | `test_slide4_example_1011_*` |
| SaS-w2-1 §1.3.3 | e^{j(ω0+2π)n} = e^{jω0n}: only f < fs/2 is unique | `tx.validate_config` (Nyquist check) | `validate_config` raises for f ≥ fs/2 |
| SaS-w3, SaS-w4 §2.1–2.3 | LTI system, impulse response, convolution y = x * h | The room is the channel: y(t) = (s * h)(t) + noise | `docs/qa_study_guide.md` §1, Exp 4 reverb tails on spectrograms |
| SaS-w6 §3.9.2 | Frequency-selective (band-pass) filter | `sync.bandpass` (Butterworth ≈ ideal BPF) | Exp 3 prediction: out-of-band interference is ignored |
| SaS-w9-1 §3.5.2 | Time shift ⇒ phase shift only, \|a_k\| unchanged | `rx` decides on magnitudes only (unknown air delay) | `test_cross_sample_rate_*`, any real recording |
| SaS-w9-1 §3.5.5 | Multiplication in time ⇔ convolution in frequency | Cutting a bit = multiplying by a window ⇒ spectrum convolved with the window's sinc ⇒ leakage | `05_window_functions.png` |
| SaS-w9-1 "Power spectrum" | Magnitude in dB: 20·log10(c_k), 0 dB = peak | `BitDecision.margin_db`, spectrogram colour scale | `*_bit_spectra.png`, `*_spectrogram.png` |
| SaS-w9-2 Table 4.2 | Rectangular pulse ⇔ sinc (2 sin ωT1 / ω) | A T_b-long tone has a sinc spectrum, main lobe 2/T_b | `04_duration_vs_resolution.png` |
| SaS-w11-3 (3-2) | DFT X(m) = Σ x(n)e^{-j2πnm/N} | `dft.direct_dft` | `test_direct_dft_matches_numpy_fft` |
| SaS-w11-3 (3-5) | f_analysis(m) = m·fs/N | `dft.dft_frequencies`, `rx.bit_spectrum` | `test_frequency_bins_and_two_tone_peaks` |
| SaS-w11-3 (3-17) | Integer-cycle tone ⇒ \|X(m)\| = A·N/2 | `dft.single_sided_spectrum` (÷ N/2) | `test_lecture_dft_example_1` (\|X(1)\| = 4) |
| SaS-w11-3 IDFT (3-23) | x(n) = (1/N) Σ X(m)e^{+j2πmn/N} | `dft.direct_idft` | `test_direct_idft_reconstructs_signal` |
| SaS-w11-3 DFT Example 1 | fs = 8 kHz, N = 8: \|X(1)\| = 4 ∠-90°, \|X(2)\| = 2 ∠45° | `run_dft_validation.plot_bins_magnitude_phase` | `test_lecture_dft_example_1`, `02_bins_magnitude_phase.png` |
| SaS-w11-3 Leakage | 3.0 vs 3.4 cycles in N = 64 (X(3) = 32 vs ≈25) | `run_dft_validation.plot_spectral_leakage` | `test_lecture_leakage_example`, `03_spectral_leakage.png` (a) |
| SaS-w11-3 Windowing | Rectangular, triangular, Hanning 0.5-0.5cos(2πn/N), Hamming (3-27 … 3-30) | `dft.window`; the receiver uses `window("hanning", N)` | `test_window_formulas`, `05_window_functions.png` |
| SaS-w11-3 Windowing | A weak tone next to a strong one is only visible after windowing | `run_dft_validation.plot_spectral_leakage` (b) | `test_hanning_reveals_weak_tone` |
| Project slide 13 | Direct DFT validated on short signals; library FFT for the full project | `dft.validate_dft_against_fft`; `rx` uses `np.fft.rfft` | `01_dft_vs_fft_validation.png` |

---

## 2. Why the receiver works (in lecture terms)

1. **Each bit is a short piece of a sinusoid.** Its spectrum is the tone's spectral line convolved with the window's sinc (SaS-w9-1 §3.5.5, SaS-w9-2 Table 4.2). Main lobe width ≈ 2/T_b (rectangular) or 4/T_b (Hanning).
2. **The DFT samples that spectrum at m·fs/N** (3-5). With T_b = 0.1 s at 48 kHz, N = 4800 and the bins are 10 Hz apart.
3. **A bit rarely holds a whole number of cycles**, so energy leaks into other bins (SaS-w11-3 leakage). The Hanning window pushes that leakage down from −13 dB to −31 dB sidelobes, which matters when f0 and f1 are close (Exp 2).
4. **The air delay only changes the phase** (time-shifting property), so comparing |X(f0)| and |X(f1)| works whatever the distance (Exp 4).
5. **The band-pass filter is an LTI frequency-selective system**, so speech or hum outside [f0 − 200, f1 + 200] Hz cannot trigger the start detector (Exp 3).

---

## 3. Lecture exercises (SaS-w11-3, last slides), applied

**Q1.** fs = 1000 Hz and the DFT bin spacing must be exactly 45 Hz. How many samples N?
- Bin spacing = fs/N (3-5) ⇒ N = 1000/45 = 22.2, which is **not an integer**. The specification is impossible as stated.
- Tell the systems engineer: use N = 22 (45.45 Hz bins) or N = 23 (43.48 Hz bins), or change fs to a multiple of 45 (e.g. fs = 990 Hz with N = 22).

**Q2.** N = 500 samples at fs = 3000 Hz. What is the DFT bin spacing?
- fs/N = 3000/500 = **6 Hz**.

**The same rule applied to our project** (fs = 48 kHz, one bit = one DFT):

| T_b | N = fs·T_b | Bin spacing fs/N = 1/T_b | Hanning main-lobe half-width 2/T_b |
|---:|---:|---:|---:|
| 100 ms | 4800 | 10 Hz | 20 Hz |
| 50 ms | 2400 | 20 Hz | 40 Hz |
| 20 ms | 960 | 50 Hz | 100 Hz |
| 10 ms | 480 | 100 Hz | 200 Hz |

This is the core of the Exp 1 (bit duration) and Exp 2 (tone separation) predictions: Δf = f1 − f0 must stay well above the main-lobe width.

---

## 4. Regenerate the lecture figures

```bash
python scripts/run_dft_validation.py    # writes results/plots/dft/01 ... 05
pytest tests/test_dft.py -v             # checks the lecture's numbers
```
