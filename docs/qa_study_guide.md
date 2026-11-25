# Individual Oral Q&A Study Guide — Acoustic Messenger (Option A)

> Designed for the **60% Individual Understanding** assessment (25% Fourier concepts, 20% Prediction & diagnosis, 15% Ownership of work; 5–7 minutes per student, no AI or teammate assistance).

---

## 1. Complete Signal Path (Be Ready to Trace End-to-End)

1. **Bits $\rightarrow$ Continuous-Phase 2-FSK (`src/tx.py`):**
   - Each bit $b_i \in \{0, 1\}$ maps to frequency $f_0 = 1{,}000\text{ Hz}$ or $f_1 = 2{,}000\text{ Hz}$ for duration $T_b = 0.1\text{ s}$ ($N_b = f_s T_b = 4{,}800$ samples at $f_s = 48{,}000\text{ Hz}$).
   - Phase is integrated continuously across bit boundaries: $\phi_{i+1} = (\phi_i + 2\pi f_i T_b) \bmod 2\pi$. Consecutive identical bits (like `11` in `1011`) form a single uninterrupted tone lasting $0.2\text{ s}$, with zero phase-jump clicks even when $f_i T_b$ is non-integer.
2. **Digital $\rightarrow$ Acoustic Channel $\rightarrow$ Digital:**
   - Laptop A DAC + speaker converts PCM WAV to acoustic pressure waves.
   - Air & room act as a linear time-invariant (LTI) channel with impulse response $h(t)$ (delay, inverse-square attenuation $1/r$, frequency-dependent absorption/diffraction, and multipath reverberation) plus additive room noise $w(t)$:
     $$y(t) = (s * h)(t) + w(t)$$
   - Laptop B microphone + ADC samples $y(t)$ into a recorded WAV file (even if recorded at 44.1 kHz instead of 48 kHz, the receiver adapts using the WAV's own `sample_rate`).
3. **Synchronization & Slicing (`src/rx.py`):**
   - Remove DC offset (`signal - mean(signal)`).
   - Locate payload start $t_0$ via either **manual `--start-time`**, **automatic in-band energy onset**, or **phase-invariant preamble matched filtering** (Hilbert envelope of cross-correlation).
   - Slice each bit interval $[t_0 + i T_b,\; t_0 + (i+1) T_b]$ with an optional 5% edge guard margin to avoid reverb spillover at transitions.
4. **Windowing + FFT + Non-Coherent Detection (`src/rx.py`):**
   - Multiply each bit segment by a **Hann window** to suppress spectral leakage sidelobes.
   - Compute `numpy.fft.rfft` (zero-padded for smooth peak sampling), measure peak spectral magnitude near $f_0$ and $f_1$, and decide `'1'` if $|X(f_1)| > |X(f_0)|$ else `'0'`.
   - Evaluate $\text{BER} = \frac{\text{incorrect bits}}{\text{transmitted bits}}$ and estimate per-bit SNR (dB) against the median out-of-band noise floor.

---

## 2. Core Fourier Concepts (25% Individual Marks)

- **Direct DFT Equation:**
  $$X[m] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi n m / N}, \qquad f_m = \frac{m f_s}{N}$$
  - Direct DFT computes $N$ sums of $N$ terms $\Rightarrow O(N^2)$ operations (validated in `src/dft.py` against `numpy.fft.fft` to $<10^{-10}$ max error).
  - Library FFT uses Cooley–Tukey divide-and-conquer $\Rightarrow O(N \log N)$ (at $N=512$, ~600$\times$ faster in our benchmark).
- **Frequency Resolution & Bit Duration:**
  $$\Delta f = \frac{f_s}{N_b} = \frac{1}{T_b}$$
  - Baseline $T_b = 0.1\text{ s} \Rightarrow \Delta f = 10\text{ Hz}$ bin spacing.
  - Shortening $T_b$ to $0.01\text{ s}$ widens bin spacing to $100\text{ Hz}$ (and Hann main lobe to $\sim 200\text{ Hz}$), lowering frequency selectivity and making the receiver vulnerable to room reverberation.
- **Why Magnitude (Non-Coherent) Instead of Phase?**
  - Sound travels at $c \approx 343\text{ m/s}$; a 2 kHz wave has wavelength $\lambda = c/f \approx 17\text{ cm}$. Moving a laptop by just $8.5\text{ cm}$ flips phase by $180^\circ$ ($\pi$ rad)! Using FFT **magnitude** $|X[k]|$ makes detection completely immune to unknown acoustic propagation delay and phase shifts.
- **Spectral Leakage & Windowing:**
  - Multiplying a sinusoid in time by a rectangular bit window corresponds to **convolution** in frequency with a $\text{sinc}$ function.
  - If a tone does not complete an exact integer number of cycles in the window, energy leaks across all bins via high $\text{sinc}$ sidelobes ($-13\text{ dB}$).
  - A **Hann window** reduces sidelobes dramatically ($-31\text{ dB}$ and fast roll-off) at the cost of doubling the main-lobe width (from $2/T_b$ to $4/T_b$) — great for wide separation, important trade-off when $\Delta f$ is very small!
- **Nyquist Criterion:**
  - Highest representable frequency is $f_s / 2$ ($24{,}000\text{ Hz}$ at $48\text{ kHz}$). Both $1{,}000\text{ Hz}$ and $2{,}000\text{ Hz}$ sit well below Nyquist and comfortably inside typical laptop speaker/mic passbands ($200\text{ Hz}–8\text{ kHz}$).

---

## 3. "Predict Before Running" — Likely Instructor Questions (20% Marks)

| Instructor Question | Clear Prediction & Fourier/Physical Explanation |
| :--- | :--- |
| **"What happens if we cut $T_b$ from $0.1\text{ s}$ to $0.01\text{ s}$?"** | Data rate jumps $10\times$, but frequency resolution worsens $10\times$ ($\Delta f = 100\text{ Hz}$), each bit integrates $10\times$ fewer samples (lower SNR), and room reverberation ($\sim 20\text{–}50\text{ ms}$) bleeds the previous bit's tone into the next interval (ISI), raising BER. |
| **"What if we move $f_1$ from $2000\text{ Hz}$ down to $1010\text{ Hz}$?"** | At $T_b = 0.1\text{ s}$, $\Delta f = 10\text{ Hz}$ equals a single DFT bin width; Hann main lobes ($\sim 20\text{ Hz}$ half-width) overlap directly, so leakage + slight acoustic noise cause frequent bit flips. |
| **"Someone talks loudly at 200–500 Hz vs. plays a 1000 Hz tone?"** | Low-frequency speech is **out-of-band**: the time waveform looks messy, but the FFT bins near $1{,}000$ and $2{,}000\text{ Hz}$ stay clean (BER $\approx 0\%$). A $1{,}000\text{ Hz}$ tone is **in-band**: it injects energy directly into the `0` bin, causing `1 -> 0` errors whenever interference exceeds the received `1` tone. |
| **"What if `start_time` is shifted by $+0.05\text{ s}$ (half a bit)?"** | Every slicing window now contains half of bit $i$ and half of bit $i+1$. Whenever adjacent bits differ (`01` or `10`), both $f_0$ and $f_1$ appear with roughly equal magnitude ($\approx 50\%$ each), so tiny noise decides the bit and BER spikes toward $25\text{–}50\%$ (while runs of identical bits like `11` still decode fine!). |
| **"What happens when we move to 5 m or put a book in front of the speaker?"** | Distance drops SNR via $1/r$ attenuation and raises the reverb-to-direct ratio. Blocking line-of-sight (NLOS) attenuates $2{,}000\text{ Hz}$ ($\lambda \approx 17\text{ cm}$) more strongly than $1{,}000\text{ Hz}$ ($\lambda \approx 34\text{ cm}$) due to weaker diffraction and stronger absorption, skewing errors toward `1 -> 0`. |
| **"Why did we zero-pad the FFT in `measure_bit_spectrum`?"** | Zero-padding interpolates the DTFT on a finer frequency grid so we sample closer to the true spectral peak (helpful when $f_1$ isn't on an exact bin), **without** improving true physical resolution (which is still strictly governed by $1/T_b$). |
