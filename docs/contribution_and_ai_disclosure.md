# Team Contribution & AI-Use Disclosure Statement

**Course:** IUT — Signals and Systems (Fall 2026), Prof. Khodjaev  
**Team:** DFT Squad  
**Project:** Option A — Acoustic Messenger

---

## 1. Individual Contributions

Branches and controlled experiments were randomly assigned across all four members so every member owns both a software subsystem (via an independent feature branch) and a controlled acoustic experiment:

| Member | Branch | Software Ownership | Controlled Experiment Led |
| :--- | :--- | :--- | :--- |
| **Shuhratov Abdulloh** | `dft-validation` | Direct $O(N^2)$ DFT (`src/dft.py`), validation script (`scripts/run_dft_validation.py`), DFT unit tests (`tests/test_dft.py`), and Fourier diagnostic figures. | **Exp 1: Bit Duration ($T_b$)** — testing $\Delta f = 1/T_b$ and room reverberation limits. |
| **Xojambetov Islam** | `tx-encoder` | Continuous-phase 2-FSK transmitter (`src/tx.py`), transmitter CLI (`scripts/encode_message.py`), TX unit tests (`tests/test_tx.py`), repository integration & PR review. | **Exp 2: Tone Separation ($\Delta f = f_1 - f_0$)** — testing frequency resolution and spectral leakage. |
| **Mirobidov Mirakbar** | `rx-decoder` | 2-FSK receiver (`src/rx.py`), manual & automatic sync (energy onset + phase-invariant preamble), SNR/BER computation, receiver CLI (`scripts/decode_recording.py`), loopback tests (`tests/test_loopback.py`). | **Exp 3: Background Interference** — comparing out-of-band vs. in-band acoustic interference. |
| **Malikov Amirxon** | `eval-plots` | Plotting suite (`src/plotting.py`), experiment runner & CSV logger (`src/experiments.py`, `scripts/run_experiment_analysis.py`), evaluation tests (`tests/test_eval_plots.py`). | **Exp 4: Distance & Position** — acoustic attenuation, NLOS frequency selectivity, and room reverb analysis. |

---

## 2. AI-Use Disclosure (Per Course Policy, Slides 15 & 18)

In accordance with the course guidelines (*"AI assistance is allowed. Verify the code and claims. Cite sources and disclose AI use"*):

- **Tools Used:** AI coding assistant (Antigravity / Gemini) was used during development for project scaffolding, Python implementation assistance, pytest test writing, and plotting boilerplate.
- **Verification & Ownership:**
  - All mathematical formulas (exponential DFT equation, continuous-phase 2-FSK synthesis, FFT bin mapping $f_k = k f_s / N$, and BER computation) were independently verified by the team against course lectures and automated unit tests (`pytest`: 24/24 passing, direct DFT matching `numpy.fft.fft` to $<10^{-10}$).
  - All physical two-laptop acoustic recordings, experimental measurements, trial logs, and failure diagnoses are conducted and analyzed by the team using real hardware.
  - During the final assessment and individual oral Q&A, all team members answer independently without AI or teammate assistance.

## 3. Technical References

- Course slides: Prof. Khodjaev, *Signals and Systems — Term Project Announcement (Fall 2026)*.
- Oppenheim, A. V., Willsky, A. S., & Nawab, S. H., *Signals and Systems* (2nd ed.), Prentice Hall.
- NumPy (`numpy.fft`) & SciPy (`scipy.signal`, `scipy.io.wavfile`) official documentation.
