# Signals and Systems — Acoustic Messenger (Option A)

- **Course:** IUT — Signals and Systems (Fall 2026), Prof. Khodjaev
- **Team:** DFT Squad
- **Project:** Option A — Acoustic Messenger (2-FSK transmission over speaker, air, and microphone)

## Team Assignments (Randomly Assigned)

| Member | Feature Branch | Code Ownership | Controlled Experiment |
| :--- | :--- | :--- | :--- |
| **Shuhratov Abdulloh** | `dft-validation` | `src/dft.py`, `scripts/run_dft_validation.py`, `tests/test_dft.py` | **Exp 1:** Bit Duration ($T_b$) |
| **Xojambetov Islam** | `tx-encoder` | `src/tx.py`, `scripts/encode_message.py`, `tests/test_tx.py` | **Exp 2:** Tone Separation ($\Delta f$) |
| **Mirobidov Mirakbar** | `rx-decoder` | `src/rx.py`, `src/sync.py`, `scripts/decode_recording.py`, `tests/test_loopback.py` | **Exp 3:** Background Interference |
| **Malikov Amirxon** | `eval-plots` | `src/plotting.py`, `src/experiments.py`, `scripts/run_experiment_analysis.py`, `tests/test_eval_plots.py` | **Exp 4:** Distance & Position (Acoustic Path) |

---

## 1. Setup & Automated Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all unit & software loopback tests
pytest -v
```

---

## 2. Quick Run Commands

### Direct DFT Validation & Fourier Plots (Slide 13)
```bash
python scripts/run_dft_validation.py
# Outputs 5 labelled figures in results/plots/dft/ (incl. the lecture's DFT, leakage and window examples)
```

### Transmit a Message (Laptop A)
```bash
# Default 32-bit baseline WAV + JSON metadata
python scripts/encode_message.py -o data/transmitted/baseline_32bit.wav

# Custom 32+ bit sequence with immediate speaker playback (Live Demo!)
python scripts/encode_message.py --bits 10110010111001010110100110010110 --play

# Encode ASCII text (>= 4 chars = >= 32 bits)
python scripts/encode_message.py --text "IUT!" -o data/transmitted/iut_32bit.wav --play
```

### Receive & Decode a Recording (Laptop B)
```bash
# Record live on Laptop B and auto-decode immediately (Live Demo!)
python scripts/record_trial.py -o data/recorded/baseline/live_demo.wav \
    --duration 6.0 --decode-meta data/transmitted/baseline_32bit.json

# Decode an existing recorded WAV (auto-sync)
python scripts/decode_recording.py data/recorded/baseline/live_demo.wav \
    --meta data/transmitted/baseline_32bit.json

# Decode with manual start time (in seconds)
python scripts/decode_recording.py data/recorded/baseline/live_demo.wav \
    --meta data/transmitted/baseline_32bit.json --start-time 0.50
```

### Controlled Experiments & Labelled Plots (Slide 14)
Full two-laptop guide: [docs/experiments.md](docs/experiments.md).
```bash
# Guided sessions: same experiment on both laptops, numbered trial slots
./lab.sh setup                                    # both laptops, once
./lab.sh tx --experiment exp2_tone_separation     # Laptop A (speaker)
./lab.sh rx --experiment exp2_tone_separation     # Laptop B (mic): record + decode + log each trial
./lab.sh analyze                                  # results/trials.csv + results/plots/

# 1. (Re)generate all experiment TX WAV/JSON files + manifest template
python scripts/prepare_experiment_audio.py

# 2. Decode one recording and save its diagnostic plots
python scripts/decode_recording.py data/recorded/baseline/live_demo.wav \
    --meta data/transmitted/baseline_32bit.json --plot-dir results/plots/baseline

# 3. Batch-evaluate all recorded trials -> results/trials.csv + results/plots/
python scripts/run_experiment_analysis.py --manifest data/recorded/manifest.json
```

---

## 3. Live Demo Workflow (Final Assessment)

When the instructor supplies a brand-new 32-bit sequence during the demo:
1. **Laptop A (Transmitter):**
   ```bash
   python scripts/encode_message.py --bits <INSTRUCTOR_BITS> -o data/transmitted/live_demo.wav --play
   ```
2. **Laptop B (Receiver):**
   ```bash
   python scripts/record_trial.py -o data/recorded/baseline/live_demo.wav --duration 6.0 \
       --decode-meta data/transmitted/live_demo.json
   ```
   (Or run `decode_recording.py` with `--expected-bits <INSTRUCTOR_BITS>` and `--num-bits 32`.)

---

## 4. Documentation & Assessment Materials

- [docs/experiment_protocol.md](docs/experiment_protocol.md) — Pre-experiment predictions, fixed vs. varied parameters, and recording checklist for all 4 experiments.
- [docs/contribution_and_ai_disclosure.md](docs/contribution_and_ai_disclosure.md) — Individual contribution summary and AI-use disclosure (Slides 15 & 18).
- [docs/experiments.md](docs/experiments.md) — How to set up both laptops, record all trials, and analyse the results.
- [docs/lecture_map.md](docs/lecture_map.md) — Which lecture concept lives where in the code, with tests and plots as evidence.
- [docs/qa_study_guide.md](docs/qa_study_guide.md) — Complete signal path, Fourier concepts, and prediction/diagnosis prep for the 5–7 minute individual oral defense.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Branch/PR workflow and shared interface contracts.
