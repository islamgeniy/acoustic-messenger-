# Signals and Systems — Acoustic Messenger (Option A)

- **Course:** IUT — Signals and Systems (Fall 2026), Prof. Khodjaev
- **Team:** DFT Squad
- **Project:** Option A — Acoustic Messenger (2-FSK transmission over speaker, air, and microphone)

## Team Assignments (Randomly Assigned)

| Member | Feature Branch | Code Ownership | Controlled Experiment |
| :--- | :--- | :--- | :--- |
| **Shuhratov Abdulloh** | `dft-validation` | `src/dft.py`, `scripts/run_dft_validation.py`, `tests/test_dft.py` | **Exp 1:** Bit Duration ($T_b$) |
| **Xojambetov Islam** | `tx-encoder` | `src/tx.py`, `scripts/encode_message.py` | **Exp 2:** Tone Separation ($\Delta f$) |
| **Mirobidov Mirakbar** | `rx-decoder` | `src/rx.py`, `scripts/decode_recording.py`, `tests/test_loopback.py` | **Exp 3:** Background Interference |
| **Malikov Amirxon** | `eval-plots` | `src/plotting.py`, `src/experiments.py`, `scripts/run_experiment_analysis.py` | **Exp 4:** Distance & Position (Acoustic Path) |

## Quick Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Layout

```text
acoustic-messenger-/
├── src/
│   ├── config.py          # Shared parameters & dataclasses
│   ├── audio_io.py        # WAV + JSON metadata I/O & optional live audio
│   ├── dft.py             # Direct DFT implementation & validation (branch: dft-validation)
│   ├── tx.py              # 2-FSK acoustic transmitter (branch: tx-encoder)
│   ├── rx.py              # 2-FSK acoustic receiver & BER (branch: rx-decoder)
│   ├── experiments.py     # Controlled experiment runner & CSV logger (branch: eval-plots)
│   └── plotting.py        # Waveforms, labelled spectra & BER plots (branch: eval-plots)
├── scripts/               # Runnable CLI entry points
├── tests/                 # Automated pytest unit & loopback tests
├── data/
│   ├── transmitted/       # Generated TX WAVs + metadata JSONs
│   └── recorded/          # Real microphone recordings by experiment
├── results/               # Generated labelled plots & master trials.csv
└── docs/                  # Experiment protocol, AI disclosure & Q&A guide
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch/PR workflow and shared interface contracts.
