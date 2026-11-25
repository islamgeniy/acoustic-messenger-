# Contributing & PR Workflow (DFT Squad)

## Branch Ownership (Randomly Assigned)

Each teammate owns one independent feature branch and opens a Pull Request into `main` for review and merge:

1. **`dft-validation` — Shuhratov Abdulloh**
   - Files: `src/dft.py`, `scripts/run_dft_validation.py`, `tests/test_dft.py`
   - Experiment: **Exp 1 — Bit Duration ($T_b$)**
2. **`tx-encoder` — Xojambetov Islam**
   - Files: `src/tx.py`, `scripts/encode_message.py`
   - Experiment: **Exp 2 — Tone Separation ($\Delta f = f_1 - f_0$)**
3. **`rx-decoder` — Mirobidov Mirakbar**
   - Files: `src/rx.py`, `scripts/decode_recording.py`, `tests/test_loopback.py`
   - Experiment: **Exp 3 — Background Interference**
4. **`eval-plots` — Malikov Amirxon**
   - Files: `src/plotting.py`, `src/experiments.py`, `scripts/run_experiment_analysis.py`
   - Experiment: **Exp 4 — Distance & Position (Acoustic Path)**

## Git Workflow

```bash
# 1. Start from the latest main on your assigned branch
git checkout <your-branch>
git pull origin main

# 2. Implement only the files assigned to your branch
# 3. Run tests before pushing
pytest

# 4. Commit and push your branch
git add <your-files>
git commit -m "<type>: <short description>"
git push -u origin <your-branch>

# 5. Open a Pull Request on GitHub targeting `main`
```

## Shared Interfaces (`src/config.py` & `src/audio_io.py`)

To avoid merge conflicts, **do not modify `src/config.py` or `src/audio_io.py` inside feature branches** unless coordinated on `main`. Use the shared dataclasses:

- `ModulationConfig` — `sample_rate`, `f0`, `f1`, `bit_duration`, `amplitude`, `lead_silence`, `trail_silence`
- `TransmissionMetadata` — saved as `<wav_stem>.json` alongside every transmitted WAV
- `BitDecision` & `DecodeResult` — returned by `src/rx.py` and consumed by `src/plotting.py` and `src/experiments.py`
- `TrialRecord` — row schema for `results/trials.csv`
