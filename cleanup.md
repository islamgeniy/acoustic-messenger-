# Cleanup Plan: Overbuilt Features & Buckets

> Baseline: commit `643bb6c` · 3,199 lines of Python · 24/24 tests passing.
> Goal: keep every **required** feature (spec slides 3, 6, 13, 14, 18), remove what is optional and hard to defend in the solo oral Q&A, and leave each module small enough that its owner can explain every line.
> Rule for each bucket: tests stay green, and behaviour on the required path (TX → WAV → RX → BER → plots) does not get worse.

---

## 1. Overbuilt features found

| # | Feature | Where | Why it goes / changes | Decision |
|---|---|---|---|---|
| 1 | **Preamble + matched-filter sync** | `tx.py`, `rx.py`, `config.py`, both CLIs | Optional per spec ("automatic synchronisation is optional"). **Buggy:** the periodic `10101010` locks 2 bits late in 15/30 noisy runs. Its logic was spread across 5 files. | **Remove** |
| 2 | Energy-onset sync heuristic | `rx.py` | 55 lines, 7 tuned magic numbers, a run-length loop and a refinement pass. | **Simplify** to one rule: "first frame whose in-band RMS reaches half the peak" |
| 3 | "SNR" estimator | `rx.py` | Hardcoded 300–4500 Hz noise band, harmonic exclusions, median trick. Hard to defend and not required. | **Replace** with the *decision margin* `20·log10(winner/loser)`: the same two magnitudes Slide 5 compares |
| 4 | Window / band / guard knobs | `rx.py`, CLI, manifest, plotting | `hann\|rect`, `band_hz`, `guard_fraction` threaded through 6 call sites. | **Fix as constants** (Hann, 5% guard) |
| 5 | Synthetic channel simulator + `--demo-sim` | `experiments.py` (~130 lines) | Generates *fake* experiment results with invented settings. That is a risk if confused with real data, and not required. | **Remove** (tests build degraded signals inline) |
| 6 | Speculative / duplicated contracts | `config.py` | `recording_path`, `snr_db`, preamble fields, redundant `to_dict`, hand-written `from_dict`, CSV field list duplicated in `experiments.py`. | **Trim** |
| 7 | `phase_continuous` flag | `tx.py` | The `False` path is never used. | **Remove**, vectorise the phase |
| 8 | Two RNG backends, `generate_random_bits`, `--random-bits/--seed` | `tx.py`, CLI | Crypto RNG for test bits. The demo uses instructor-given bits. | **Remove** |
| 9 | Duplicate validation | `tx.py` | Bits validated 2×, preamble 2×. | Preamble gone. Bits: **kept** one cheap check per public entry point (`encode_to_wav`, `modulate_fsk`) |
| 10 | Three DFT implementations | `dft.py` | Spec asks for *one* direct DFT. `direct_dft_matrix` and `direct_idft` are used only by tests. | **Keep only `direct_dft`** |
| 11 | 165-line single plot function, hardcoded annotation text | `run_dft_validation.py` | Annotations would lie if the parameters changed. | **One function per figure**, computed labels |
| 12 | CLI flag overload | `encode_message.py` (14 flags), `decode_recording.py` (11) | Defaults re-declared by hand. Flags no experiment uses. | **Trim** to what the experiments and demo need |
| 13 | Two scripts doing "decode one WAV" | `decode_recording.py` and `run_experiment_analysis.py --single` | Overlapping modes. | **Merge**: `decode_recording.py --plot-dir` |
| 14 | Manifest template built by 4 copy-pasted blocks | `prepare_experiment_audio.py` | 497-line JSON from repetitive code. | **Data-driven table** |
| 15 | Keyword-triggered Exp 4 plot | `experiments.py` (`"distance" in name`) | String matching on experiment names. | **Make generic**: tone magnitudes per setting, for every experiment |
| 16 | Silent signal processing in I/O | `audio_io.py` | `save_wav` silently rescales. `load_wav` silently removes DC. try/except around a hard dependency, duplicated. | **Explicit**: raise on clipping, no hidden DC step |
| 17 | `assert` used for runtime checks | 5 places in CLIs and modules | Type-checker appeasement. | **Remove** |
| 18 | `sys.path` hack in every script | 6 scripts | Duplicated boilerplate. | Keep for now (needed for `python scripts/x.py`); revisit later |

---

## 2. Buckets (cleaning order)

| Bucket | Files | Owner | Status |
|---|---|---|---|
| **B1 DFT** | `src/dft.py`, `scripts/run_dft_validation.py`, `tests/test_dft.py` | Shuhratov | ✅ |
| **B2 Transmitter + preamble removal** | `src/tx.py`, `scripts/encode_message.py`, `tests/test_tx.py` (+ preamble bits of `config.py`, `rx.py`) | Islam | ✅ |
| **B3 Receiver** | `src/rx.py`, `scripts/decode_recording.py`, `scripts/record_trial.py`, `tests/test_loopback.py` | Mirakbar | ✅ |
| **B4 Experiments & plots** | `src/experiments.py`, `src/plotting.py`, `scripts/run_experiment_analysis.py`, `scripts/prepare_experiment_audio.py`, `tests/test_eval_plots.py`, `data/recorded/manifest_template.json` | Amirxon | ✅ |
| **B5 Shared core** | `src/config.py`, `src/audio_io.py` | shared | ✅ |
| **B6 Docs & repo hygiene** | `README.md`, `CONTRIBUTING.md`, `docs/*.md`, `.gitkeep` files | shared | ✅ |

Slide-based wording for comments and docstrings is deferred until the lecture PDFs are reviewed (next phase).

---

## 3. Progress log

**Result: 3,199 → 2,013 lines of Python (−37%). 23/23 tests pass.**

| File | Before | After |
|---|---:|---:|
| `src/rx.py` | 337 | 181 |
| `src/experiments.py` | 412 | 134 |
| `src/plotting.py` | 374 | 225 |
| `src/tx.py` | 218 | 136 |
| `src/dft.py` | 156 | 124 |
| `src/config.py` | 158 | 141 |
| `src/audio_io.py` | 117 | 78 |
| `scripts/*` (6 files) | 961 | 598 |
| `tests/*` (4 files) | 465 | 395 |

### B1 DFT ✅
- Removed `direct_dft_matrix` and `direct_idft` (only `direct_dft` is required) and their tests.
- `run_dft_validation.py`: one function per figure. Annotation text is now computed from the spectrum, not hardcoded. Same 4 output files.

### B2 Transmitter ✅
- Preamble removed everywhere (tx, config, rx, CLI, tests).
- `modulate_fsk` is vectorised: `phase[n] = 2π/fs · Σ_{k<n} f[k]`. **Verified identical to the old output (≤ 1e-10)**, and all 9 regenerated WAVs are byte-identical.
- Removed `phase_continuous`, `generate_random_bits` (+ crypto RNG), `_` in bitstrings, `assert`.
- `encode_message.py`: 14 → 8 flags, defaults read from `ModulationConfig()`.

### B3 Receiver ✅
- **Sync:** I tried 3 simpler "single threshold" detectors. All were much worse (they lock a whole bit late when the first tone is weak, e.g. from room echo). So the proven 2-stage algorithm was **kept** and rewritten: named constants each with a one-line reason, a vectorised "sustained" check instead of a loop, no hop bookkeeping.
- **Bug fixed (pre-existing):** auto-sync failed whenever silence was < ~20% of the recording (the noise estimate used the 20th percentile). Now uses the 5th percentile. With 0.1 s silence it locks to 0.3 ms (old: 0.43 s off). Regression test added.
- "SNR" estimator → **decision margin** `20·log10(stronger/weaker)` (computed property, Slide 5 magnitudes).
- Window / band / guard knobs → module constants. `compute_ber` now rejects length mismatches (BER can no longer exceed 1).
- New small helpers `bit_segment`, `bit_spectrum`, `tone_magnitudes`, reused by the plots (no duplicated slicing).
- `decode_recording.py`: removed `--window/--guard/--save-json`, added `--plot-dir`.
- Benchmark (16 simulated room conditions × 5 seeds, auto-sync): start error unchanged (≤ 0.3 ms) in all normal conditions. Trials with errors went 21 → 19 / 80 (only the deliberately terrible "far + blocked" case fails, as before).

### B4 Experiments & plots ✅
- Removed the synthetic channel simulator and `--demo-sim` fake-results preview.
- Removed `run_experiment_analysis.py --single` (merged into `decode_recording.py --plot-dir`).
- Keyword-triggered Exp 4 chart → generic `plot_tone_magnitudes` for every experiment.
- CSV columns come from `TrialRecord` itself (no duplicated field list). Manifest paths are always relative to the manifest file.
- `prepare_experiment_audio.py`: 4 copy-pasted blocks → 2 data tables. The generated manifest is byte-identical.
- Output names changed: `*_ber_snr.png` → `*_ber_margin.png`, `*_freq_attenuation.png` → `*_tone_magnitudes.png`.

### B5 Shared core ✅
- `config.py`: removed preamble, `snr_db`, `mean_snr_db`, `recording_path`, redundant `to_dict` lines. `from_dict` is one line and still ignores keys from old JSON files.
- `audio_io.py`: `save_wav` **raises** instead of silently rescaling. The hidden DC removal was removed (verified: DC offsets up to 0.3 change nothing, because band-pass + Hann reject DC). No try/except around `sounddevice`.

### B6 Docs & hygiene ✅
- README, CONTRIBUTING, and all 3 docs updated for the removed features and renamed outputs.
- Removed 5 redundant `.gitkeep` files and the stale `results/sim_preview/` ignore rule.

### Known limitations (not changed)
- Auto-sync needs roughly 14 dB in-band SNR. For very weak or far recordings, pass `--start-time` (manual start is allowed by Slide 6).
- On a perfectly clean (synthetic) signal the margin is huge (~120 dB) because the losing tone is ≈ 0. Real recordings sit far lower.
- `sys.path` boilerplate in scripts (item 18) is still there.
- Comments and docstrings still need aligning with the lecture slides (next phase).
