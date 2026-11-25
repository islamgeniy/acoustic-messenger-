# Running the Experiments (Two Laptops)

How to set up, record, and analyse the controlled experiments with real hardware.
Predictions and the reasoning behind each setting are in [experiment_protocol.md](experiment_protocol.md).

- **Laptop A (TX):** plays the transmitted WAV through its speaker.
- **Laptop B (RX):** records with its microphone, decodes each trial immediately, and keeps the log.

The spec (Slide 14) requires, for each experiment: a prediction, **≥ 3 settings × 3 trials**, changing **one factor** only, and keeping **failed trials** (Slide 6). The tools below enforce the 3 × 3 layout and log every trial.

| Experiment | Leader | Factor (the only thing that changes) | Settings | TX file |
|---|---|---|---|---|
| `exp1_bit_duration` | Shuhratov Abdulloh | Bit duration T_b | 100, 50, 20, 10 ms | different per setting |
| `exp2_tone_separation` | Xojambetov Islam | f1 (f0 = 1000 Hz) | Δf = 1000, 200, 50, 10 Hz | different per setting |
| `exp3_background_interference` | Mirobidov Mirakbar | Background sound | quiet, out-of-band, in-band | baseline |
| `exp4_distance_position` | Malikov Amirxon | Acoustic path | 0.5 m, 2 m, 5 m LOS, 2 m blocked | baseline |

45 trials in total. Plan about **15 minutes per experiment** once the laptops are set up.

---

## 1. One-time setup (both laptops)

```bash
git clone https://github.com/islamgeniy/acoustic-messenger-.git
cd acoustic-messenger-
./lab.sh setup        # creates .venv, installs packages, runs the tests, lists audio devices
```

Windows (no `lab.sh`):

```bat
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m sounddevice
```

`setup` must end with all tests passing. The transmit WAVs are already in `data/transmitted/`, so nothing needs to be generated.

### Check the audio devices

`./lab.sh devices` lists the devices: `>` marks the default input and `<` the default output.

- **Laptop B must record from its built-in microphone.** On a Mac, an iPhone (Continuity), Zoom, or Camo can silently become the default input. Either select the built-in mic in *System Settings → Sound → Input*, or pass its index: `./lab.sh rx --device 0`.
- **Laptop A must play through its built-in speakers**, not Bluetooth headphones. Bluetooth adds delay and changes the sound.

### Microphone permission and processing

- **macOS:** the first recording asks for microphone access for *Terminal* (or your IDE). Allow it, then restart the terminal. If the recording is silent, check *System Settings → Privacy & Security → Microphone*.
- Turn **off** voice processing on Laptop B, because it filters out steady tones:
  - **macOS:** Control Centre → *Mic Mode* → **Standard**, not *Voice Isolation*.
  - **Windows:** *Sound settings → Input device → Audio enhancements* → **Off**.

---

## 2. Fixed conditions (write them down once, keep them for all 45 trials)

| Condition | Value | Why |
|---|---|---|
| Laptop A output volume | **70%** (note the exact value) | Changing it would change two factors at once |
| Laptop B input level | fixed, no auto gain | Same reason |
| Baseline distance | **0.5 m**, speaker facing the mic | Baseline for Exp 1–3 |
| Room | same quiet room, door and window closed | Background noise is Exp 3's factor only |
| Laptops | same two laptops for every trial | Different hardware has a different frequency response |

Measure distances from **Laptop A's speaker grille to Laptop B's microphone hole**. Photograph each setup: it's useful evidence for the presentation.

---

## 3. Running a session

Start the same experiment on both laptops:

```bash
# Laptop A                                        # Laptop B
./lab.sh tx --experiment exp2_tone_separation     ./lab.sh rx --experiment exp2_tone_separation
```

Both screens show the same numbered slots, e.g. `Slot 5/12: Tone Separation (Δf) = 200 Hz, trial 2`. **Always check that both laptops show the same slot number.**

For every slot:

1. **B** presses **Enter**. A `3-2-1` countdown starts, then B shows **`>>> PLAY NOW <<<`** and records.
2. The person at B says **"go"**. **A** presses **Enter** and the transmission plays (1.3–4.2 s).
3. **B** saves `data/recorded/<experiment>/<name>_t<k>.wav`, decodes it at once, and prints:
   ```
   saved data/recorded/exp2_tone_separation/exp2_df200_t2.wav | peak 0.31 | start 1.734 s | errors 0/32 | BER 0.0% | margin 41.2 dB
   ```
4. **B** chooses:
   - **Enter**: keep the trial (the normal case, **also when BER > 0**);
   - **type a note** + Enter: keep it with a note (e.g. `fan was on`, `door slammed at 2 s`);
   - **`d`**: discard. Use this **only for operator mistakes** (A never pressed play, wrong slot). The file is renamed `*_discarded_<time>.wav` and the slot is recorded again.
5. **A** presses **Enter** to move to the next slot (or **`r`** to replay if B asks).

> **Rule: never discard a trial because it decoded badly.** Bad trials are the "difficult or failed case" the spec asks you to explain (Slides 6 and 14).

### Useful options

| Option | Use |
|---|---|
| `--experiment exp3_background_interference` | Run one experiment (omit it to go through all 45 slots) |
| `--start 7` | Continue from slot 7 after a break. Use the same number on both laptops |
| `--device 0` | Choose the audio device by index (see `./lab.sh devices`) |
| `--redo` (rx only) | Also show slots already recorded; a new recording replaces the old one in the log |

The receiver saves the log after **every** trial, so quitting with `q` loses nothing. Restarting `rx` skips slots that are already done.

---

## 4. What to set up for each experiment

The receiver prints a SETUP reminder when an experiment starts. Change **only** the listed factor.

### Exp 1: Bit duration (Shuhratov)
- Baseline position: 0.5 m, line of sight, quiet room.
- Nothing physical changes. Laptop A automatically plays a different TX file per setting (`exp1_tb100ms` … `exp1_tb10ms`).
- Watch: at 10 ms the room echo lasts longer than one bit, so expect errors on bits that follow a tone change.

### Exp 2: Tone separation (Islam)
- Baseline position, quiet room.
- Laptop A plays `exp2_df1000` … `exp2_df10` automatically (f1 = 2000 / 1200 / 1050 / 1010 Hz).
- Watch: at Δf = 10 Hz the two tones fall inside one Hanning main lobe (±20 Hz at T_b = 0.1 s), so expect a failed case.

### Exp 3: Background interference (Mirakbar)
- Baseline position. Interference comes from a **phone about 30 cm from Laptop B**, at the **same phone volume** for every trial of a setting.

| Setting | What the phone plays (start it *before* B presses Enter, keep it running) |
|---|---|
| Quiet baseline | nothing |
| Out-of-band speech/music | a 400 Hz tone, or speech / bass-heavy music (energy below ~800 Hz) |
| In-band tone/music (~1–2 kHz) | a 1000 Hz tone, whistling, or bright music |

- Free tone generators: any "tone generator" app, or a "1000 Hz tone" video.
- Watch: out-of-band noise is removed by the band-pass filter (prediction BER ≈ 0). In-band noise adds energy at f0 or f1, so expect errors.

### Exp 4: Distance and position (Amirxon)
- Quiet room; only the geometry changes.

| Setting | Setup |
|---|---|
| 0.5m LOS | 0.5 m, speaker facing the mic |
| 2.0m LOS | 2.0 m, facing |
| 5.0m LOS | 5.0 m (across the room / corridor), facing |
| 2.0m NLOS (blocked) | 2.0 m, Laptop A's speaker turned away **or** a thick book right in front of it. Use the same blocking for all 3 trials |

- Watch: blocking weakens 2 kHz more than 1 kHz. See `exp4_distance_position_tone_magnitudes.png` after the analysis.

---

## 5. After recording: analyse

```bash
./lab.sh analyze
# = python scripts/run_experiment_analysis.py --manifest data/recorded/manifest.json
```

This decodes every trial in `data/recorded/manifest.json` and writes:

| Output | Content |
|---|---|
| `results/trials.csv` | One row per trial: setting, BER, bit errors, decision margin, passed, notes (**failed trials included**) |
| `results/plots/<exp>/<exp>_ber_margin.png` | BER and decision margin vs. setting, mean ± std and every trial |
| `results/plots/<exp>/<exp>_tone_magnitudes.png` | Received f0 / f1 magnitude per setting |
| `results/plots/<exp>/best_*_*.png` | Waveform, spectrogram, per-bit magnitudes, and bit spectra of the best trial |
| `results/plots/<exp>/difficult_*_*.png` | The same four plots for the worst trial, i.e. your "difficult or failed case" |

### When automatic synchronisation fails

Auto-sync needs a clearly audible signal. If a trial has a strange `start` time or a BER near 50%:

1. Plot it and look at where the tone really starts:
   ```bash
   .venv/bin/python scripts/decode_recording.py data/recorded/exp4_distance_position/5p0m_los_t2.wav \
       --meta data/transmitted/baseline_32bit.json --plot-dir results/plots/check
   ```
2. Read the true start of the first bit from `*_waveform.png` / `*_spectrogram.png`, and test it with `--start-time 1.62`.
3. Put that value into the trial's `"start_time"` in `data/recorded/manifest.json` and add a note such as `"manual start: auto-sync locked late"`. Manual start is allowed (Slide 6). Then run `./lab.sh analyze` again.

Report how many trials needed a manual start; that is itself a result.

---

## 6. Saving the results

On Laptop B, after each experiment:

```bash
git add data/recorded/ results/
git commit -m "data(exp2): tone separation recordings, 12 trials"
git push
```

All recordings together are about 30 MB, which is fine for the repo. The recordings, `manifest.json`, `trials.csv`, and the plots are the "input/output audio, labelled plots and trial records" required for the final assessment (Slide 18).

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `WARNING: almost silent` | A didn't play, A's volume is muted, or B records from the wrong device (`./lab.sh devices`, then `--device`). Discard with `d` and redo |
| `WARNING: the recording clipped` | Lower A's volume (for **all** trials, then restart that experiment) or move the laptops apart |
| BER ≈ 50% in the quiet baseline | Usually sync. Follow §5 "When automatic synchronisation fails" |
| Recording ends before the tone | A pressed play too late. Discard (`d`) and redo; recording length is TX length + 3 s |
| `PortAudioError` / invalid sample rate | The mic doesn't support 48 kHz: `--sample-rate 44100` (the receiver handles any rate) |
| macOS records silence | Microphone permission for Terminal (§1), then restart the terminal |
| Both screens show different slots | Quit both with `q`, restart both with the same `--start N` |
