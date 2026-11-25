#!/usr/bin/env python3
"""Guided two-laptop experiment session (see docs/experiments.md).

Both laptops walk through the same numbered trial slots, taken from
`data/recorded/manifest_template.json` (3 trials per setting, Slide 14):

    Laptop A (speaker):    python scripts/lab_session.py tx --experiment exp2_tone_separation
    Laptop B (microphone): python scripts/lab_session.py rx --experiment exp2_tone_separation

Per slot: B presses Enter -> 3-2-1 countdown -> "PLAY NOW" -> A presses Enter to play.
B saves the recording, decodes it at once, and adds it to `data/recorded/manifest.json`.
Every decoded trial is kept, including failures (Slide 6); only operator mistakes
(e.g. nothing was played) may be discarded.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_metadata, load_wav, play_audio, record_audio, save_wav
from src.config import DEFAULT_SAMPLE_RATE, RECORDED_DIR
from src.rx import decode_signal

TEMPLATE = RECORDED_DIR / "manifest_template.json"
MANIFEST = RECORDED_DIR / "manifest.json"
EXTRA_RECORD_SECONDS = 3.0   # recording = TX length + 3 s, covers the human reaction delay
COUNTDOWN_SECONDS = 3

SETUP = {
    "exp1_bit_duration": "0.5 m apart, speaker facing mic, quiet room. Only Tb changes (different TX file).",
    "exp2_tone_separation": "0.5 m apart, speaker facing mic, quiet room. Only f1 changes (different TX file).",
    "exp3_background_interference": "0.5 m apart, line of sight. Interference from a phone ~30 cm from Laptop B: "
                                    "Quiet = none; Out-of-band = speech / low music / 400 Hz tone; "
                                    "In-band = 1000-2000 Hz tone, whistling or bright music.",
    "exp4_distance_position": "Distance between speaker and mic as in the setting name. NLOS = Laptop A's speaker "
                              "turned away from Laptop B or blocked by a thick book.",
}


def load_slots(experiment: str | None) -> list[dict]:
    """Trial slots from the manifest template, optionally for one experiment only."""
    slots = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    return [s for s in slots if experiment is None or s["experiment"] == experiment]


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []


def upsert_manifest(entries: list[dict], entry: dict) -> list[dict]:
    """Replace the entry for the same recording file, or append it."""
    return [e for e in entries if e["rx_wav"] != entry["rx_wav"]] + [entry]


def announce(index: int, total: int, slot: dict, last_experiment: str | None) -> None:
    if slot["experiment"] != last_experiment:
        print(f"\n######## {slot['experiment']} (leader: {slot['leader']}) ########")
        print(f"SETUP: {SETUP.get(slot['experiment'], '')}")
    print(f"\n--- Slot {index}/{total}: {slot['factor']} = {slot['setting']}, trial {slot['trial_num']} ---")


def run_tx(slots: list[dict], device: int | str | None, start: int) -> None:
    """Laptop A: play the transmission of each slot when the receiver says GO."""
    last = None
    for i, slot in enumerate(slots, 1):
        if i < start:
            continue
        announce(i, len(slots), slot, last)
        last = slot["experiment"]
        wav = (RECORDED_DIR / slot["tx_meta"]).with_suffix(".wav")
        fs, signal = load_wav(wav)
        while True:
            key = input(f"[Enter] play {wav.name} when Laptop B shows PLAY NOW, [s] skip, [q] quit: ").strip().lower()
            if key == "q":
                return
            if key == "s":
                break
            play_audio(signal, fs, device)
            if input("Played. [Enter] next slot, [r] replay this slot: ").strip().lower() != "r":
                break


def run_rx(slots: list[dict], device: int | str | None, sample_rate: int, redo: bool, start: int) -> None:
    """Laptop B: record each slot, decode it immediately, and log it in the manifest."""
    manifest = load_manifest()
    done = {e["rx_wav"] for e in manifest}
    last = None
    for i, slot in enumerate(slots, 1):
        if i < start or (slot["rx_wav"] in done and not redo):
            continue  # already recorded: rerun with --redo to record it again
        announce(i, len(slots), slot, last)
        last = slot["experiment"]
        meta = load_metadata(RECORDED_DIR / slot["tx_meta"])
        cfg = meta.config
        duration = cfg.lead_silence + len(meta.bits) * cfg.bit_duration + cfg.trail_silence + EXTRA_RECORD_SECONDS

        while True:
            key = input("[Enter] start recording, [s] skip, [q] quit: ").strip().lower()
            if key == "q":
                return
            if key == "s":
                break
            for sec in range(COUNTDOWN_SECONDS, 0, -1):
                print(f"  recording in {sec}...", flush=True)
                time.sleep(1.0)
            print(f"  >>> PLAY NOW <<<   (recording {duration:.1f} s)", flush=True)
            audio = record_audio(duration, sample_rate, device)

            out = RECORDED_DIR / slot["rx_wav"]
            save_wav(out, audio, sample_rate)
            result = decode_signal(audio, sample_rate, len(meta.bits), cfg, expected_bits=meta.bits)
            print(f"  saved {out.relative_to(PROJECT_ROOT)} | peak {abs(audio).max():.2f} | "
                  f"start {result.start_time:.3f} s | errors {result.bit_errors}/{len(meta.bits)} | "
                  f"BER {result.ber:.1%} | margin {result.mean_margin_db:.1f} dB")
            if abs(audio).max() < 0.01:
                print("  WARNING: almost silent. Was the transmission played? Check volume and microphone.")
            elif abs(audio).max() > 0.99:
                print("  WARNING: the recording clipped. Lower Laptop A's volume or the mic input level.")

            answer = input("  [Enter] keep, [d] discard (operator error only!), or type a note to keep with it: ").strip()
            if answer.lower() == "d":
                stamp = datetime.now().strftime("%H%M%S")
                out.rename(out.with_name(f"{out.stem}_discarded_{stamp}.wav"))
                print("  discarded; record this slot again.")
                continue
            manifest = upsert_manifest(manifest, {**slot, "notes": answer})
            MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # saved after every trial
            break
    print(f"\nManifest: {MANIFEST.relative_to(PROJECT_ROOT)} ({len(manifest)} trials). "
          "Next: ./lab.sh analyze  (or python scripts/run_experiment_analysis.py --manifest data/recorded/manifest.json)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guided two-laptop experiment session.")
    parser.add_argument("role", choices=["tx", "rx"], help="tx = Laptop A (speaker), rx = Laptop B (microphone).")
    parser.add_argument("--experiment", help="Only this experiment, e.g. exp2_tone_separation (default: all).")
    parser.add_argument("--device", help="Audio device index or name (see `python -m sounddevice`).")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Recording rate (rx only).")
    parser.add_argument("--start", type=int, default=1, help="Begin at this slot number (same on both laptops).")
    parser.add_argument("--redo", action="store_true", help="rx: also re-record slots already in the manifest.")
    args = parser.parse_args(argv)

    slots = load_slots(args.experiment)
    if not slots:
        parser.error(f"No slots for experiment {args.experiment!r}; see {TEMPLATE.name}.")
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    print(f"Role: {'Laptop A — TRANSMITTER (speaker)' if args.role == 'tx' else 'Laptop B — RECEIVER (microphone)'}"
          f" | {len(slots)} slots")
    if args.role == "tx":
        run_tx(slots, device, args.start)
    else:
        run_rx(slots, device, args.sample_rate, args.redo, args.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
