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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_metadata, load_wav, play_audio, record_audio, save_wav
from src.config import DEFAULT_SAMPLE_RATE, RECORDED_DIR, TransmissionMetadata
from src.rx import decode_transmission

TEMPLATE = RECORDED_DIR / "manifest_template.json"
MANIFEST = RECORDED_DIR / "manifest.json"
EXTRA_RECORD_SECONDS = 3.0   # recording = TX length + 3 s, covers the human reaction delay
COUNTDOWN_SECONDS = 3
MIN_CONFIDENT_MARGIN_DB = 10.0  # below this the tones are barely above the background

SETUP = {
    "exp1_bit_duration": "0.5 m apart, speaker facing mic, quiet room. Only Tb changes (different TX file).",
    "exp2_tone_separation": "0.5 m apart, speaker facing mic, quiet room. Only f1 changes (different TX file).",
    "exp3_background_interference": "0.5 m apart, line of sight. Interference from a phone ~30 cm from Laptop B: "
                                    "Quiet = none; Out-of-band = speech / low music / 400 Hz tone; "
                                    "In-band = 1000-2000 Hz tone, whistling or bright music.",
    "exp4_distance_position": "Distance between speaker and mic as in the setting name. NLOS = Laptop A's speaker "
                              "turned away from Laptop B or blocked by a thick book.",
}


class QuitSession(Exception):
    """The operator typed q."""


def load_slots(experiment: str | None) -> list[dict]:
    """Trial slots from the manifest template, optionally for one experiment only."""
    slots = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    return [s for s in slots if experiment is None or s["experiment"] == experiment]


def select_slots(slots: list[dict], start: int, already_recorded: set[str]) -> list[tuple[int, dict]]:
    """Number the slots 1..N (same numbers on both laptops) and keep those still to do."""
    return [(n, s) for n, s in enumerate(slots, 1) if n >= start and s["rx_wav"] not in already_recorded]


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []


def save_manifest(entries: list[dict]) -> None:
    MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def upsert_manifest(entries: list[dict], entry: dict) -> list[dict]:
    """Replace the entry for the same recording file, or append it."""
    return [e for e in entries if e["rx_wav"] != entry["rx_wav"]] + [entry]


def ask(prompt: str) -> str:
    """Read the operator's answer; `q` ends the session from any prompt."""
    answer = input(prompt).strip()
    if answer.lower() == "q":
        raise QuitSession
    return answer


def announced(selected: list[tuple[int, dict]], total: int):
    """Yield each slot after printing its header (and the setup reminder when the experiment changes)."""
    current_experiment = None
    for number, slot in selected:
        if slot["experiment"] != current_experiment:
            current_experiment = slot["experiment"]
            print(f"\n######## {current_experiment} (leader: {slot['leader']}) ########")
            print(f"SETUP: {SETUP.get(current_experiment, '')}")
        print(f"\n--- Slot {number}/{total}: {slot['factor']} = {slot['setting']}, trial {slot['trial_num']} ---")
        yield slot


def run_tx(selected: list[tuple[int, dict]], total: int, device: int | str | None) -> None:
    """Laptop A: play the transmission of each slot when the receiver says GO."""
    for slot in announced(selected, total):
        wav = (RECORDED_DIR / slot["tx_meta"]).with_suffix(".wav")
        sample_rate, signal = load_wav(wav)
        while ask(f"[Enter] play {wav.name} when Laptop B shows PLAY NOW, [s] skip, [q] quit: ").lower() != "s":
            play_audio(signal, sample_rate, device)
            if input("Played. [Enter] next slot, [r] replay this slot: ").strip().lower() != "r":
                break


def run_rx(selected: list[tuple[int, dict]], total: int, device: int | str | None, sample_rate: int) -> None:
    """Laptop B: record each slot until the operator keeps a take, and log it in the manifest."""
    manifest = load_manifest()
    for slot in announced(selected, total):
        while ask("[Enter] start recording, [s] skip, [q] quit: ").lower() != "s":
            recording = record_trial(slot, device, sample_rate)
            note = input("  [Enter] keep, [d] discard (operator error only!), or type a note to keep with it: ").strip()
            if note.lower() == "d":
                recording.rename(recording.with_name(f"{recording.stem}_discarded_{datetime.now():%H%M%S}.wav"))
                print("  discarded; record this slot again.")
                continue
            manifest = upsert_manifest(manifest, {**slot, "notes": note})
            save_manifest(manifest)  # after every trial, so quitting loses nothing
            break


def record_trial(slot: dict, device: int | str | None, sample_rate: int) -> Path:
    """Count down, record, save, decode at once, and print the result with any warnings."""
    meta = load_metadata(RECORDED_DIR / slot["tx_meta"])
    duration = recording_seconds(meta)
    for second in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"  recording in {second}...", flush=True)
        time.sleep(1.0)
    print(f"  >>> PLAY NOW <<<   (recording {duration:.1f} s)", flush=True)
    audio = record_audio(duration, sample_rate, device)

    path = save_wav(RECORDED_DIR / slot["rx_wav"], audio, sample_rate)
    result = decode_transmission(audio, sample_rate, meta)
    print(f"  saved {path.relative_to(PROJECT_ROOT)} | peak {np.max(np.abs(audio)):.2f} | {result.summary()}")
    for warning in recording_warnings(np.max(np.abs(audio)), result.mean_margin_db):
        print(f"  WARNING: {warning}")
    return path


def recording_seconds(meta: TransmissionMetadata) -> float:
    """Length of the transmission plus `EXTRA_RECORD_SECONDS` for the operators' reaction time."""
    cfg = meta.config
    return cfg.lead_silence + len(meta.bits) * cfg.bit_duration + cfg.trail_silence + EXTRA_RECORD_SECONDS


def recording_warnings(peak: float, mean_margin_db: float) -> list[str]:
    """Problems with the take itself (as opposed to a genuinely hard setting)."""
    if peak < 0.01:
        return ["almost silent. Was the transmission played? Check volume and the input device (./lab.sh devices)."]
    warnings = []
    if peak > 0.99:
        warnings.append("the recording clipped. Lower Laptop A's volume or the mic input level.")
    if mean_margin_db < MIN_CONFIDENT_MARGIN_DB:
        warnings.append("the tones are barely above the background. If the transmission was clearly audible, "
                        "check Laptop B's Mic Mode (Voice Isolation removes steady tones) and its input device.")
    return warnings


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
    already_recorded = set() if args.role == "tx" or args.redo else {e["rx_wav"] for e in load_manifest()}
    selected = select_slots(slots, args.start, already_recorded)

    is_tx = args.role == "tx"
    print(f"Role: {'Laptop A — TRANSMITTER (speaker)' if is_tx else 'Laptop B — RECEIVER (microphone)'}"
          f" | {len(selected)} of {len(slots)} slots to do")
    try:
        if is_tx:
            run_tx(selected, len(slots), device)
        else:
            run_rx(selected, len(slots), device, args.sample_rate)
    except QuitSession:
        print("Session stopped.")
    if not is_tx:
        print(f"\nManifest: {MANIFEST.relative_to(PROJECT_ROOT)}. Next: ./lab.sh analyze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
