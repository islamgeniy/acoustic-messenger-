#!/usr/bin/env python3
"""Live microphone recording helper for Laptop B (experiments & final live demo).

Examples:
    # Record 6 seconds to a file and immediately decode against baseline metadata
    python scripts/record_trial.py -o data/recorded/baseline/trial1.wav \
        --duration 6.0 --decode-meta data/transmitted/baseline_32bit.json

    # Record without decoding (batch-decode later with run_experiment_analysis.py)
    python scripts/record_trial.py -o data/recorded/exp4_distance_position/2m_nlos_t1.wav --duration 6.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_metadata, record_audio, save_wav
from src.config import DEFAULT_SAMPLE_RATE
from src.rx import decode_transmission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record mono audio from the default microphone to a WAV file."
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output recorded WAV path.")
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Recording duration in seconds (default: 6.0 s; covers 4.2 s 32-bit TX + buffer).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Recording sample rate in Hz (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds to count down before recording starts (default: 3).",
    )
    parser.add_argument(
        "--decode-meta",
        type=Path,
        default=None,
        help="Optional TX metadata JSON to auto-decode immediately after recording.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional manual start time when using --decode-meta.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for sec in range(args.countdown, 0, -1):
        print(f"Starting recording in {sec}...", flush=True)
        time.sleep(1.0)

    print(f"RECORDING for {args.duration:.1f} s at {args.sample_rate} Hz (play Laptop A now!)...", flush=True)
    audio = record_audio(args.duration, args.sample_rate)
    out_wav = save_wav(args.output, audio, args.sample_rate)
    print(f"Saved recording -> {out_wav}")

    if args.decode_meta is not None:
        meta = load_metadata(args.decode_meta)
        result = decode_transmission(audio, args.sample_rate, meta, args.start_time)
        print(f"Decoded bits : {result.decoded_bits}")
        print(f"Expected     : {meta.bits}")
        print(result.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
