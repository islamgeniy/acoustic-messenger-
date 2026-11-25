#!/usr/bin/env python3
"""CLI for decoding a recorded 2-FSK acoustic WAV file and reporting BER.

Examples:
    # Auto-sync decode using the companion TX metadata JSON
    python scripts/decode_recording.py data/recorded/baseline/trial1.wav \
        --meta data/transmitted/baseline_32bit.json

    # Manual start time (Slide 6: manual start selection is allowed) + diagnostic plots
    python scripts/decode_recording.py data/recorded/baseline/trial1.wav \
        --meta data/transmitted/baseline_32bit.json --start-time 0.50 --plot-dir results/plots/baseline

    # Decode without metadata JSON (provide parameters directly)
    python scripts/decode_recording.py recording.wav --num-bits 32 --f0 1000 --f1 2000 --bit-duration 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_metadata
from src.config import ModulationConfig
from src.plotting import plot_trial
from src.rx import decode_wav
from src.tx import bits_to_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode a recorded 2-FSK acoustic WAV file and compute BER.")
    parser.add_argument("recording", type=Path, help="Path to the recorded WAV file.")
    parser.add_argument("--meta", type=Path, help="TX metadata JSON (provides f0, f1, Tb and the expected bits).")
    parser.add_argument("--expected-bits", help="Expected bitstring for BER (overrides the bits in --meta).")
    parser.add_argument("--num-bits", type=int, help="Bits to decode (default: length of the expected bits).")
    parser.add_argument("--f0", type=float, help="Bit '0' frequency in Hz (overrides --meta).")
    parser.add_argument("--f1", type=float, help="Bit '1' frequency in Hz (overrides --meta).")
    parser.add_argument("--bit-duration", type=float, help="Bit duration Tb in seconds (overrides --meta).")
    parser.add_argument("--start-time", type=float, help="Manual start time in seconds (default: automatic).")
    parser.add_argument("--as-text", action="store_true", help="Also print the decoded bits as 8-bit text.")
    parser.add_argument("--plot-dir", type=Path, help="Save waveform, spectrogram and bit-spectrum plots here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg, expected_bits = ModulationConfig(), args.expected_bits
    if args.meta is not None:
        meta = load_metadata(args.meta)
        cfg = meta.config
        expected_bits = expected_bits or meta.bits
    cfg.f0 = args.f0 or cfg.f0
    cfg.f1 = args.f1 or cfg.f1
    cfg.bit_duration = args.bit_duration or cfg.bit_duration

    num_bits = args.num_bits or (len(expected_bits) if expected_bits else None)
    if num_bits is None:
        parser.error("Specify --num-bits, --expected-bits, or --meta.")

    result, signal, sample_rate = decode_wav(
        args.recording, num_bits, cfg, start_time=args.start_time, expected_bits=expected_bits
    )

    print("=== 2-FSK Acoustic Receiver ===")
    print(f"Recording       : {args.recording}")
    print(f"Sync method     : {result.sync_method} (start = {result.start_time:.4f} s)")
    print(f"Frequencies     : 0 -> {cfg.f0:.1f} Hz | 1 -> {cfg.f1:.1f} Hz | Tb = {cfg.bit_duration:.4f} s")
    print()
    print(f"{'Bit':>3} | {'Interval (s)':>13} | {'Near f0':>9} | {'Near f1':>9} | {'Dec':>3} | {'Exp':>3} | {'OK':>3} | {'Margin dB':>9}")
    print("-" * 74)
    for d in result.bit_decisions:
        ok = "-" if d.is_correct is None else ("Y" if d.is_correct else "N")
        print(
            f"{d.index + 1:3d} | {d.start_time:6.3f}-{d.end_time:6.3f} | {d.mag_f0:9.4f} | {d.mag_f1:9.4f} | "
            f"{d.decision:>3} | {d.expected or '-':>3} | {ok:>3} | {d.margin_db:9.1f}"
        )

    print()
    print(f"Decoded bits    : {result.decoded_bits}")
    if result.ber is not None:
        print(f"Expected bits   : {result.expected_bits}")
        print(f"Bit errors      : {result.bit_errors} / {len(result.expected_bits)}")
        print(f"BER             : {result.ber:.4f} ({result.ber * 100:.2f}%)")
    print(f"Mean margin     : {result.mean_margin_db:.1f} dB")
    if args.as_text and len(result.decoded_bits) % 8 == 0:
        print(f"Decoded text    : {bits_to_text(result.decoded_bits)!r}")

    if args.plot_dir is not None:
        print("Saved plots:")
        for path in plot_trial(signal, sample_rate, result, args.plot_dir, prefix=args.recording.stem):
            print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
