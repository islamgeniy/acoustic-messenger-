#!/usr/bin/env python3
"""CLI for decoding a recorded 2-FSK acoustic WAV file and reporting BER.

Examples:
    # Auto-sync decode using companion TX metadata JSON
    python scripts/decode_recording.py data/recorded/baseline/trial1.wav \
        --meta data/transmitted/baseline_32bit.json

    # Manual start-time decode (e.g., for timing-offset checks or noisy rooms)
    python scripts/decode_recording.py data/recorded/baseline/trial1.wav \
        --meta data/transmitted/baseline_32bit.json --start-time 0.50

    # Decode without metadata JSON (provide parameters directly)
    python scripts/decode_recording.py recording.wav --num-bits 32 --f0 1000 --f1 2000 --bit-duration 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_metadata
from src.config import DEFAULT_BIT_DURATION, DEFAULT_F0, DEFAULT_F1, ModulationConfig
from src.rx import decode_wav
from src.tx import bits_to_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode a recorded 2-FSK acoustic WAV file and compute BER."
    )
    parser.add_argument("recording", type=Path, help="Path to the recorded WAV file.")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional companion TX JSON metadata (auto-loads f0, f1, Tb, expected bits).",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=None,
        help="Number of payload bits to decode (inferred from --meta or --expected-bits if omitted).",
    )
    parser.add_argument(
        "--expected-bits",
        type=str,
        default=None,
        help="Optional expected bitstring for BER evaluation.",
    )
    parser.add_argument("--f0", type=float, default=None, help="Bit '0' frequency in Hz.")
    parser.add_argument("--f1", type=float, default=None, help="Bit '1' frequency in Hz.")
    parser.add_argument(
        "--bit-duration",
        type=float,
        default=None,
        help="Bit interval duration in seconds.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Manual start time in seconds (omit for automatic synchronization).",
    )
    parser.add_argument(
        "--window",
        choices=["hann", "rect"],
        default="hann",
        help="FFT window applied to each bit interval (default: hann).",
    )
    parser.add_argument(
        "--guard",
        type=float,
        default=0.05,
        help="Fraction of each bit trimmed at both edges to reduce boundary reverb (default: 0.05).",
    )
    parser.add_argument(
        "--as-text",
        action="store_true",
        help="Also decode the recovered bitstring as 8-bit UTF-8 text.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save full decode result as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.meta is not None:
        meta = load_metadata(args.meta)
        cfg = meta.config
        expected_bits = args.expected_bits if args.expected_bits is not None else meta.bits
    else:
        cfg = ModulationConfig(
            f0=DEFAULT_F0,
            f1=DEFAULT_F1,
            bit_duration=DEFAULT_BIT_DURATION,
        )
        expected_bits = args.expected_bits

    # Allow explicit CLI flags to override metadata parameters
    if args.f0 is not None:
        cfg.f0 = args.f0
    if args.f1 is not None:
        cfg.f1 = args.f1
    if args.bit_duration is not None:
        cfg.bit_duration = args.bit_duration

    num_bits = args.num_bits
    if num_bits is None:
        if expected_bits is not None:
            num_bits = len(expected_bits)
        else:
            parser.error("Specify --num-bits, --expected-bits, or --meta.")

    result = decode_wav(
        args.recording,
        num_bits=num_bits,
        config=cfg,
        start_time=args.start_time,
        expected_bits=expected_bits,
        window=args.window,
        guard_fraction=args.guard,
    )

    print("=== 2-FSK Acoustic Receiver ===")
    print(f"Recording       : {args.recording}")
    print(f"Sync method     : {result.sync_method} (payload start = {result.start_time:.4f} s)")
    print(f"Frequencies     : 0 -> {cfg.f0:.1f} Hz | 1 -> {cfg.f1:.1f} Hz | Tb = {cfg.bit_duration:.4f} s")
    print()
    print(f"{'Bit':>3} | {'Interval (s)':>13} | {'Near f0':>9} | {'Near f1':>9} | {'Dec':>3} | {'Exp':>3} | {'OK':>3} | {'SNR(dB)':>7}")
    print("-" * 70)
    for d in result.bit_decisions:
        exp_str = d.expected if d.expected is not None else "-"
        ok_str = "Y" if d.is_correct else ("N" if d.is_correct is False else "-")
        snr_str = f"{d.snr_db:7.1f}" if d.snr_db is not None else "      -"
        print(
            f"{d.index + 1:3d} | {d.start_time:6.3f}-{d.end_time:6.3f} | "
            f"{d.mag_f0:9.4f} | {d.mag_f1:9.4f} | {d.decision:>3} | {exp_str:>3} | {ok_str:>3} | {snr_str}"
        )

    print()
    print(f"Decoded bits    : {result.decoded_bits}")
    if result.expected_bits is not None:
        print(f"Expected bits   : {result.expected_bits}")
        assert result.bit_errors is not None and result.ber is not None
        print(f"Bit errors      : {result.bit_errors} / {len(result.expected_bits)}")
        print(f"BER             : {result.ber:.4f} ({result.ber * 100:.2f}%)")
    if result.mean_snr_db is not None:
        print(f"Mean est. SNR   : {result.mean_snr_db:.2f} dB")
    if args.as_text and len(result.decoded_bits) % 8 == 0:
        print(f"Decoded text    : {bits_to_text(result.decoded_bits)!r}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"Saved JSON      : {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
