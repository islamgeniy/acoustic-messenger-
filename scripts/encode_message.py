#!/usr/bin/env python3
"""CLI for generating (and optionally playing) 2-FSK acoustic transmissions.

Examples:
    # Default 32-bit baseline transmission
    python scripts/encode_message.py -o data/transmitted/baseline_32bit.wav

    # Custom bit sequence (e.g., live demo) + immediate speaker playback
    python scripts/encode_message.py --bits 10110010111001010110100110010110 --play

    # Encode ASCII text (>= 4 chars -> >= 32 bits)
    python scripts/encode_message.py --text "IUT!" -o data/transmitted/iut_32bit.wav

    # Exp 2 (Tone Separation): narrow separation f0=1000 Hz, f1=1050 Hz
    python scripts/encode_message.py --f0 1000 --f1 1050 -o data/transmitted/exp2_df50.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is importable when invoked as `python scripts/encode_message.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import play_audio
from src.config import (
    DEFAULT_AMPLITUDE,
    DEFAULT_BIT_DURATION,
    DEFAULT_F0,
    DEFAULT_F1,
    DEFAULT_LEAD_SILENCE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TEST_BITS,
    DEFAULT_TRAIL_SILENCE,
    TRANSMITTED_DIR,
    ModulationConfig,
)
from src.tx import encode_to_wav, generate_random_bits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode a binary or text message into a 2-FSK acoustic WAV file."
    )

    msg_group = parser.add_mutually_exclusive_group()
    msg_group.add_argument(
        "--bits",
        type=str,
        help="Binary payload string (e.g., 32+ bits of '0' and '1').",
    )
    msg_group.add_argument(
        "--text",
        type=str,
        help="Text message to convert to 8-bit binary (4+ ASCII chars = 32+ bits).",
    )
    msg_group.add_argument(
        "--random-bits",
        type=int,
        metavar="N",
        help="Generate a random N-bit payload (use with optional --seed).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed when using --random-bits.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output WAV path (default: data/transmitted/tx_<params>.wav).",
    )
    parser.add_argument("--f0", type=float, default=DEFAULT_F0, help="Bit '0' frequency in Hz.")
    parser.add_argument("--f1", type=float, default=DEFAULT_F1, help="Bit '1' frequency in Hz.")
    parser.add_argument(
        "--bit-duration",
        type=float,
        default=DEFAULT_BIT_DURATION,
        help="Duration of each bit in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate in Hz.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=DEFAULT_AMPLITUDE,
        help="Peak normalized amplitude in (0, 1].",
    )
    parser.add_argument(
        "--lead-silence",
        type=float,
        default=DEFAULT_LEAD_SILENCE,
        help="Leading silence before transmission in seconds.",
    )
    parser.add_argument(
        "--trail-silence",
        type=float,
        default=DEFAULT_TRAIL_SILENCE,
        help="Trailing silence after transmission in seconds.",
    )
    parser.add_argument(
        "--preamble",
        action="store_true",
        help="Prepend synchronization preamble bits before payload.",
    )
    parser.add_argument(
        "--preamble-bits",
        type=str,
        default="10101010",
        help="Preamble bit pattern used when --preamble is enabled.",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow messages shorter than 32 bits (e.g., 4-bit Slide-4 sanity check).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the generated audio immediately through the default speaker.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes stored in the companion JSON metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = ModulationConfig(
        sample_rate=args.sample_rate,
        f0=args.f0,
        f1=args.f1,
        bit_duration=args.bit_duration,
        amplitude=args.amplitude,
        lead_silence=args.lead_silence,
        trail_silence=args.trail_silence,
        use_preamble=args.preamble,
        preamble_bits=args.preamble_bits,
    )

    bits: str | None = None
    text: str | None = None
    if args.text is not None:
        text = args.text
    elif args.random_bits is not None:
        bits = generate_random_bits(args.random_bits, seed=args.seed)
    elif args.bits is not None:
        bits = args.bits
    else:
        bits = DEFAULT_TEST_BITS

    if args.output is not None:
        out_path = args.output
    else:
        tag = f"f0_{int(round(cfg.f0))}_f1_{int(round(cfg.f1))}_tb_{int(round(cfg.bit_duration * 1000))}ms"
        out_path = TRANSMITTED_DIR / f"tx_{tag}.wav"

    wav_path, meta_path, metadata, signal = encode_to_wav(
        out_path,
        bits=bits,
        text=text,
        config=cfg,
        allow_short=args.allow_short,
        notes=args.notes,
    )

    total_bits = len(metadata.bits) + (len(cfg.preamble_bits) if cfg.use_preamble else 0)
    active_dur = total_bits * cfg.bit_duration
    total_dur = len(signal) / float(cfg.sample_rate)

    print("=== 2-FSK Acoustic Transmitter ===")
    if metadata.text is not None:
        print(f"Text            : {metadata.text!r}")
    print(f"Payload bits    : {metadata.bits} ({len(metadata.bits)} bits)")
    if cfg.use_preamble:
        print(f"Preamble bits   : {cfg.preamble_bits} ({len(cfg.preamble_bits)} bits)")
    print(f"Frequencies     : 0 -> {cfg.f0:.1f} Hz | 1 -> {cfg.f1:.1f} Hz (Δf = {abs(cfg.f1 - cfg.f0):.1f} Hz)")
    print(f"Bit duration    : {cfg.bit_duration:.4f} s ({cfg.samples_per_bit} samples/bit @ {cfg.sample_rate} Hz)")
    print(f"Active window   : [{cfg.lead_silence:.3f} s, {cfg.lead_silence + active_dur:.3f} s] (total {total_dur:.3f} s)")
    print(f"Saved WAV       : {wav_path}")
    print(f"Saved metadata  : {meta_path}")

    if args.play:
        print("Playing audio through default speaker...")
        play_audio(signal, cfg.sample_rate)
        print("Playback complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
