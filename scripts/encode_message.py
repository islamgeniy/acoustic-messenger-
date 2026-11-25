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

    # Slide 4 sanity check: 4 bits "1011"
    python scripts/encode_message.py --bits 1011 --allow-short --play
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
from src.config import DEFAULT_TEST_BITS, TRANSMITTED_DIR, ModulationConfig
from src.tx import encode_to_wav


def build_parser() -> argparse.ArgumentParser:
    defaults = ModulationConfig()
    parser = argparse.ArgumentParser(description="Encode a binary or text message into a 2-FSK acoustic WAV file.")

    msg = parser.add_mutually_exclusive_group()
    msg.add_argument("--bits", help=f"Binary payload, 32+ bits of '0'/'1' (default: {DEFAULT_TEST_BITS}).")
    msg.add_argument("--text", help="Text message, sent as 8 bits per byte (4+ ASCII chars = 32+ bits).")

    parser.add_argument("-o", "--output", type=Path, help="Output WAV path (default: data/transmitted/tx_<params>.wav).")
    parser.add_argument("--f0", type=float, default=defaults.f0, help="Bit '0' frequency in Hz.")
    parser.add_argument("--f1", type=float, default=defaults.f1, help="Bit '1' frequency in Hz.")
    parser.add_argument("--bit-duration", type=float, default=defaults.bit_duration, help="Bit duration Tb in seconds.")
    parser.add_argument("--allow-short", action="store_true", help="Allow fewer than 32 bits (e.g., Slide 4 '1011').")
    parser.add_argument("--play", action="store_true", help="Play the generated audio through the default speaker.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ModulationConfig(f0=args.f0, f1=args.f1, bit_duration=args.bit_duration)

    if args.output is not None:
        out_path = args.output
    else:
        tag = f"f0_{cfg.f0:.0f}_f1_{cfg.f1:.0f}_tb_{cfg.bit_duration * 1000:.0f}ms"
        out_path = TRANSMITTED_DIR / f"tx_{tag}.wav"

    if args.text is not None:
        wav_path, meta_path, metadata, signal = encode_to_wav(out_path, text=args.text, config=cfg, allow_short=args.allow_short)
    else:
        bits = args.bits if args.bits is not None else DEFAULT_TEST_BITS
        wav_path, meta_path, metadata, signal = encode_to_wav(out_path, bits=bits, config=cfg, allow_short=args.allow_short)

    active_dur = len(metadata.bits) * cfg.bit_duration
    print("=== 2-FSK Acoustic Transmitter ===")
    if metadata.text is not None:
        print(f"Text            : {metadata.text!r}")
    print(f"Payload bits    : {metadata.bits} ({len(metadata.bits)} bits)")
    print(f"Frequencies     : 0 -> {cfg.f0:.1f} Hz | 1 -> {cfg.f1:.1f} Hz (Δf = {abs(cfg.f1 - cfg.f0):.1f} Hz)")
    print(f"Bit duration    : {cfg.bit_duration:.4f} s ({cfg.samples_per_bit} samples/bit @ {cfg.sample_rate} Hz)")
    print(f"Active window   : [{cfg.lead_silence:.3f} s, {cfg.lead_silence + active_dur:.3f} s] "
          f"(total {len(signal) / cfg.sample_rate:.3f} s)")
    print(f"Saved WAV       : {wav_path}")
    print(f"Saved metadata  : {meta_path}")

    if args.play:
        print("Playing audio through default speaker...")
        play_audio(signal, cfg.sample_rate)
        print("Playback complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
