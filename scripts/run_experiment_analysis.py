#!/usr/bin/env python3
"""Batch experiment analysis and diagnostic plotting CLI.

Modes:
    1. Evaluate real recorded trials from a manifest JSON:
       python scripts/run_experiment_analysis.py --manifest data/recorded/manifest.json

    2. Generate full diagnostic plots for a single recorded WAV:
       python scripts/run_experiment_analysis.py --single data/recorded/baseline/trial1.wav \
           --meta data/transmitted/baseline_32bit.json --out-dir results/plots/baseline

    3. Run a clearly labelled simulation preview across all 4 controlled experiments:
       python scripts/run_experiment_analysis.py --demo-sim
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_metadata, load_wav
from src.config import DEFAULT_TEST_BITS, PLOTS_DIR, RESULTS_DIR, TRIALS_CSV
from src.experiments import (
    evaluate_manifest,
    generate_single_trial_plots,
    load_manifest,
    run_simulation_preview,
)
from src.rx import decode_signal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Acoustic Messenger trials and generate labelled plots."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--manifest",
        type=Path,
        help="Path to JSON manifest listing real experiment recordings.",
    )
    mode.add_argument(
        "--single",
        type=Path,
        help="Path to a single recorded WAV file to decode and plot.",
    )
    mode.add_argument(
        "--demo-sim",
        action="store_true",
        help="Run synthetic channel preview of all 4 experiments into results/sim_preview/.",
    )

    parser.add_argument("--meta", type=Path, default=None, help="TX metadata JSON (required with --single).")
    parser.add_argument("--start-time", type=float, default=None, help="Optional manual start time (with --single).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated plots.")
    parser.add_argument("--csv", type=Path, default=TRIALS_CSV, help="Output master trials CSV path (with --manifest).")
    parser.add_argument("--prefix", type=str, default="recording", help="Filename prefix for --single plots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo_sim:
        out_root = args.out_dir if args.out_dir is not None else RESULTS_DIR / "sim_preview"
        records, plots = run_simulation_preview(DEFAULT_TEST_BITS, out_root)
        print(f"=== Simulation Preview Complete ({len(records)} simulated trials) ===")
        print(f"CSV   : {out_root / 'sim_preview_trials.csv'}")
        print(f"Plots : {len(plots)} PNGs saved in {out_root}")
        return 0

    if args.single is not None:
        if args.meta is None:
            parser.error("--single requires --meta <tx_metadata.json>.")
        meta = load_metadata(args.meta)
        fs, sig = load_wav(args.single, remove_dc=True)
        res = decode_signal(
            sig,
            fs,
            num_bits=len(meta.bits),
            config=meta.config,
            start_time=args.start_time,
            expected_bits=meta.bits,
            recording_path=str(args.single),
        )
        out_dir = args.out_dir if args.out_dir is not None else PLOTS_DIR / "single"
        plots = generate_single_trial_plots(sig, fs, res, out_dir, prefix=args.prefix)
        print("=== Single Recording Analysis ===")
        print(f"Sync      : {res.sync_method} (start = {res.start_time:.4f} s)")
        if res.ber is not None:
            print(f"BER       : {res.ber:.4f} ({res.ber * 100:.2f}%, {res.bit_errors}/{len(meta.bits)} errors)")
        if res.mean_snr_db is not None:
            print(f"Mean SNR  : {res.mean_snr_db:.2f} dB")
        print("Saved plots:")
        for p in plots:
            print(f"  - {p}")
        return 0

    assert args.manifest is not None
    specs = load_manifest(args.manifest)
    plots_root = args.out_dir if args.out_dir is not None else PLOTS_DIR
    records, plots = evaluate_manifest(specs, plots_root=plots_root, csv_path=args.csv)
    print(f"=== Manifest Evaluation Complete ({len(records)} trials) ===")
    print(f"Master CSV : {args.csv}")
    print(f"Plots      : {len(plots)} PNGs saved under {plots_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
