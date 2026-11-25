#!/usr/bin/env python3
"""Evaluate all recorded experiment trials listed in a manifest JSON.

Writes `results/trials.csv` (every trial, including failures) and per-experiment
plots under `results/plots/<experiment>/`.

Example:
    python scripts/run_experiment_analysis.py --manifest data/recorded/manifest.json

To decode and plot a single recording, use `scripts/decode_recording.py --plot-dir`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PLOTS_DIR, TRIALS_CSV
from src.experiments import evaluate_manifest, load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate recorded experiment trials and generate labelled plots.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest listing the recorded trials.")
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR, help="Root folder for the generated plots.")
    parser.add_argument("--csv", type=Path, default=TRIALS_CSV, help="Output trials CSV path.")
    args = parser.parse_args(argv)

    records, plots = evaluate_manifest(load_manifest(args.manifest), plots_root=args.out_dir, csv_path=args.csv)
    failed = sum(not r.passed for r in records)
    print(f"=== Manifest Evaluation Complete ({len(records)} trials, {failed} with bit errors) ===")
    print(f"Trials CSV : {args.csv}")
    print(f"Plots      : {len(plots)} PNGs saved under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
