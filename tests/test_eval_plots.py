"""Tests for the experiment evaluation and plotting (`src/experiments.py`, `src/plotting.py`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.audio_io import save_wav
from src.config import DEFAULT_TEST_BITS, ModulationConfig
from src.experiments import evaluate_manifest, load_manifest
from src.tx import encode_to_wav


def test_manifest_evaluation_retains_failed_trials_and_generates_plots(tmp_path: Path) -> None:
    cfg = ModulationConfig(lead_silence=0.25, trail_silence=0.25)
    _, tx_meta, _, clean = encode_to_wav(tmp_path / "tx_base.wav", bits=DEFAULT_TEST_BITS, config=cfg)

    # Two "recordings": a clear one, and one attenuated to 0.5% under heavy noise (must fail)
    rng = np.random.default_rng(2)
    save_wav(tmp_path / "rx_good.wav", 0.6 * clean + 0.005 * rng.standard_normal(clean.size), cfg.sample_rate)
    save_wav(tmp_path / "rx_bad.wav", 0.005 * clean + 0.15 * rng.standard_normal(clean.size), cfg.sample_rate)

    common = {"experiment": "exp4_distance_position", "leader": "Malikov Amirxon",
              "factor": "Distance & Position", "trial_num": 1, "tx_meta": str(tx_meta), "start_time": 0.25}
    manifest = [
        {**common, "setting": "0.5m LOS", "rx_wav": "rx_good.wav"},  # relative to the manifest folder
        {**common, "setting": "2.0m NLOS", "rx_wav": "rx_bad.wav"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    csv_out = tmp_path / "trials.csv"
    records, saved_plots = evaluate_manifest(load_manifest(manifest_path), plots_root=tmp_path / "plots", csv_path=csv_out)

    assert records[0].ber == 0.0 and records[0].passed is True
    assert records[1].ber > 0.0 and records[1].passed is False  # failed trial retained

    with csv_out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and rows[1]["passed"] == "False"

    # 2 experiment plots + 4 diagnostic plots each for the best and the difficult trial
    assert len(saved_plots) == 10
    for p in saved_plots:
        assert p.exists() and p.stat().st_size > 1_000


def test_decode_cli_writes_diagnostic_plots(tmp_path: Path) -> None:
    tx_wav, tx_meta, _, _ = encode_to_wav(tmp_path / "single.wav", bits=DEFAULT_TEST_BITS)
    out_dir = tmp_path / "plots"
    script = Path(__file__).resolve().parent.parent / "scripts" / "decode_recording.py"

    proc = subprocess.run(
        [sys.executable, str(script), str(tx_wav), "--meta", str(tx_meta), "--plot-dir", str(out_dir)],
        capture_output=True, text=True, check=True,
    )
    assert "BER             : 0.0000 (0.00%)" in proc.stdout
    for kind in ("waveform", "spectrogram", "bit_decisions", "bit_spectra"):
        assert (out_dir / f"single_{kind}.png").exists()
