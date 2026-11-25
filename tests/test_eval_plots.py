"""Unit tests for the evaluation harness and plotting utilities (`src/experiments.py`, `src/plotting.py`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from src.audio_io import save_wav
from src.config import DEFAULT_TEST_BITS, ModulationConfig
from src.experiments import evaluate_manifest, load_manifest, simulate_acoustic_channel
from src.tx import encode_to_wav


def test_manifest_evaluation_retains_failed_trials_and_generates_plots(tmp_path: Path) -> None:
    cfg = ModulationConfig(lead_silence=0.25, trail_silence=0.25)
    tx_wav, tx_meta, _, clean = encode_to_wav(
        tmp_path / "tx_base.wav",
        bits=DEFAULT_TEST_BITS,
        config=cfg,
    )

    # Create 2 synthetic recordings: 1 clean (0.5m LOS) and 1 heavily degraded (NLOS)
    rx_good = tmp_path / "rx_good.wav"
    rx_bad = tmp_path / "rx_bad.wav"
    save_wav(rx_good, simulate_acoustic_channel(clean, cfg.sample_rate, gain=0.6, noise_std=0.005, seed=1), cfg.sample_rate)
    save_wav(
        rx_bad,
        simulate_acoustic_channel(
            clean, cfg.sample_rate, gain=0.04, noise_std=0.25, lowpass_hz=1200.0, seed=2
        ),
        cfg.sample_rate,
    )

    manifest_path = tmp_path / "manifest.json"
    manifest = [
        {
            "experiment": "exp4_distance_position",
            "leader": "Malikov Amirxon",
            "factor": "Distance & Position",
            "setting": "0.5m LOS",
            "trial_num": 1,
            "tx_meta": str(tx_meta),
            "rx_wav": str(rx_good),
            "start_time": 0.25,
        },
        {
            "experiment": "exp4_distance_position",
            "leader": "Malikov Amirxon",
            "factor": "Distance & Position",
            "setting": "2.0m NLOS",
            "trial_num": 1,
            "tx_meta": str(tx_meta),
            "rx_wav": str(rx_bad),
            "start_time": 0.25,
        },
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    specs = load_manifest(manifest_path)
    csv_out = tmp_path / "trials.csv"
    plots_out = tmp_path / "plots"
    records, saved_plots = evaluate_manifest(specs, plots_root=plots_out, csv_path=csv_out)

    assert len(records) == 2
    assert records[0].ber == 0.0 and records[0].passed is True
    assert records[1].ber > 0.0 and records[1].passed is False  # failed trial retained!

    # Verify CSV contains both rows including the failed trial
    with csv_out.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[1]["passed"] == "False"

    # Verify performance, frequency-attenuation, and diagnostic plots exist
    assert len(saved_plots) >= 6
    for p in saved_plots:
        assert p.exists() and p.stat().st_size > 1_000


def test_cli_single_recording_mode(tmp_path: Path) -> None:
    tx_wav, tx_meta, _, _ = encode_to_wav(tmp_path / "single.wav", bits=DEFAULT_TEST_BITS)
    out_dir = tmp_path / "single_plots"
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_experiment_analysis.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--single",
            str(tx_wav),
            "--meta",
            str(tx_meta),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "BER       : 0.0000 (0.00%" in proc.stdout
    assert (out_dir / "recording_waveform.png").exists()
    assert (out_dir / "recording_spectrogram.png").exists()
    assert (out_dir / "recording_bit_decisions.png").exists()
    assert (out_dir / "recording_bit_spectra.png").exists()
