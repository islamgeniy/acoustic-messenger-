"""Controlled experiment batch runner, CSV trial logger, and channel simulator.

Manages evaluation across all 4 controlled experiments, writes the master
`results/trials.csv` (retaining failed trials per Slide 6 & Slide 14), and
generates standardized waveform, spectrogram, bit-spectrum, and BER/SNR plots.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from src.audio_io import load_metadata, load_wav
from src.config import DecodeResult, ModulationConfig, TrialRecord
from src.plotting import (
    plot_bit_decisions,
    plot_bit_spectra_comparison,
    plot_experiment_performance,
    plot_frequency_attenuation,
    plot_spectrogram,
    plot_waveform_with_boundaries,
)
from src.rx import decode_signal, decode_wav
from src.tx import modulate_fsk


TRIAL_CSV_FIELDS = [
    "experiment",
    "leader",
    "factor",
    "setting",
    "trial_num",
    "tx_wav",
    "rx_wav",
    "start_time",
    "total_bits",
    "bit_errors",
    "ber",
    "mean_snr_db",
    "passed",
    "notes",
]


@dataclass
class TrialSpec:
    """Manifest entry describing one recorded trial to evaluate."""

    experiment: str
    leader: str
    factor: str
    setting: str
    trial_num: int
    tx_meta: Path
    rx_wav: Path
    start_time: Optional[float] = None
    window: str = "hann"
    guard_fraction: float = 0.05
    notes: str = ""


def write_trials_csv(records: Sequence[TrialRecord], csv_path: str | Path) -> Path:
    """Write all `TrialRecord` rows (including failed trials) to CSV."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIAL_CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            row = rec.to_dict()
            row["start_time"] = f"{rec.start_time:.4f}"
            row["ber"] = f"{rec.ber:.6f}"
            row["mean_snr_db"] = f"{rec.mean_snr_db:.2f}" if rec.mean_snr_db is not None else ""
            writer.writerow(row)
    return out


def load_manifest(manifest_path: str | Path) -> list[TrialSpec]:
    """Load a JSON list of trial specifications relative to the manifest directory."""
    mpath = Path(manifest_path)
    base_dir = mpath.parent
    with mpath.open("r", encoding="utf-8") as fh:
        raw_items = json.load(fh)

    specs: list[TrialSpec] = []
    for item in raw_items:
        tx_meta = Path(item["tx_meta"])
        rx_wav = Path(item["rx_wav"])
        if not tx_meta.is_absolute():
            tx_meta = (base_dir / tx_meta).resolve() if (base_dir / tx_meta).exists() else tx_meta.resolve()
        if not rx_wav.is_absolute():
            rx_wav = (base_dir / rx_wav).resolve() if (base_dir / rx_wav).exists() else rx_wav.resolve()
        specs.append(
            TrialSpec(
                experiment=item["experiment"],
                leader=item["leader"],
                factor=item["factor"],
                setting=str(item["setting"]),
                trial_num=int(item["trial_num"]),
                tx_meta=tx_meta,
                rx_wav=rx_wav,
                start_time=item.get("start_time"),
                window=item.get("window", "hann"),
                guard_fraction=float(item.get("guard_fraction", 0.05)),
                notes=item.get("notes", ""),
            )
        )
    return specs


def generate_single_trial_plots(
    signal: np.ndarray,
    sample_rate: int,
    result: DecodeResult,
    out_dir: str | Path,
    prefix: str = "trial",
    window: str = "hann",
    guard_fraction: float = 0.05,
) -> list[Path]:
    """Generate waveform, spectrogram, bit-decision bars, and pass/fail bit spectra."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    saved.append(
        plot_waveform_with_boundaries(
            signal, sample_rate, result, target / f"{prefix}_waveform.png",
            title=f"{prefix}: Waveform & Bit Boundaries",
        )
    )
    saved.append(
        plot_spectrogram(
            signal, sample_rate, result.config, target / f"{prefix}_spectrogram.png",
            title=f"{prefix}: Spectrogram (f0={result.config.f0:.0f} Hz, f1={result.config.f1:.0f} Hz)",
        )
    )
    saved.append(
        plot_bit_decisions(
            result, target / f"{prefix}_bit_decisions.png",
            title=f"{prefix}: Per-Bit Spectral Magnitudes",
        )
    )

    # Pick one clean/successful bit and one failed (or lowest-margin) bit for spectral inspection
    ok_bits = [d.index for d in result.bit_decisions if d.is_correct is True]
    err_bits = [d.index for d in result.bit_decisions if d.is_correct is False]
    selected: list[int] = []
    if ok_bits:
        selected.append(ok_bits[0])
    if err_bits:
        selected.append(err_bits[0])
    elif len(ok_bits) > 1:
        # Show both a '0' bit and a '1' bit when all bits succeed
        zero_bits = [d.index for d in result.bit_decisions if d.decision == "0"]
        one_bits = [d.index for d in result.bit_decisions if d.decision == "1"]
        selected = [zero_bits[0] if zero_bits else 0, one_bits[0] if one_bits else min(1, len(result.bit_decisions) - 1)]
    if not selected:
        selected = [0]

    saved.append(
        plot_bit_spectra_comparison(
            signal,
            sample_rate,
            result,
            selected,
            target / f"{prefix}_bit_spectra.png",
            title=f"{prefix}: Labelled Bit-Interval Spectra",
            window=window,
            guard_fraction=guard_fraction,
        )
    )
    return saved


def evaluate_manifest(
    specs: Sequence[TrialSpec],
    plots_root: str | Path,
    csv_path: str | Path,
) -> tuple[list[TrialRecord], list[Path]]:
    """Decode all trials in `specs`, write `trials.csv`, and generate plots per experiment."""
    records: list[TrialRecord] = []
    saved_plots: list[Path] = []
    plots_base = Path(plots_root)

    by_exp: dict[str, list[tuple[TrialSpec, TrialRecord, DecodeResult, np.ndarray, int]]] = defaultdict(list)

    for spec in specs:
        meta = load_metadata(spec.tx_meta)
        fs, sig = load_wav(spec.rx_wav, remove_dc=True)
        res = decode_signal(
            sig,
            fs,
            num_bits=len(meta.bits),
            config=meta.config,
            start_time=spec.start_time,
            expected_bits=meta.bits,
            window=spec.window,
            guard_fraction=spec.guard_fraction,
            recording_path=str(spec.rx_wav),
        )
        assert res.bit_errors is not None and res.ber is not None
        rec = TrialRecord(
            experiment=spec.experiment,
            leader=spec.leader,
            factor=spec.factor,
            setting=spec.setting,
            trial_num=spec.trial_num,
            tx_wav=meta.wav_path or spec.tx_meta.stem,
            rx_wav=spec.rx_wav.name,
            start_time=res.start_time,
            total_bits=len(meta.bits),
            bit_errors=res.bit_errors,
            ber=res.ber,
            mean_snr_db=res.mean_snr_db,
            passed=(res.bit_errors == 0),
            notes=spec.notes,
        )
        records.append(rec)
        by_exp[spec.experiment].append((spec, rec, res, sig, fs))

    write_trials_csv(records, csv_path)

    for exp_name, items in by_exp.items():
        exp_dir = plots_base / exp_name
        exp_records = [x[1] for x in items]
        factor_label = items[0][0].factor
        leader = items[0][0].leader

        saved_plots.append(
            plot_experiment_performance(
                exp_records,
                exp_dir / f"{exp_name}_ber_snr.png",
                title=f"{exp_name} ({factor_label}) — Led by {leader}",
                xlabel=factor_label,
            )
        )

        # Best (lowest BER) and worst (highest BER) diagnostic plots for Slide 14
        sorted_items = sorted(items, key=lambda x: (x[1].ber, -(x[1].mean_snr_db or 0.0)))
        best_spec, _, best_res, best_sig, best_fs = sorted_items[0]
        saved_plots.extend(
            generate_single_trial_plots(
                best_sig, best_fs, best_res, exp_dir,
                prefix=f"best_{best_spec.setting.replace(' ', '_')}",
                window=best_spec.window,
                guard_fraction=best_spec.guard_fraction,
            )
        )
        if len(sorted_items) > 1:
            worst_spec, _, worst_res, worst_sig, worst_fs = sorted_items[-1]
            saved_plots.extend(
                generate_single_trial_plots(
                    worst_sig, worst_fs, worst_res, exp_dir,
                    prefix=f"difficult_{worst_spec.setting.replace(' ', '_')}",
                    window=worst_spec.window,
                    guard_fraction=worst_spec.guard_fraction,
                )
            )

        # Bonus frequency-attenuation chart when factor involves distance/position
        if "distance" in exp_name.lower() or "position" in factor_label.lower():
            by_setting: dict[str, list[DecodeResult]] = defaultdict(list)
            for spec, _, res, _, _ in items:
                by_setting[spec.setting].append(res)
            saved_plots.append(
                plot_frequency_attenuation(
                    by_setting,
                    exp_dir / f"{exp_name}_freq_attenuation.png",
                )
            )

    return records, saved_plots


def simulate_acoustic_channel(
    signal: np.ndarray,
    sample_rate: int,
    *,
    gain: float = 1.0,
    noise_std: float = 0.005,
    reverb_decay_ms: float = 0.0,
    reverb_wet: float = 0.0,
    lowpass_hz: Optional[float] = None,
    interf_freq_hz: Optional[float] = None,
    interf_amp: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic room-channel model for software dry-runs and unit tests.

    Models inverse-square gain attenuation, optional obstacle low-pass filtering
    (stronger attenuation at 2,000 Hz than 1,000 Hz), exponential room
    reverberation impulse response `h[n]`, additive Gaussian noise, and an
    optional interfering tone.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(signal, dtype=np.float64) * gain

    if lowpass_hz is not None and 0.0 < lowpass_hz < sample_rate / 2.0:
        sos = butter(2, lowpass_hz / (sample_rate / 2.0), btype="lowpass", output="sos")
        out = sosfilt(sos, out)

    if reverb_decay_ms > 0.0 and reverb_wet > 0.0:
        ir_len = max(16, int(round((reverb_decay_ms * 4.0 / 1000.0) * sample_rate)))
        t_ir = np.arange(ir_len, dtype=np.float64) / float(sample_rate)
        tau = reverb_decay_ms / 1000.0
        ir = rng.standard_normal(ir_len) * np.exp(-t_ir / tau)
        ir /= np.sqrt(np.sum(ir * ir)) + 1e-12
        ir = ir * reverb_wet
        ir[0] += 1.0
        out = lfilter(ir, [1.0], out)

    if interf_freq_hz is not None and interf_amp > 0.0:
        t = np.arange(out.size, dtype=np.float64) / float(sample_rate)
        out = out + interf_amp * np.sin(2.0 * np.pi * interf_freq_hz * t)

    if noise_std > 0.0:
        out = out + noise_std * rng.standard_normal(out.size)

    return out


def run_simulation_preview(
    bits: str,
    out_root: str | Path,
) -> tuple[list[TrialRecord], list[Path]]:
    """Run a clearly labelled simulation preview across all 4 controlled experiments."""
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    all_records: list[TrialRecord] = []
    all_plots: list[Path] = []

    # Exp 1: Bit Duration (Shuhratov Abdulloh)
    exp1_recs: list[TrialRecord] = []
    for tb in (0.10, 0.05, 0.02, 0.01):
        cfg = ModulationConfig(bit_duration=tb, lead_silence=0.3, trail_silence=0.3)
        clean = modulate_fsk(bits, cfg)
        for trial in range(1, 4):
            rx = simulate_acoustic_channel(clean, cfg.sample_rate, gain=0.4, noise_std=0.02, reverb_decay_ms=25.0, reverb_wet=0.65, seed=100 + trial + int(tb * 1000))
            res = decode_signal(rx, cfg.sample_rate, len(bits), cfg, start_time=0.3, expected_bits=bits)
            assert res.bit_errors is not None and res.ber is not None
            exp1_recs.append(TrialRecord("exp1_bit_duration_SIM", "Shuhratov Abdulloh", "Bit Duration (Tb)", f"{int(tb*1000)} ms", trial, "sim_tx", "sim_rx", res.start_time, len(bits), res.bit_errors, res.ber, res.mean_snr_db, res.bit_errors == 0, "simulation preview"))
    all_records.extend(exp1_recs)
    all_plots.append(plot_experiment_performance(exp1_recs, root / "exp1_bit_duration_sim.png", title="[SIM PREVIEW] Exp 1: Bit Duration (Shuhratov Abdulloh)", xlabel="Bit Duration (Tb)"))

    # Exp 2: Tone Separation (Xojambetov Islam)
    exp2_recs: list[TrialRecord] = []
    for df in (1000.0, 200.0, 50.0, 10.0):
        cfg = ModulationConfig(f0=1000.0, f1=1000.0 + df, bit_duration=0.10, lead_silence=0.3, trail_silence=0.3)
        clean = modulate_fsk(bits, cfg)
        for trial in range(1, 4):
            rx = simulate_acoustic_channel(clean, cfg.sample_rate, gain=0.4, noise_std=0.05, reverb_decay_ms=20.0, reverb_wet=0.3, seed=200 + trial + int(df))
            res = decode_signal(rx, cfg.sample_rate, len(bits), cfg, start_time=0.3, expected_bits=bits)
            assert res.bit_errors is not None and res.ber is not None
            exp2_recs.append(TrialRecord("exp2_tone_separation_SIM", "Xojambetov Islam", "Tone Separation (Δf)", f"{int(df)} Hz", trial, "sim_tx", "sim_rx", res.start_time, len(bits), res.bit_errors, res.ber, res.mean_snr_db, res.bit_errors == 0, "simulation preview"))
    all_records.extend(exp2_recs)
    all_plots.append(plot_experiment_performance(exp2_recs, root / "exp2_tone_separation_sim.png", title="[SIM PREVIEW] Exp 2: Tone Separation (Xojambetov Islam)", xlabel="Tone Separation Δf = f1 - f0"))

    # Exp 3: Background Interference (Mirobidov Mirakbar)
    exp3_recs: list[TrialRecord] = []
    cfg_base = ModulationConfig(lead_silence=0.3, trail_silence=0.3)
    clean_base = modulate_fsk(bits, cfg_base)
    interf_settings = [
        ("Quiet baseline", None, 0.0, 0.01),
        ("Out-of-band 400 Hz", 400.0, 0.45, 0.02),
        ("In-band 1000 Hz (low)", 1000.0, 0.15, 0.02),
        ("In-band 1000 Hz (loud)", 1000.0, 0.55, 0.02),
    ]
    for label, f_int, a_int, n_std in interf_settings:
        for trial in range(1, 4):
            rx = simulate_acoustic_channel(clean_base, cfg_base.sample_rate, gain=0.4, noise_std=n_std, interf_freq_hz=f_int, interf_amp=a_int, seed=300 + trial + len(label))
            res = decode_signal(rx, cfg_base.sample_rate, len(bits), cfg_base, start_time=0.3, expected_bits=bits)
            assert res.bit_errors is not None and res.ber is not None
            exp3_recs.append(TrialRecord("exp3_interference_SIM", "Mirobidov Mirakbar", "Background Interference", label, trial, "sim_tx", "sim_rx", res.start_time, len(bits), res.bit_errors, res.ber, res.mean_snr_db, res.bit_errors == 0, "simulation preview"))
    all_records.extend(exp3_recs)
    all_plots.append(plot_experiment_performance(exp3_recs, root / "exp3_interference_sim.png", title="[SIM PREVIEW] Exp 3: Background Interference (Mirobidov Mirakbar)", xlabel="Interference Condition"))

    # Exp 4: Distance & Position (Malikov Amirxon)
    exp4_recs: list[TrialRecord] = []
    exp4_res_by_setting: dict[str, list[DecodeResult]] = defaultdict(list)
    pos_settings = [
        ("0.5m LOS", 0.60, 0.012, 10.0, 0.15, None),
        ("2.0m LOS", 0.18, 0.015, 25.0, 0.35, None),
        ("5.0m LOS", 0.055, 0.022, 45.0, 0.65, 2600.0),
        ("2.0m NLOS (blocked)", 0.075, 0.020, 40.0, 0.70, 1250.0),
    ]
    nlos_sample_rx = None
    nlos_sample_res = None
    for label, gain, n_std, rev_ms, rev_wet, lp_hz in pos_settings:
        for trial in range(1, 4):
            rx = simulate_acoustic_channel(clean_base, cfg_base.sample_rate, gain=gain, noise_std=n_std, reverb_decay_ms=rev_ms, reverb_wet=rev_wet, lowpass_hz=lp_hz, seed=400 + trial + len(label))
            res = decode_signal(rx, cfg_base.sample_rate, len(bits), cfg_base, start_time=0.3, expected_bits=bits)
            assert res.bit_errors is not None and res.ber is not None
            exp4_recs.append(TrialRecord("exp4_distance_position_SIM", "Malikov Amirxon", "Distance & Position", label, trial, "sim_tx", "sim_rx", res.start_time, len(bits), res.bit_errors, res.ber, res.mean_snr_db, res.bit_errors == 0, "simulation preview"))
            exp4_res_by_setting[label].append(res)
            if "NLOS" in label and trial == 1:
                nlos_sample_rx, nlos_sample_res = rx, res

    all_records.extend(exp4_recs)
    all_plots.append(plot_experiment_performance(exp4_recs, root / "exp4_distance_position_sim.png", title="[SIM PREVIEW] Exp 4: Distance & Position (Malikov Amirxon)", xlabel="Acoustic Path Position"))
    all_plots.append(plot_frequency_attenuation(exp4_res_by_setting, root / "exp4_freq_attenuation_sim.png", title="[SIM PREVIEW] Exp 4: Frequency-Dependent Attenuation (1 kHz vs. 2 kHz)"))
    if nlos_sample_rx is not None and nlos_sample_res is not None:
        all_plots.extend(generate_single_trial_plots(nlos_sample_rx, cfg_base.sample_rate, nlos_sample_res, root / "exp4_nlos_diagnostics", prefix="nlos_sim"))

    write_trials_csv(all_records, root / "sim_preview_trials.csv")
    return all_records, all_plots
