"""Controlled experiment evaluation: decode every recorded trial, log it, and plot it.

A manifest JSON lists the trials (experiment, setting, trial number, TX metadata,
RX recording). Every trial, including failed ones, is written to `trials.csv`
(Slides 6 and 14), and each experiment gets its performance plots plus the full
diagnostic plots of its best and its most difficult trial.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.audio_io import load_metadata, load_wav
from src.config import DecodeResult, TrialRecord
from src.plotting import plot_experiment_performance, plot_tone_magnitudes, plot_trial
from src.rx import decode_transmission


@dataclass
class TrialSpec:
    """One manifest entry: which recording to decode against which transmission."""

    experiment: str
    leader: str
    factor: str
    setting: str
    trial_num: int
    tx_meta: Path
    rx_wav: Path
    start_time: Optional[float] = None   # None = automatic start detection
    notes: str = ""


@dataclass
class Trial:
    """A decoded trial: its CSV row plus what the plots need."""

    spec: TrialSpec
    record: TrialRecord
    result: DecodeResult
    signal: np.ndarray
    sample_rate: int


def load_manifest(manifest_path: str | Path) -> list[TrialSpec]:
    """Load a JSON list of trials. Relative paths are relative to the manifest's folder."""
    path = Path(manifest_path)
    specs = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        item["tx_meta"] = (path.parent / item["tx_meta"]).resolve()
        item["rx_wav"] = (path.parent / item["rx_wav"]).resolve()
        item["setting"] = str(item["setting"])
        specs.append(TrialSpec(**item))
    return specs


def evaluate_trial(spec: TrialSpec) -> Trial:
    """Decode one recording against its transmitted bits."""
    meta = load_metadata(spec.tx_meta)
    sample_rate, signal = load_wav(spec.rx_wav)
    result = decode_transmission(signal, sample_rate, meta, spec.start_time)
    record = TrialRecord(
        experiment=spec.experiment,
        leader=spec.leader,
        factor=spec.factor,
        setting=spec.setting,
        trial_num=spec.trial_num,
        tx_wav=meta.wav_path or spec.tx_meta.stem,
        rx_wav=spec.rx_wav.name,
        start_time=round(result.start_time, 4),
        total_bits=len(meta.bits),
        bit_errors=result.bit_errors,
        ber=round(result.ber, 6),
        mean_margin_db=round(result.mean_margin_db, 2),
        passed=result.bit_errors == 0,
        notes=spec.notes,
    )
    return Trial(spec, record, result, signal, sample_rate)


def write_trials_csv(records: Sequence[TrialRecord], csv_path: str | Path) -> Path:
    """Write every trial (including failed ones) to CSV."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(TrialRecord)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)
    return out


def evaluate_manifest(
    specs: Sequence[TrialSpec], plots_root: str | Path, csv_path: str | Path
) -> tuple[list[TrialRecord], list[Path]]:
    """Decode all trials, write `trials.csv`, and generate the plots of each experiment."""
    trials = [evaluate_trial(spec) for spec in specs]
    write_trials_csv([t.record for t in trials], csv_path)

    by_experiment: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_experiment[t.spec.experiment].append(t)
    plots = [path for name, group in by_experiment.items() for path in plot_experiment(name, group, Path(plots_root) / name)]
    return [t.record for t in trials], plots


def plot_experiment(name: str, trials: Sequence[Trial], out_dir: Path) -> list[Path]:
    """Performance and tone-magnitude plots, plus full diagnostics of the best and the most difficult trial."""
    factor, leader = trials[0].spec.factor, trials[0].spec.leader
    results_by_setting: dict[str, list[DecodeResult]] = defaultdict(list)
    for t in trials:
        results_by_setting[t.spec.setting].append(t.result)

    plots = [
        plot_experiment_performance([t.record for t in trials], out_dir / f"{name}_ber_margin.png",
                                    title=f"{name} ({factor}) — Led by {leader}", xlabel=factor),
        plot_tone_magnitudes(results_by_setting, out_dir / f"{name}_tone_magnitudes.png",
                             title=f"{name}: Received Tone Magnitudes per Setting"),
    ]
    for label, t in best_and_most_difficult(trials):
        plots.extend(plot_trial(t.signal, t.sample_rate, t.result, out_dir, f"{label}_{t.spec.setting.replace(' ', '_')}"))
    return plots


def best_and_most_difficult(trials: Sequence[Trial]) -> list[tuple[str, Trial]]:
    """Lowest BER (ties: highest margin) and highest BER: the success and the failure case of Slide 14."""
    ranked = sorted(trials, key=lambda t: (t.record.ber, -t.record.mean_margin_db))
    return [("best", ranked[0])] if len(ranked) == 1 else [("best", ranked[0]), ("difficult", ranked[-1])]
