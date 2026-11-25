"""Labelled plots for Acoustic Messenger (Slide 14: waveforms, labelled spectra, performance plots).

Per recording:   waveform with bit boundaries, spectrogram, per-bit magnitudes (Slide 5),
                 and the FFT spectrum of a correct bit vs. a failed bit.
Per experiment:  BER and decision margin vs. setting, and received tone magnitudes vs. setting.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from src.config import DecodeResult, TrialRecord
from src.rx import bit_segment, bit_spectrum, tone_magnitudes

COLOR_F0, COLOR_F1, COLOR_ERROR = "#1f77b4", "#ff7f0e", "#d62728"


def _save(fig: plt.Figure, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _ber_text(result: DecodeResult) -> str:
    return f" | BER = {result.ber * 100:.2f}% ({result.bit_errors}/{len(result.bit_decisions)})" if result.ber is not None else ""


def plot_waveform(signal: np.ndarray, sample_rate: int, result: DecodeResult, path: str | Path, title: str) -> Path:
    """Time-domain waveform with bit boundaries, decoded bits, and failed bits shaded red."""
    t = np.arange(signal.size) / sample_rate
    y_lim = max(1.18 * float(np.max(np.abs(signal))), 0.05)

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(t, signal, color=COLOR_F0, linewidth=0.6, alpha=0.85)
    for d in result.bit_decisions:
        ax.axvline(d.start_time, color="#555555", linestyle=":", linewidth=0.7, alpha=0.6)
        if d.is_correct is False:
            ax.axvspan(d.start_time, d.end_time, color=COLOR_ERROR, alpha=0.18)
        color = "#b00020" if d.is_correct is False else "#1b5e20"
        ax.text((d.start_time + d.end_time) / 2, 0.88 * y_lim, d.decision,
                ha="center", va="center", fontsize=7.5, fontweight="bold", color=color)
    ax.axvline(result.bit_decisions[-1].end_time, color="#555555", linestyle=":", linewidth=0.7, alpha=0.6)

    ax.set_ylim(-y_lim, y_lim)
    ax.set_title(f"{title} (sync: {result.sync_method}, start = {result.start_time:.3f} s{_ber_text(result)})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.25)
    return _save(fig, path)


def plot_spectrogram(signal: np.ndarray, sample_rate: int, result: DecodeResult, path: str | Path, title: str) -> Path:
    """Short-time Fourier transform with dashed lines at f0 and f1."""
    cfg = result.config
    nperseg = min(signal.size, max(256, int(sample_rate * min(0.04, 0.6 * cfg.bit_duration))))
    f, t, sxx = spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=int(0.75 * nperseg), window="hann")
    keep = f <= min(sample_rate / 2, 1.8 * max(cfg.f0, cfg.f1) + 400.0)
    sxx_db = 10.0 * np.log10(sxx[keep] + 1e-16)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    mesh = ax.pcolormesh(t, f[keep], sxx_db, shading="gouraud", cmap="viridis",
                         vmin=float(np.percentile(sxx_db, 15)), vmax=float(sxx_db.max()))
    fig.colorbar(mesh, ax=ax, label="Power Spectral Density (dB)")
    ax.axhline(cfg.f0, color="#ffffff", linestyle="--", linewidth=1.1, label=f"Bit '0' (f0 = {cfg.f0:.0f} Hz)")
    ax.axhline(cfg.f1, color="#ffeb3b", linestyle="--", linewidth=1.1, label=f"Bit '1' (f1 = {cfg.f1:.0f} Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.85)
    return _save(fig, path)


def plot_bit_decisions(result: DecodeResult, path: str | Path, title: str) -> Path:
    """Slide 5 as a bar chart: magnitude near f0 vs. near f1 for every bit, errors marked."""
    cfg = result.config
    idx = np.arange(1, len(result.bit_decisions) + 1)
    m0 = np.array([d.mag_f0 for d in result.bit_decisions])
    m1 = np.array([d.mag_f1 for d in result.bit_decisions])
    y_top = max(1.15 * float(max(m0.max(), m1.max())), 1e-3)

    fig, ax = plt.subplots(figsize=(12, 4.3))
    ax.bar(idx - 0.19, m0, width=0.38, color=COLOR_F0, label=f"Near f0 ({cfg.f0:.0f} Hz -> '0')")
    ax.bar(idx + 0.19, m1, width=0.38, color=COLOR_F1, label=f"Near f1 ({cfg.f1:.0f} Hz -> '1')")
    errors = [d.index + 1 for d in result.bit_decisions if d.is_correct is False]
    for i in errors:
        ax.axvspan(i - 0.5, i + 0.5, color=COLOR_ERROR, alpha=0.16)
    if errors:
        ax.plot(errors, [0.94 * y_top] * len(errors), "v", color=COLOR_ERROR, markersize=7, label="Bit error")

    ax.set_title(f"{title}{_ber_text(result)}")
    ax.set_xlabel("Bit Index (1-based)")
    ax.set_ylabel("Spectral Magnitude")
    ax.set_ylim(0.0, y_top)
    if idx.size <= 40:
        ax.set_xticks(idx)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


def plot_bit_spectra(
    signal: np.ndarray, sample_rate: int, result: DecodeResult, bit_indices: Sequence[int], path: str | Path, title: str
) -> Path:
    """FFT magnitude of selected bit intervals, exactly as the receiver measured them."""
    cfg = result.config
    pad = max(400.0, 0.5 * abs(cfg.f1 - cfg.f0))
    f_lo, f_hi = min(cfg.f0, cfg.f1) - pad, max(cfg.f0, cfg.f1) + pad

    fig, axes = plt.subplots(1, len(bit_indices), figsize=(5.8 * len(bit_indices), 4.3), squeeze=False)
    for ax, i in zip(axes[0], bit_indices):
        d = result.bit_decisions[i]
        freqs, mags = bit_spectrum(bit_segment(signal, sample_rate, d.start_time, cfg.bit_duration), sample_rate)
        mag_f0, mag_f1 = tone_magnitudes(freqs, mags, cfg.f0, cfg.f1)
        shown = (freqs >= f_lo) & (freqs <= f_hi)
        ax.plot(freqs[shown], mags[shown], color=COLOR_F0, linewidth=1.6)
        ax.axvline(cfg.f0, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"f0={cfg.f0:.0f} Hz ({mag_f0:.3f})")
        ax.axvline(cfg.f1, color=COLOR_ERROR, linestyle="--", linewidth=1.2, label=f"f1={cfg.f1:.0f} Hz ({mag_f1:.3f})")

        status = {True: "OK", False: "ERROR", None: "DECODED"}[d.is_correct]
        expected = f"exp={d.expected}, " if d.expected is not None else ""
        ax.set_title(f"Bit #{i + 1} [{status}] ({expected}dec={d.decision}, margin={d.margin_db:.1f} dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized FFT Magnitude")
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12)
    return _save(fig, path)


def plot_trial(signal: np.ndarray, sample_rate: int, result: DecodeResult, out_dir: str | Path, prefix: str) -> list[Path]:
    """All four diagnostic plots for one recording.

    The spectrum plot shows the first correct bit next to the first failed bit,
    or, if every bit is correct, the first '0' next to the first '1'.
    """
    def first(condition) -> int | None:
        return next((d.index for d in result.bit_decisions if condition(d)), None)

    if first(lambda d: d.is_correct is False) is not None:
        pair = (first(lambda d: d.is_correct is not False), first(lambda d: d.is_correct is False))
    else:
        pair = (first(lambda d: d.decision == "0"), first(lambda d: d.decision == "1"))
    selected = sorted({i for i in pair if i is not None})

    out, cfg = Path(out_dir), result.config
    return [
        plot_waveform(signal, sample_rate, result, out / f"{prefix}_waveform.png", f"{prefix}: Waveform & Bit Boundaries"),
        plot_spectrogram(signal, sample_rate, result, out / f"{prefix}_spectrogram.png",
                         f"{prefix}: Spectrogram (f0={cfg.f0:.0f} Hz, f1={cfg.f1:.0f} Hz)"),
        plot_bit_decisions(result, out / f"{prefix}_bit_decisions.png", f"{prefix}: Per-Bit Spectral Magnitudes"),
        plot_bit_spectra(signal, sample_rate, result, selected, out / f"{prefix}_bit_spectra.png",
                         f"{prefix}: Labelled Bit-Interval Spectra"),
    ]


def plot_experiment_performance(records: Sequence[TrialRecord], path: str | Path, title: str, xlabel: str) -> Path:
    """BER (%) and mean decision margin (dB) per setting: mean ± std line plus every individual trial."""
    by_setting: dict[str, list[TrialRecord]] = defaultdict(list)
    for r in records:
        by_setting[r.setting].append(r)
    settings = list(by_setting)
    x = np.arange(len(settings))

    fig, (ax_ber, ax_margin) = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)
    panels = [
        (ax_ber, lambda r: 100.0 * r.ber, COLOR_ERROR, "Bit Error Rate (%)"),
        (ax_margin, lambda r: r.mean_margin_db, COLOR_F0, "Mean Decision Margin (dB)"),
    ]
    for ax, value, color, label in panels:
        per_setting = [[value(r) for r in by_setting[s]] for s in settings]
        means = [np.mean(v) for v in per_setting]
        ax.errorbar(x, means, yerr=[np.std(v) for v in per_setting], fmt="o-", color=color,
                    linewidth=2.0, markersize=7, capsize=5, label="Mean ± std over trials")
        for xi, vals in zip(x, per_setting):  # every trial, including failed ones
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(xi + np.asarray(jitter), vals, color="#333333", alpha=0.75, s=32, zorder=5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax_ber.set_ylim(-2.0, max(10.0, max(100.0 * r.ber for r in records) + 8.0))
    ax_ber.set_title(title)
    ax_margin.axhline(0.0, color="#555555", linewidth=0.8)
    ax_margin.set_xlabel(xlabel)
    ax_margin.set_xticks(x)
    ax_margin.set_xticklabels(settings, rotation=12, ha="right")
    return _save(fig, path)


def plot_tone_magnitudes(results_by_setting: dict[str, list[DecodeResult]], path: str | Path, title: str) -> Path:
    """Mean received magnitude of the f0 tone (on '0' bits) and the f1 tone (on '1' bits) per setting.

    Shows frequency-dependent effects, e.g. an obstacle weakening 2 kHz more than 1 kHz,
    or in-band interference adding energy to one tone.
    """
    settings = list(results_by_setting)
    mean_f0, mean_f1 = [], []
    for s in settings:
        bits = [d for r in results_by_setting[s] for d in r.bit_decisions]
        mean_f0.append(np.mean([d.mag_f0 for d in bits if (d.expected or d.decision) == "0"] or [0.0]))
        mean_f1.append(np.mean([d.mag_f1 for d in bits if (d.expected or d.decision) == "1"] or [0.0]))

    x = np.arange(len(settings))
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(x - 0.175, mean_f0, width=0.35, color=COLOR_F0, label="f0 tone magnitude on '0' bits")
    ax.bar(x + 0.175, mean_f1, width=0.35, color=COLOR_F1, label="f1 tone magnitude on '1' bits")
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=12, ha="right")
    ax.set_xlabel("Setting")
    ax.set_ylabel("Mean Received Spectral Magnitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)
