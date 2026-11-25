"""Standardized, fully labelled visualization utilities for Acoustic Messenger.

Generates publication-ready figures required by Slide 14:
  - Time-domain waveforms with bit-boundary overlays
  - STFT spectrograms (showing tone tracks, interference, and room reverb tails)
  - Per-bit magnitude comparison charts (Slide 5 format)
  - Labelled per-bit FFT spectra (highlighting successful vs. failed bits)
  - Controlled experiment performance curves (BER & SNR vs. varied factor)
  - Frequency-dependent attenuation plot (for Exp 4: Distance & Position)
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

from src.config import DecodeResult, ModulationConfig, TrialRecord
from src.rx import measure_bit_spectrum


def plot_waveform_with_boundaries(
    signal: np.ndarray,
    sample_rate: int,
    result: DecodeResult,
    output_path: str | Path,
    title: str = "Received Acoustic Waveform & Bit Boundaries",
) -> Path:
    """Plot time-domain waveform with bit-interval dividers and pass/fail markers."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = np.arange(signal.size, dtype=np.float64) / float(sample_rate)
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(t, signal, color="#1f77b4", linewidth=0.6, alpha=0.85, label="Recorded signal")

    y_lim = max(float(np.max(np.abs(signal))) * 1.18, 0.05) if signal.size else 1.0
    ax.set_ylim(-y_lim, y_lim)

    for idx, dec in enumerate(result.bit_decisions):
        ax.axvline(dec.start_time, color="#555555", linestyle=":", linewidth=0.7, alpha=0.6)
        mid = 0.5 * (dec.start_time + dec.end_time)
        if dec.is_correct is False:
            ax.axvspan(dec.start_time, dec.end_time, color="#d62728", alpha=0.18)
            txt_color = "#b00020"
        elif dec.is_correct is True:
            txt_color = "#1b5e20"
        else:
            txt_color = "#222222"
        # Show bit labels cleanly without crowding for <= 64 bits
        if len(result.bit_decisions) <= 64:
            ax.text(
                mid,
                y_lim * 0.88,
                dec.decision,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color=txt_color,
            )

    if result.bit_decisions:
        ax.axvline(
            result.bit_decisions[-1].end_time,
            color="#555555",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )

    ber_txt = f" | BER = {result.ber * 100:.2f}%" if result.ber is not None else ""
    ax.set_title(f"{title} (sync: {result.sync_method}, start = {result.start_time:.3f} s{ber_txt})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    config: ModulationConfig,
    output_path: str | Path,
    title: str = "Acoustic Transmission Spectrogram (STFT)",
) -> Path:
    """Plot time-frequency spectrogram with horizontal markers at f0 and f1."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    nperseg = min(signal.size, max(256, int(round(sample_rate * min(0.04, config.bit_duration * 0.6)))))
    noverlap = int(nperseg * 0.75)
    f, t, sxx = spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann")
    sxx_db = 10.0 * np.log10(sxx + 1e-16)

    max_freq = min(sample_rate / 2.0, max(config.f0, config.f1) * 1.8 + 400.0)
    f_mask = f <= max_freq

    fig, ax = plt.subplots(figsize=(11, 4.4))
    vmin = float(np.percentile(sxx_db[f_mask], 15))
    vmax = float(np.max(sxx_db[f_mask]))
    mesh = ax.pcolormesh(t, f[f_mask], sxx_db[f_mask], shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="Power Spectral Density (dB)")

    ax.axhline(config.f0, color="#ffffff", linestyle="--", linewidth=1.1, alpha=0.85, label=f"Bit '0' (f0 = {config.f0:.0f} Hz)")
    ax.axhline(config.f1, color="#ffeb3b", linestyle="--", linewidth=1.1, alpha=0.85, label=f"Bit '1' (f1 = {config.f1:.0f} Hz)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_bit_decisions(
    result: DecodeResult,
    output_path: str | Path,
    title: str = "Per-Bit Spectral Magnitudes Near f0 vs. f1",
) -> Path:
    """Plot Slide-5 style per-bit magnitude comparison for f0 and f1."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    indices = np.arange(1, len(result.bit_decisions) + 1)
    m0 = np.array([d.mag_f0 for d in result.bit_decisions], dtype=np.float64)
    m1 = np.array([d.mag_f1 for d in result.bit_decisions], dtype=np.float64)

    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.3))
    ax.bar(indices - width / 2, m0, width=width, label=f"Near f0 ({result.config.f0:.0f} Hz -> '0')", color="#1f77b4")
    ax.bar(indices + width / 2, m1, width=width, label=f"Near f1 ({result.config.f1:.0f} Hz -> '1')", color="#ff7f0e")

    # Highlight bit errors clearly
    err_indices = [d.index + 1 for d in result.bit_decisions if d.is_correct is False]
    y_top = max(float(np.max(np.concatenate([m0, m1]))) * 1.15, 1e-3) if indices.size else 1.0
    for ei in err_indices:
        ax.axvspan(ei - 0.5, ei + 0.5, color="#d62728", alpha=0.16)
        ax.plot(ei, y_top * 0.94, marker="v", color="#d62728", markersize=7)

    if err_indices:
        ax.plot([], [], marker="v", linestyle="None", color="#d62728", label="Bit error")

    ber_txt = f" | BER = {result.ber * 100:.2f}% ({result.bit_errors}/{len(indices)})" if result.ber is not None else ""
    ax.set_title(f"{title}{ber_txt}")
    ax.set_xlabel("Bit Index (1-based)")
    ax.set_ylabel("Spectral Magnitude")
    ax.set_ylim(0.0, y_top)
    if indices.size <= 40:
        ax.set_xticks(indices)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_bit_spectra_comparison(
    signal: np.ndarray,
    sample_rate: int,
    result: DecodeResult,
    bit_indices: Sequence[int],
    output_path: str | Path,
    title: str = "Labelled Bit-Interval FFT Spectra",
    window: str = "hann",
    guard_fraction: float = 0.05,
) -> Path:
    """Plot labelled FFT magnitude spectra for selected bit indices (e.g., pass vs. fail)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    valid_indices = [i for i in bit_indices if 0 <= i < len(result.bit_decisions)]
    if not valid_indices:
        valid_indices = [0]

    cfg = result.config
    n_panels = len(valid_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.8 * n_panels, 4.3), squeeze=False)
    axes_flat = axes[0]

    f_lo = max(0.0, min(cfg.f0, cfg.f1) - max(400.0, 0.5 * abs(cfg.f1 - cfg.f0)))
    f_hi = min(sample_rate / 2.0, max(cfg.f0, cfg.f1) + max(400.0, 0.5 * abs(cfg.f1 - cfg.f0)))

    for ax, bit_idx in zip(axes_flat, valid_indices):
        dec = result.bit_decisions[bit_idx]
        trim = guard_fraction * cfg.bit_duration
        s0 = max(0, int(round((dec.start_time + trim) * sample_rate)))
        s1 = min(signal.size, int(round((dec.end_time - trim) * sample_rate)))
        seg = signal[s0:s1]

        mag_f0, mag_f1, snr_db, freqs, mags = measure_bit_spectrum(
            seg, sample_rate, cfg.f0, cfg.f1, window=window
        )
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        ax.plot(freqs[mask], mags[mask], color="#1f77b4", linewidth=1.6)
        ax.axvline(cfg.f0, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"f0={cfg.f0:.0f} Hz ({mag_f0:.3f})")
        ax.axvline(cfg.f1, color="#d62728", linestyle="--", linewidth=1.2, label=f"f1={cfg.f1:.0f} Hz ({mag_f1:.3f})")

        status = "OK" if dec.is_correct else ("ERROR" if dec.is_correct is False else "DECODED")
        exp_txt = f"exp={dec.expected}, " if dec.expected is not None else ""
        ax.set_title(f"Bit #{bit_idx + 1} [{status}] ({exp_txt}dec={dec.decision}, SNR={snr_db:.1f} dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized FFT Magnitude")
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_experiment_performance(
    records: Sequence[TrialRecord],
    output_path: str | Path,
    title: str = "Controlled Experiment Performance",
    xlabel: str = "Experiment Setting",
    setting_order: Sequence[str] | None = None,
) -> Path:
    """Plot BER (%) and estimated SNR (dB) across settings with individual trials."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[TrialRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.setting].append(rec)

    if setting_order is not None:
        settings = [s for s in setting_order if s in grouped] + [s for s in grouped if s not in setting_order]
    else:
        settings = list(grouped.keys())

    x = np.arange(len(settings), dtype=np.float64)
    ber_means = [float(np.mean([r.ber * 100.0 for r in grouped[s]])) for s in settings]
    ber_stds = [float(np.std([r.ber * 100.0 for r in grouped[s]])) for s in settings]

    snr_means = []
    snr_stds = []
    has_snr = False
    for s in settings:
        vals = [r.mean_snr_db for r in grouped[s] if r.mean_snr_db is not None and np.isfinite(r.mean_snr_db)]
        if vals:
            has_snr = True
            snr_means.append(float(np.mean(vals)))
            snr_stds.append(float(np.std(vals)))
        else:
            snr_means.append(np.nan)
            snr_stds.append(0.0)

    n_rows = 2 if has_snr else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.5, 4.0 * n_rows), sharex=True, squeeze=False)
    ax_ber = axes[0, 0]

    ax_ber.plot(x, ber_means, "o-", color="#d62728", linewidth=2.0, markersize=7, label="Mean BER (%)")
    ax_ber.errorbar(x, ber_means, yerr=ber_stds, fmt="none", ecolor="#d62728", capsize=5, alpha=0.7)

    # Plot every individual trial (retaining failed trials visually)
    for xi, s in zip(x, settings):
        trial_bers = [r.ber * 100.0 for r in grouped[s]]
        jitter = np.linspace(-0.08, 0.08, len(trial_bers)) if len(trial_bers) > 1 else [0.0]
        ax_ber.scatter(
            xi + np.asarray(jitter),
            trial_bers,
            color="#333333",
            alpha=0.75,
            s=32,
            zorder=5,
            label="Individual trials" if xi == 0 else None,
        )

    ax_ber.set_ylabel("Bit Error Rate (%)")
    ax_ber.set_ylim(-2.0, max(10.0, max(ber_means) + max(ber_stds) + 8.0))
    ax_ber.set_title(title)
    ax_ber.legend()
    ax_ber.grid(True, alpha=0.3)

    if has_snr:
        ax_snr = axes[1, 0]
        ax_snr.plot(x, snr_means, "s-", color="#1f77b4", linewidth=2.0, markersize=7, label="Mean Estimated SNR (dB)")
        ax_snr.errorbar(x, snr_means, yerr=snr_stds, fmt="none", ecolor="#1f77b4", capsize=5, alpha=0.7)
        for xi, s in zip(x, settings):
            vals = [r.mean_snr_db for r in grouped[s] if r.mean_snr_db is not None and np.isfinite(r.mean_snr_db)]
            if vals:
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
                ax_snr.scatter(xi + np.asarray(jitter), vals, color="#333333", alpha=0.75, s=32, zorder=5)
        ax_snr.set_ylabel("Estimated SNR (dB)")
        ax_snr.set_xlabel(xlabel)
        ax_snr.legend()
        ax_snr.grid(True, alpha=0.3)
    else:
        ax_ber.set_xlabel(xlabel)

    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(settings, rotation=12, ha="right")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_frequency_attenuation(
    setting_to_results: dict[str, list[DecodeResult]],
    output_path: str | Path,
    title: str = "Frequency-Dependent Acoustic Attenuation Across Positions (f0 vs. f1)",
    setting_order: Sequence[str] | None = None,
) -> Path:
    """Compare mean received magnitude of bit '0' (f0) vs. bit '1' (f1) across positions.

    Highlights why obstacles / off-axis paths often attenuate the higher frequency
    tone (2,000 Hz) more strongly than the lower frequency tone (1,000 Hz).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    settings = (
        [s for s in setting_order if s in setting_to_results]
        if setting_order is not None
        else list(setting_to_results.keys())
    )

    mean_f0_on_zero = []
    mean_f1_on_one = []
    for s in settings:
        res_list = setting_to_results[s]
        vals0 = [
            d.mag_f0
            for r in res_list
            for d in r.bit_decisions
            if (d.expected == "0" if d.expected is not None else d.decision == "0")
        ]
        vals1 = [
            d.mag_f1
            for r in res_list
            for d in r.bit_decisions
            if (d.expected == "1" if d.expected is not None else d.decision == "1")
        ]
        mean_f0_on_zero.append(float(np.mean(vals0)) if vals0 else 0.0)
        mean_f1_on_one.append(float(np.mean(vals1)) if vals1 else 0.0)

    x = np.arange(len(settings), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(x - width / 2, mean_f0_on_zero, width=width, color="#1f77b4", label="Received 1,000 Hz magnitude (bit '0')")
    ax.bar(x + width / 2, mean_f1_on_one, width=width, color="#ff7f0e", label="Received 2,000 Hz magnitude (bit '1')")

    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=12, ha="right")
    ax.set_xlabel("Acoustic Position / Path Setting")
    ax.set_ylabel("Mean Received Spectral Magnitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
