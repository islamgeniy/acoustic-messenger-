#!/usr/bin/env python3
"""Run direct DFT validation and generate Fourier concept plots (Slide 13).

Outputs labelled plots into `results/plots/dft/`:
  1. 01_dft_vs_fft_validation.png  — Direct DFT vs numpy.fft.fft + residual error
  2. 02_bins_magnitude_phase.png   — Frequency bins, magnitude, and phase of a 2-tone signal
  3. 03_spectral_leakage.png       — Coherent vs non-coherent sampling & Hann windowing
  4. 04_duration_vs_resolution.png — Bit duration T_b vs frequency resolution (Δf = 1/T_b)
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PLOTS_DIR
from src.dft import direct_dft, single_sided_spectrum, validate_dft_against_fft


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_dft_vs_fft(output_dir: Path) -> Path:
    """Compare direct DFT with numpy.fft.fft on short signals and plot the residual."""
    fs = 16_000
    sizes = [64, 128, 256, 512]

    print("=== Direct DFT vs. numpy.fft.fft Validation ===")
    print(f"{'N':>6} | {'Max |ΔX|':>12} | {'Max |ΔMag|':>12} | {'Max |ΔPhase|':>12} | {'DFT (ms)':>9} | {'FFT (ms)':>9}")
    print("-" * 74)
    for n in sizes:
        t = np.arange(n) / fs
        x = 0.8 * np.cos(2 * np.pi * 1000.0 * t) + 0.5 * np.sin(2 * np.pi * 2000.0 * t)
        report, x_direct, x_fft = validate_dft_against_fft(x, fs)
        print(
            f"{report.n_samples:6d} | {report.max_complex_error:12.3e} | "
            f"{report.max_magnitude_error:12.3e} | {report.max_phase_error_rad:12.3e} | "
            f"{report.direct_time_ms:9.3f} | {report.fft_time_ms:9.3f}"
        )

    # Plot the largest N from the loop above
    freqs, mag_direct, _ = single_sided_spectrum(x_direct, fs)
    _, mag_fft, _ = single_sided_spectrum(x_fft, fs)
    residual = np.abs(x_direct[: freqs.size] - x_fft[: freqs.size]) + 1e-18

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(freqs, mag_fft, label="numpy.fft.fft (O(N log N))", linewidth=2.2, color="#1f77b4")
    ax1.plot(freqs, mag_direct, "--", label="Direct DFT (O(N²))", linewidth=1.5, color="#d62728")
    ax1.set_ylabel("Normalized Magnitude")
    ax1.set_title(f"Direct DFT vs. Library FFT Validation (N = {sizes[-1]}, fs = {fs} Hz)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(freqs, residual, color="#2ca02c", linewidth=1.2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Absolute Error |X_dft - X_fft|")
    ax2.set_title(f"Complex Spectral Residual (max error = {report.max_complex_error:.2e})")
    ax2.grid(True, which="both", alpha=0.3)
    return _save(fig, output_dir / "01_dft_vs_fft_validation.png")


def plot_bins_magnitude_phase(output_dir: Path) -> Path:
    """Show DFT bins, magnitude and phase of cos(1000 Hz) + 0.6·sin(2000 Hz)."""
    fs, n = 8_000, 128  # bin spacing fs/N = 62.5 Hz -> 1000 Hz = bin 16, 2000 Hz = bin 32
    t = np.arange(n) / fs
    x = 1.0 * np.cos(2 * np.pi * 1000.0 * t) + 0.6 * np.sin(2 * np.pi * 2000.0 * t)
    freqs, mag, phase = single_sided_spectrum(direct_dft(x), fs)
    active = mag > 0.05  # show phase only where there is energy

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.stem(freqs, mag, basefmt=" ")
    for m in np.flatnonzero(active):
        label = f"Bin {m}: {freqs[m]:.0f} Hz\n|X| = {mag[m]:.2f}, ∠ = {round(float(np.degrees(phase[m])))}°"
        ax1.annotate(label, xy=(freqs[m], mag[m]), xytext=(freqs[m] + 250, mag[m] * 0.92),
                     arrowprops={"arrowstyle": "->"})
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"DFT Frequency Bins, Magnitude & Phase (N = {n}, fs = {fs} Hz, Δf = {fs / n:.1f} Hz)")
    ax1.grid(True, alpha=0.3)

    ax2.stem(freqs, np.where(active, np.degrees(phase), 0.0), linefmt="C1-", markerfmt="C1o", basefmt=" ")
    ax2.set_ylim(-120, 120)
    ax2.set_yticks([-90, -45, 0, 45, 90])
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title("Phase at Active Frequency Bins (cos -> 0°, sin -> -90°)")
    ax2.grid(True, alpha=0.3)
    return _save(fig, output_dir / "02_bins_magnitude_phase.png")


def plot_spectral_leakage(output_dir: Path) -> Path:
    """Compare a bin-centred tone, a half-bin-offset tone, and the same tone with a Hann window."""
    fs, n = 8_000, 256  # Δf = 31.25 Hz -> 1000 Hz = bin 32 (coherent), 1015.625 Hz = bin 32.5
    t = np.arange(n) / fs
    coherent = np.sin(2 * np.pi * 1000.0 * t)
    leaky = np.sin(2 * np.pi * 1015.625 * t)

    freqs, m_coherent, _ = single_sided_spectrum(direct_dft(coherent), fs)
    _, m_rect, _ = single_sided_spectrum(direct_dft(leaky), fs)
    _, m_hann, _ = single_sided_spectrum(direct_dft(leaky * np.hanning(n)), fs)
    m_hann *= 2.0  # the Hann window has coherent gain 0.5

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.semilogy(freqs, m_coherent + 1e-6, "o-", markersize=4, label="1000.0 Hz (integer cycles, Rect)")
    ax.semilogy(freqs, m_rect + 1e-6, "s-", markersize=4, label="1015.6 Hz (half-bin offset, Rect -> severe leakage)")
    ax.semilogy(freqs, m_hann + 1e-6, "^-", markersize=4, label="1015.6 Hz (half-bin offset, Hann -> low sidelobes)")
    ax.set_xlim(500, 1600)
    ax.set_ylim(1e-5, 2.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Magnitude (log scale)")
    ax.set_title("Spectral Leakage: Coherent Bin Alignment vs. Non-Integer Cycles & Windowing")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, output_dir / "03_spectral_leakage.png")


def plot_duration_vs_resolution(output_dir: Path) -> Path:
    """Option A trade-off: shorter bit duration T_b widens the spectral main lobe (Δf = 1/T_b)."""
    fs = 48_000
    f0, f1 = 1_000.0, 1_100.0
    durations = [0.10, 0.02, 0.005]  # Δf = 10 Hz, 50 Hz, 200 Hz

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, tb in zip(axes, durations):
        n = int(round(fs * tb))
        t = np.arange(n) / fs
        n_pad = max(8192, 4 * n)  # zero-padding draws the smooth DTFT envelope
        f_pad = np.fft.rfftfreq(n_pad, d=1.0 / fs)
        for f, color in ((f0, "#1f77b4"), (f1, "#ff7f0e")):
            mag = np.abs(np.fft.rfft(np.sin(2 * np.pi * f * t), n=n_pad)) / (n / 2.0)
            bit = "0" if f == f0 else "1"
            ax.plot(f_pad, mag, label=f"Bit '{bit}' ({f:.0f} Hz)", color=color, linewidth=1.8)
            ax.axvline(f, color=color, linestyle=":", alpha=0.7)
        ax.set_xlim(700, 1400)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"Tb = {tb * 1000:.0f} ms  (Δf = {1.0 / tb:.0f} Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Normalized Magnitude")
    fig.suptitle("Option A Detection Trade-off: Shorter Bit Duration Widens Main Lobes (Δf = 1/Tb)", fontsize=12)
    return _save(fig, output_dir / "04_duration_vs_resolution.png")


def main() -> int:
    out_dir = PLOTS_DIR / "dft"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        plot_dft_vs_fft(out_dir),
        plot_bins_magnitude_phase(out_dir),
        plot_spectral_leakage(out_dir),
        plot_duration_vs_resolution(out_dir),
    ]
    print("\nSaved DFT validation plots:")
    for path in saved:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
