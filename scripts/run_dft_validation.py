#!/usr/bin/env python3
"""Run direct DFT validation and generate Fourier concept plots (Slide 13).

Outputs labelled plots into `results/plots/dft/`:
  1. 01_dft_vs_fft_validation.png — Direct DFT vs numpy.fft.fft + residual error
  2. 02_bins_magnitude_phase.png  — Frequency bins, magnitude, and phase of 2-tone signal
  3. 03_spectral_leakage.png      — Coherent vs non-coherent sampling & Hann windowing
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


def generate_validation_plots(output_dir: Path) -> list[Path]:
    """Run all DFT validations and save labelled presentation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # ------------------------------------------------------------------
    # 1. Direct DFT vs numpy.fft.fft across short signal lengths
    # ------------------------------------------------------------------
    fs_short = 16_000
    sizes = [64, 128, 256, 512]
    reports = []
    last_x_direct = None
    last_x_fft = None

    print("=== Direct DFT vs. numpy.fft.fft Validation ===")
    print(f"{'N':>6} | {'Max |ΔX|':>12} | {'Max |ΔMag|':>12} | {'Max |ΔPhase|':>12} | {'DFT (ms)':>9} | {'FFT (ms)':>9}")
    print("-" * 74)
    for n in sizes:
        t = np.arange(n, dtype=np.float64) / fs_short
        sig = 0.8 * np.cos(2 * np.pi * 1000.0 * t) + 0.5 * np.sin(2 * np.pi * 2000.0 * t)
        rep, xd, xf = validate_dft_against_fft(sig, fs_short)
        reports.append(rep)
        last_x_direct, last_x_fft = xd, xf
        print(
            f"{rep.n_samples:6d} | {rep.max_complex_error:12.3e} | "
            f"{rep.max_magnitude_error:12.3e} | {rep.max_phase_error_rad:12.3e} | "
            f"{rep.direct_time_ms:9.3f} | {rep.fft_time_ms:9.3f}"
        )

    assert last_x_direct is not None and last_x_fft is not None
    freqs, mag_d, _ = single_sided_spectrum(last_x_direct, fs_short)
    _, mag_f, _ = single_sided_spectrum(last_x_fft, fs_short)
    residual = np.abs(last_x_direct[: freqs.size] - last_x_fft[: freqs.size]) + 1e-18

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(freqs, mag_f, label="numpy.fft.fft (O(N log N))", linewidth=2.2, color="#1f77b4")
    ax1.plot(freqs, mag_d, "--", label="Direct DFT (O(N²))", linewidth=1.5, color="#d62728")
    ax1.set_ylabel("Normalized Magnitude")
    ax1.set_title(f"Direct DFT vs. Library FFT Validation (N = {sizes[-1]}, fs = {fs_short} Hz)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(freqs, residual, color="#2ca02c", linewidth=1.2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Absolute Error |X_dft - X_fft|")
    ax2.set_title(f"Complex Spectral Residual (max error = {reports[-1].max_complex_error:.2e})")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    p1 = output_dir / "01_dft_vs_fft_validation.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(p1)

    # ------------------------------------------------------------------
    # 2. Frequency bins, magnitude, and phase (1000 Hz cos + 2000 Hz sin)
    # ------------------------------------------------------------------
    fs_bin = 8_000
    n_bin = 128  # bin spacing Δf = 8000 / 128 = 62.5 Hz -> 1000 Hz = bin 16, 2000 Hz = bin 32
    t_bin = np.arange(n_bin, dtype=np.float64) / fs_bin
    sig_bin = 1.0 * np.cos(2 * np.pi * 1000.0 * t_bin) + 0.6 * np.cos(
        2 * np.pi * 2000.0 * t_bin - np.pi / 2.0
    )
    spec_bin = direct_dft(sig_bin)
    f_b, mag_b, phase_b = single_sided_spectrum(spec_bin, fs_bin)
    # Mask phase on zero-energy bins so phase of active tones is crystal clear
    phase_masked = np.where(mag_b > 0.05, np.degrees(phase_b), np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.stem(f_b, mag_b, basefmt=" ")
    ax1.annotate("Bin 16: 1000 Hz\n|X| = 1.00, ∠ = 0°", xy=(1000, 1.0), xytext=(1250, 0.92),
                 arrowprops={"arrowstyle": "->"})
    ax1.annotate("Bin 32: 2000 Hz\n|X| = 0.60, ∠ = -90°", xy=(2000, 0.6), xytext=(2250, 0.65),
                 arrowprops={"arrowstyle": "->"})
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"DFT Frequency Bins, Magnitude & Phase (N = {n_bin}, fs = {fs_bin} Hz, Δf = {fs_bin/n_bin:.1f} Hz)")
    ax1.grid(True, alpha=0.3)

    ax2.stem(f_b, np.nan_to_num(phase_masked, nan=0.0), linefmt="C1-", markerfmt="C1o", basefmt=" ")
    ax2.set_ylim(-120, 120)
    ax2.set_yticks([-90, -45, 0, 45, 90])
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title("Phase at Active Frequency Bins (cos -> 0°, sin -> -90°)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p2 = output_dir / "02_bins_magnitude_phase.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(p2)

    # ------------------------------------------------------------------
    # 3. Spectral leakage: integer vs non-integer cycles & Hann window
    # ------------------------------------------------------------------
    fs_leak = 8_000
    n_leak = 256  # Δf = 31.25 Hz -> bin 32 = 1000.0 Hz (coherent), 1015.625 Hz = bin 32.5 (max leakage)
    t_leak = np.arange(n_leak, dtype=np.float64) / fs_leak
    sig_coherent = np.sin(2 * np.pi * 1000.0 * t_leak)
    sig_leaky = np.sin(2 * np.pi * 1015.625 * t_leak)
    hann = np.hanning(n_leak)

    f_l, m_coh, _ = single_sided_spectrum(direct_dft(sig_coherent), fs_leak)
    _, m_leak_rect, _ = single_sided_spectrum(direct_dft(sig_leaky), fs_leak)
    _, m_leak_hann, _ = single_sided_spectrum(direct_dft(sig_leaky * hann), fs_leak)
    m_leak_hann *= 2.0  # coherent gain compensation for Hann window

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.semilogy(f_l, m_coh + 1e-6, "o-", markersize=4, label="1000.0 Hz (integer cycles, Rect)")
    ax.semilogy(f_l, m_leak_rect + 1e-6, "s-", markersize=4, label="1015.6 Hz (half-bin offset, Rect -> severe leakage)")
    ax.semilogy(f_l, m_leak_hann + 1e-6, "^-", markersize=4, label="1015.6 Hz (half-bin offset, Hann -> low sidelobes)")
    ax.set_xlim(500, 1600)
    ax.set_ylim(1e-5, 2.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Magnitude (log scale)")
    ax.set_title("Spectral Leakage: Coherent Bin Alignment vs. Non-Integer Cycles & Windowing")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    p3 = output_dir / "03_spectral_leakage.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved.append(p3)

    # ------------------------------------------------------------------
    # 4. Option A: Interval duration T_b vs frequency resolution Δf = 1/T_b
    # ------------------------------------------------------------------
    fs_res = 48_000
    f0_demo, f1_demo = 1_000.0, 1_100.0  # Δf_tones = 100 Hz
    durations = [0.10, 0.02, 0.005]      # Δf_bin = 10 Hz, 50 Hz, 200 Hz

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, tb in zip(axes, durations):
        n_s = int(round(fs_res * tb))
        t_s = np.arange(n_s, dtype=np.float64) / fs_res
        # Zero-pad to 8192 for smooth DTFT envelope alongside actual DFT bins
        sig_0 = np.sin(2 * np.pi * f0_demo * t_s)
        sig_1 = np.sin(2 * np.pi * f1_demo * t_s)
        n_pad = max(8192, n_s * 4)
        f_pad = np.fft.rfftfreq(n_pad, d=1.0 / fs_res)
        m0 = np.abs(np.fft.rfft(sig_0, n=n_pad)) / (n_s / 2.0)
        m1 = np.abs(np.fft.rfft(sig_1, n=n_pad)) / (n_s / 2.0)

        ax.plot(f_pad, m0, label="Bit '0' (1000 Hz)", color="#1f77b4", linewidth=1.8)
        ax.plot(f_pad, m1, label="Bit '1' (1100 Hz)", color="#ff7f0e", linewidth=1.8)
        ax.axvline(f0_demo, color="#1f77b4", linestyle=":", alpha=0.7)
        ax.axvline(f1_demo, color="#ff7f0e", linestyle=":", alpha=0.7)
        ax.set_xlim(700, 1400)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"Tb = {tb*1000:.0f} ms  (Δf = {1.0/tb:.0f} Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Normalized Magnitude")
    fig.suptitle(
        "Option A Detection Trade-off: Shorter Bit Duration Widens Main Lobes (Δf = 1/Tb)",
        fontsize=12,
    )
    fig.tight_layout()
    p4 = output_dir / "04_duration_vs_resolution.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    saved.append(p4)

    return saved


def main() -> int:
    out_dir = PLOTS_DIR / "dft"
    saved = generate_validation_plots(out_dir)
    print("\nSaved DFT validation plots:")
    for path in saved:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
