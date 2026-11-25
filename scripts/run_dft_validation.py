#!/usr/bin/env python3
"""Run direct DFT validation and generate the Fourier concept plots (Slide 13).

Figures 02, 03 and 05 recreate the examples from lecture SaS-w11-3 (Lyons ch. 3) with our
own direct DFT, so every number can be checked against the slides.

Outputs labelled plots into `results/plots/dft/`:
  1. 01_dft_vs_fft_validation.png  — Direct DFT vs numpy.fft.fft + residual error
  2. 02_bins_magnitude_phase.png   — Lecture DFT Example 1: bins, magnitude, phase, real/imag parts
  3. 03_spectral_leakage.png       — Lecture leakage example (3.0 vs 3.4 cycles) and Hanning windowing
  4. 04_duration_vs_resolution.png — Bit duration T_b vs frequency resolution (Δf = 1/T_b)
  5. 05_window_functions.png       — Rectangular / triangular / Hanning / Hamming windows and their spectra
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
from src.dft import dft_frequencies, direct_dft, single_sided_spectrum, validate_dft_against_fft, window


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
    """SaS-w11-3 DFT Example 1: fs = 8 kHz, N = 8,
    x(n) = sin(2π·1000·n·ts) + 0.5·sin(2π·2000·n·ts + 3π/4).

    Expected (slides): |X(1)| = 4 at -90°, |X(2)| = 2 at +45°, mirror images at m = 7, 6.
    """
    fs, n_samples = 8_000, 8
    n = np.arange(n_samples)
    x = np.sin(2 * np.pi * 1000 * n / fs) + 0.5 * np.sin(2 * np.pi * 2000 * n / fs + 3 * np.pi / 4)
    X = direct_dft(x)
    m = np.arange(n_samples)
    f_khz = dft_frequencies(n_samples, fs) / 1000.0
    active = np.abs(X) > 1e-9  # phase of an all-zero bin is meaningless
    phase_deg = np.where(active, np.degrees(np.angle(X)), 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    panels = [
        (axes[0, 0], np.abs(X), "Magnitude |X(m)|"),
        (axes[0, 1], X.real, "Real part of X(m)"),
        (axes[1, 0], phase_deg, "Phase of X(m) (degrees)"),
        (axes[1, 1], X.imag, "Imaginary part of X(m)"),
    ]
    for ax, values, label in panels:
        ax.stem(f_khz, values, basefmt="k-")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    for i in np.flatnonzero(active & (m < n_samples // 2)):  # bins above N/2 only mirror these
        axes[0, 0].annotate(f"m={m[i]}: |X|={abs(X[i]):.0f}, ∠={round(float(phase_deg[i]))}°",
                            xy=(f_khz[i], abs(X[i])), xytext=(f_khz[i] + 0.15, abs(X[i]) * 0.85), fontsize=8)
    axes[1, 0].set_yticks([-90, -45, 0, 45, 90])
    for ax in axes[1]:
        ax.set_xlabel("Analysis frequency f(m) = m·fs/N (kHz)")
    fig.suptitle("Lecture DFT Example 1 (N = 8, fs = 8 kHz): sin(2π·1000t) + 0.5·sin(2π·2000t + 3π/4)\n"
                 "bins m = 5…7 mirror m = 3…1 (real input: |X(N-m)| = |X(m)|)")
    return _save(fig, output_dir / "02_bins_magnitude_phase.png")


def plot_spectral_leakage(output_dir: Path) -> Path:
    """SaS-w11-3 leakage and windowing examples, 64-point DFT.

    (a) 3.0 cycles -> one bin (|X(3)| = A·N/2 = 32); 3.4 cycles -> energy leaks into every bin;
        the Hanning window removes the far leakage.
    (b) A weak 0.1-amplitude 7-cycle tone beside the 3.4-cycle tone is buried by rectangular
        leakage but visible as a peak at m = 7 after Hanning windowing.
    """
    n_samples = 64
    n = np.arange(n_samples)
    m = np.arange(n_samples // 2)
    hann = window("hanning", n_samples)

    def tone(cycles: float, amp: float = 1.0) -> np.ndarray:
        return amp * np.sin(2 * np.pi * cycles * n / n_samples)

    def mag(x: np.ndarray) -> np.ndarray:  # first half of the bins (the rest mirror them)
        return np.abs(direct_dft(x))[: n_samples // 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(m, mag(tone(3.0)), "o-", markersize=4, label="3.0 cycles, rectangular (no leakage)")
    ax1.plot(m, mag(tone(3.4)), "s-", markersize=4, label="3.4 cycles, rectangular (leakage)")
    ax1.plot(m, mag(tone(3.4) * hann), "^-", markersize=4, label="3.4 cycles, Hanning window")
    ax1.set_title("(a) DFT leakage: integer vs. non-integer cycles in N = 64")

    weak_pair = tone(3.4) + tone(7.0, 0.1)
    ax2.plot(m, mag(weak_pair), "s-", markersize=4, label="rectangular window")
    ax2.plot(m, mag(weak_pair * hann), "^-", markersize=4, label="Hanning window")
    ax2.annotate("0.1-amplitude tone at m = 7", xy=(7, mag(weak_pair * hann)[7]), xytext=(10, 6),
                 arrowprops={"arrowstyle": "->"})
    ax2.set_title("(b) Windowing reveals a weak tone next to a strong one")

    for ax in (ax1, ax2):
        ax.set_xlabel("Bin m  (analysis frequency m·fs/N)")
        ax.set_ylabel("DFT output magnitude |X(m)|")
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
    return _save(fig, output_dir / "03_spectral_leakage.png")


def plot_duration_vs_resolution(output_dir: Path) -> Path:
    """Option A trade-off: shorter bit duration T_b widens the spectral main lobe (Δf = 1/T_b).

    A tone that lasts T_b seconds is a sinusoid times a rectangular pulse, so its spectrum is a
    sinc whose main lobe is 2/T_b wide (SaS-w9-2 Table 4.2; Lyons 3.13). Zero-padding only draws
    that curve more smoothly (Lyons 3.11). A longer bit also gives more processing gain:
    each doubling of N adds 3 dB of SNR in the tone's bin (Lyons 3.12).
    """
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


def plot_window_functions(output_dir: Path) -> Path:
    """The four lecture windows w(n) and their magnitude responses in dB (Lyons 3-27 ... 3-30)."""
    n_samples, n_pad = 64, 4096
    names = ["rectangular", "triangular", "hanning", "hamming"]
    bins = np.arange(n_pad // 2) * n_samples / n_pad  # frequency in units of fs/N

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(13, 4.6))
    for name in names:
        w = window(name, n_samples)
        response = np.abs(np.fft.rfft(w, n=n_pad))[: n_pad // 2]
        ax_t.plot(np.arange(n_samples), w, label=name.capitalize())
        ax_f.plot(bins, 20 * np.log10(response / response.max() + 1e-12), label=name.capitalize())

    ax_t.set_title(f"Window coefficients w(n), N = {n_samples}")
    ax_t.set_xlabel("n")
    ax_t.set_ylabel("w(n)")
    ax_f.set_title("Window magnitude response |W(m)| (0 dB = peak)")
    ax_f.set_xlabel("Frequency (multiples of fs/N)")
    ax_f.set_ylabel("dB")
    ax_f.set_xlim(0, 8)
    ax_f.set_ylim(-80, 3)
    for ax in (ax_t, ax_f):
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
    return _save(fig, output_dir / "05_window_functions.png")


def main() -> int:
    out_dir = PLOTS_DIR / "dft"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        plot_dft_vs_fft(out_dir),
        plot_bins_magnitude_phase(out_dir),
        plot_spectral_leakage(out_dir),
        plot_duration_vs_resolution(out_dir),
        plot_window_functions(out_dir),
    ]
    print("\nSaved DFT validation plots:")
    for path in saved:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
