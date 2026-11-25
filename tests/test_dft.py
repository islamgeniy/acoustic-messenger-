"""Unit tests for the direct DFT implementation (`src/dft.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from src.dft import (
    dft_frequencies,
    direct_dft,
    direct_idft,
    single_sided_spectrum,
    validate_dft_against_fft,
    window,
)


@pytest.mark.parametrize("n_samples", [16, 64, 128, 256])
def test_direct_dft_matches_numpy_fft(n_samples: int) -> None:
    rng = np.random.default_rng(42 + n_samples)
    x = rng.standard_normal(n_samples)

    x_direct = direct_dft(x)
    x_fft = np.fft.fft(x)

    assert np.max(np.abs(x_direct - x_fft)) < 1e-11


def test_frequency_bins_and_two_tone_peaks() -> None:
    fs = 8_000
    n = 128  # Δf = 62.5 Hz -> 1000 Hz at bin 16, 2000 Hz at bin 32
    t = np.arange(n, dtype=np.float64) / fs
    x = 0.9 * np.cos(2 * np.pi * 1000.0 * t) + 0.4 * np.sin(2 * np.pi * 2000.0 * t)

    freqs_full = dft_frequencies(n, fs)
    assert freqs_full[16] == pytest.approx(1000.0)
    assert freqs_full[32] == pytest.approx(2000.0)

    freqs, mag, phase = single_sided_spectrum(direct_dft(x), fs)
    assert freqs[16] == pytest.approx(1000.0)
    assert mag[16] == pytest.approx(0.9, abs=1e-10)
    assert phase[16] == pytest.approx(0.0, abs=1e-10)

    assert freqs[32] == pytest.approx(2000.0)
    assert mag[32] == pytest.approx(0.4, abs=1e-10)
    assert phase[32] == pytest.approx(-np.pi / 2.0, abs=1e-10)


def test_validate_dft_report_metrics() -> None:
    fs = 16_000
    n = 256
    t = np.arange(n, dtype=np.float64) / fs
    x = np.cos(2 * np.pi * 1000.0 * t) + 0.5 * np.sin(2 * np.pi * 2000.0 * t)

    report, _, x_fft = validate_dft_against_fft(x, fs)
    assert report.n_samples == n
    assert report.max_complex_error < 1e-10
    assert report.max_complex_error / np.max(np.abs(x_fft)) < 1e-12
    assert report.max_magnitude_error < 1e-10
    assert report.max_phase_error_rad < 1e-11


def test_lecture_dft_example_1() -> None:
    """SaS-w11-3 / Lyons Example 1: fs = 8 kHz, N = 8, x(n) = sin(2π·1000·n·ts) + 0.5·sin(2π·2000·n·ts + 3π/4)."""
    fs, n_samples = 8_000, 8
    n = np.arange(n_samples)
    x = np.sin(2 * np.pi * 1000 * n / fs) + 0.5 * np.sin(2 * np.pi * 2000 * n / fs + 3 * np.pi / 4)
    X = direct_dft(x)

    assert dft_frequencies(n_samples, fs)[1] == pytest.approx(1000.0)  # f_analysis(1) = 1·8000/8
    assert abs(X[1]) == pytest.approx(4.0)                     # A·N/2 = 1·8/2 (Lyons 3-17)
    assert np.degrees(np.angle(X[1])) == pytest.approx(-90.0)  # a sine is a cosine with -90° phase
    assert abs(X[2]) == pytest.approx(2.0)                     # 0.5·8/2
    assert np.degrees(np.angle(X[2])) == pytest.approx(45.0)   # 135° - 90°
    for m in (0, 3, 4, 5):
        assert abs(X[m]) < 1e-12
    assert abs(X[7]) == pytest.approx(abs(X[1]))  # real input: |X(N-m)| = |X(m)|


def test_direct_idft_reconstructs_signal() -> None:
    x = np.random.default_rng(7).standard_normal(64)
    reconstructed = direct_idft(direct_dft(x))
    assert np.max(np.abs(reconstructed.real - x)) < 1e-11
    assert np.max(np.abs(reconstructed.imag)) < 1e-11


def test_lecture_leakage_example() -> None:
    """SaS-w11-3: 64-point DFT of 3.0 cycles (no leakage) vs. 3.4 cycles (energy leaks into every bin)."""
    n = np.arange(64)
    whole = np.abs(direct_dft(np.sin(2 * np.pi * 3.0 * n / 64)))
    assert whole[3] == pytest.approx(32.0)               # A·N/2
    assert np.max(np.delete(whole[:32], 3)) < 1e-10       # nothing anywhere else

    leaky = np.abs(direct_dft(np.sin(2 * np.pi * 3.4 * n / 64)))
    assert leaky[3] == pytest.approx(25.1, abs=0.1)       # slide: ≈ 25
    assert leaky[4] == pytest.approx(15.4, abs=0.1)       # slide: ≈ 15.5
    assert leaky[10] > 1.0                                 # leakage far from the tone

    hann = np.abs(direct_dft(np.sin(2 * np.pi * 3.4 * n / 64) * window("hanning", 64)))
    assert hann[10] < leaky[10] / 20                       # Hanning suppresses far leakage


def test_hanning_reveals_weak_tone() -> None:
    """SaS-w11-3: a 0.1-amplitude 7-cycle tone next to a 1.0-amplitude 3.4-cycle tone."""
    n = np.arange(64)
    x = np.sin(2 * np.pi * 3.4 * n / 64) + 0.1 * np.sin(2 * np.pi * 7 * n / 64)
    rect = np.abs(direct_dft(x))
    hann = np.abs(direct_dft(x * window("hanning", 64)))
    assert not rect[7] > rect[6]                      # rectangular: buried in leakage
    assert hann[7] > 2 * max(hann[6], hann[8])        # Hanning: a visible peak at m = 7


def test_window_formulas() -> None:
    n = np.arange(8)
    assert np.allclose(window("rectangular", 8), 1.0)
    assert np.allclose(window("triangular", 8), [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25])
    assert np.allclose(window("hanning", 8), 0.5 - 0.5 * np.cos(2 * np.pi * n / 8))
    assert np.allclose(window("hamming", 8), 0.54 - 0.46 * np.cos(2 * np.pi * n / 8))
    with pytest.raises(ValueError):
        window("kaiser", 8)
