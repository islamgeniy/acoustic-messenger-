"""Unit tests for the direct DFT implementation (`src/dft.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from src.dft import (
    dft_frequencies,
    direct_dft,
    direct_dft_matrix,
    direct_idft,
    single_sided_spectrum,
    validate_dft_against_fft,
)


@pytest.mark.parametrize("n_samples", [16, 64, 128, 256])
def test_direct_dft_matches_numpy_fft(n_samples: int) -> None:
    rng = np.random.default_rng(42 + n_samples)
    x = rng.standard_normal(n_samples)

    x_direct = direct_dft(x)
    x_matrix = direct_dft_matrix(x)
    x_fft = np.fft.fft(x)

    assert np.max(np.abs(x_direct - x_fft)) < 1e-11
    assert np.max(np.abs(x_matrix - x_fft)) < 1e-11


def test_direct_idft_reconstructs_signal() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(64)
    reconstructed = direct_idft(direct_dft(x))
    assert np.max(np.abs(reconstructed.real - x)) < 1e-11
    assert np.max(np.abs(reconstructed.imag)) < 1e-11


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
