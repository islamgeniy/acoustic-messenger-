"""Direct Discrete Fourier Transform (DFT) implementation and validation.

Implements the exact exponential-form DFT equation from Slide 13:

    X[m] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * n * m / N)
    f_m  = m * f_s / N

Direct DFT runs in O(N^2) time and is validated on short signals against
`numpy.fft.fft`. The full acoustic receiver uses library FFTs (O(N log N)).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Tuple

import numpy as np


@dataclass
class DFTValidationReport:
    """Comparison summary between direct DFT and library FFT."""

    n_samples: int
    sample_rate: int
    max_complex_error: float
    max_magnitude_error: float
    max_phase_error_rad: float
    direct_time_ms: float
    fft_time_ms: float


def dft_frequencies(n_samples: int, sample_rate: float) -> np.ndarray:
    """Return bin center frequencies f_m = m * f_s / N for m = 0, ..., N - 1."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    m = np.arange(n_samples, dtype=np.float64)
    return m * (float(sample_rate) / float(n_samples))


def direct_dft(x: np.ndarray) -> np.ndarray:
    """Compute the 1-D DFT directly from the exponential summation formula.

    X[m] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * n * m / N)
    """
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D input array, got shape {arr.shape}")
    n_samples = arr.size
    if n_samples == 0:
        return np.array([], dtype=np.complex128)

    n = np.arange(n_samples, dtype=np.float64)
    spectrum = np.empty(n_samples, dtype=np.complex128)
    for m in range(n_samples):
        twiddle = np.exp(-1j * 2.0 * np.pi * m * n / n_samples)
        spectrum[m] = np.sum(arr * twiddle)
    return spectrum


def direct_dft_matrix(x: np.ndarray) -> np.ndarray:
    """Compute the 1-D DFT via explicit Vandermonde DFT matrix multiplication."""
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D input array, got shape {arr.shape}")
    n_samples = arr.size
    if n_samples == 0:
        return np.array([], dtype=np.complex128)

    idx = np.arange(n_samples, dtype=np.float64)
    w = np.exp(-1j * 2.0 * np.pi * np.outer(idx, idx) / n_samples)
    return w @ arr


def direct_idft(X: np.ndarray) -> np.ndarray:
    """Compute the 1-D inverse DFT directly from its exponential sum."""
    spec = np.asarray(X, dtype=np.complex128)
    if spec.ndim != 1:
        raise ValueError(f"Expected 1-D input spectrum, got shape {spec.shape}")
    n_samples = spec.size
    if n_samples == 0:
        return np.array([], dtype=np.complex128)

    m = np.arange(n_samples, dtype=np.float64)
    signal = np.empty(n_samples, dtype=np.complex128)
    for n in range(n_samples):
        twiddle = np.exp(1j * 2.0 * np.pi * m * n / n_samples)
        signal[n] = np.sum(spec * twiddle) / n_samples
    return signal


def single_sided_spectrum(
    spectrum: np.ndarray,
    sample_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract positive-frequency bins, normalized amplitude, and phase (rad).

    Normalizes amplitude so a pure sinusoid of amplitude A produces a peak of
    height A at its exact bin frequency.
    """
    n_samples = spectrum.size
    half = n_samples // 2 + 1
    freqs = dft_frequencies(n_samples, sample_rate)[:half]
    pos = spectrum[:half]

    mag = np.abs(pos) / n_samples
    if n_samples > 2:
        if n_samples % 2 == 0:
            mag[1:-1] *= 2.0
        else:
            mag[1:] *= 2.0

    phase = np.angle(pos)
    return freqs, mag, phase


def validate_dft_against_fft(
    x: np.ndarray,
    sample_rate: int,
    phase_mag_threshold: float = 1e-6,
) -> Tuple[DFTValidationReport, np.ndarray, np.ndarray]:
    """Compare `direct_dft(x)` against `numpy.fft.fft(x)` and time both."""
    arr = np.asarray(x, dtype=np.float64)

    t0 = time.perf_counter()
    x_direct = direct_dft(arr)
    direct_ms = (time.perf_counter() - t0) * 1_000.0

    t1 = time.perf_counter()
    x_fft = np.fft.fft(arr)
    fft_ms = (time.perf_counter() - t1) * 1_000.0

    max_complex_err = float(np.max(np.abs(x_direct - x_fft)))
    max_mag_err = float(np.max(np.abs(np.abs(x_direct) - np.abs(x_fft))))

    # Only compare phase where spectral magnitude is non-negligible
    significant = np.abs(x_fft) > (phase_mag_threshold * np.max(np.abs(x_fft)))
    if np.any(significant):
        phase_diff = np.angle(np.exp(1j * (np.angle(x_direct[significant]) - np.angle(x_fft[significant]))))
        max_phase_err = float(np.max(np.abs(phase_diff)))
    else:
        max_phase_err = 0.0

    report = DFTValidationReport(
        n_samples=arr.size,
        sample_rate=sample_rate,
        max_complex_error=max_complex_err,
        max_magnitude_error=max_mag_err,
        max_phase_error_rad=max_phase_err,
        direct_time_ms=direct_ms,
        fft_time_ms=fft_ms,
    )
    return report, x_direct, x_fft
