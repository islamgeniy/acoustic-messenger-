"""Direct Discrete Fourier Transform (DFT), inverse DFT, and window functions.

Follows lecture SaS-w11-3 "An Intuitive Guide to the DFT" (Lyons, *Understanding DSP*, ch. 3)
and the term-project Slide 13:

    DFT  (Lyons 3-2):   X(m) = sum_{n=0}^{N-1} x(n) e^{-j 2 pi n m / N}
    IDFT (Lyons 3-23):  x(n) = (1/N) sum_{m=0}^{N-1} X(m) e^{+j 2 pi m n / N}
    Bin frequency (Lyons 3-5):  f_analysis(m) = m fs / N

The direct DFT costs O(N^2) multiplications, so it is only used on short signals
to validate `numpy.fft.fft` (O(N log N)), which the receiver then uses (Slide 13).
"""

from __future__ import annotations

from dataclasses import dataclass
import time

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
    """Analysis frequency of every bin: f_analysis(m) = m fs / N, m = 0 ... N-1 (Lyons 3-5).

    Example (lecture): fs = 8000 Hz, N = 8 -> bins every 1000 Hz.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return np.arange(n_samples) * (float(sample_rate) / n_samples)


def direct_dft(x: np.ndarray) -> np.ndarray:
    """X(m) = sum_{n=0}^{N-1} x(n) e^{-j 2 pi n m / N}, computed term by term (Lyons 3-2).

    Each X(m) correlates x(n) with a complex sinusoid of exactly m cycles per N samples.
    """
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D input array, got shape {arr.shape}")
    n_samples = arr.size
    n = np.arange(n_samples)
    spectrum = np.empty(n_samples, dtype=np.complex128)
    for m in range(n_samples):
        spectrum[m] = np.sum(arr * np.exp(-2j * np.pi * n * m / n_samples))
    return spectrum


def direct_idft(X: np.ndarray) -> np.ndarray:
    """x(n) = (1/N) sum_{m=0}^{N-1} X(m) e^{+j 2 pi m n / N}, computed term by term (Lyons 3-23)."""
    spec = np.asarray(X, dtype=np.complex128)
    if spec.ndim != 1:
        raise ValueError(f"Expected 1-D input spectrum, got shape {spec.shape}")
    n_samples = spec.size
    m = np.arange(n_samples)
    signal = np.empty(n_samples, dtype=np.complex128)
    for n in range(n_samples):
        signal[n] = np.sum(spec * np.exp(2j * np.pi * m * n / n_samples)) / n_samples
    return signal


def window(name: str, n_samples: int) -> np.ndarray:
    """Window coefficients w(n), n = 0 ... N-1, exactly as defined in the lecture (Lyons 3-27 ... 3-30).

    rectangular: w(n) = 1                                  (what a plain DFT uses implicitly)
    triangular:  w(n) = n/(N/2) up to N/2, then 2 - n/(N/2)
    hanning:     w(n) = 0.5 - 0.5 cos(2 pi n / N)          (also called Hann / raised cosine)
    hamming:     w(n) = 0.54 - 0.46 cos(2 pi n / N)

    Tapering both ends of the interval toward zero lowers the sinc sidelobes, so less
    energy leaks into far-away bins, at the cost of a wider main lobe.
    """
    n = np.arange(n_samples)
    half = n_samples / 2.0
    formulas = {
        "rectangular": lambda: np.ones(n_samples),
        "triangular": lambda: np.where(n <= half, n / half, 2.0 - n / half),
        "hanning": lambda: 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_samples),
        "hamming": lambda: 0.54 - 0.46 * np.cos(2.0 * np.pi * n / n_samples),
    }
    if name not in formulas:
        raise ValueError(f"Unknown window {name!r}; choose from {sorted(formulas)}")
    return formulas[name]()


def single_sided_spectrum(
    spectrum: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive-frequency bins with amplitude and phase (radians).

    A real sinusoid of amplitude A that completes a whole number of cycles in N
    samples gives |X(m)| = A N / 2 (Lyons 3-17), so dividing by N/2 shows the
    peak at height A. (DC and the fs/2 bin are divided by N, since they have no
    mirror-image bin.)
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
    return freqs, mag, np.angle(pos)


def validate_dft_against_fft(
    x: np.ndarray,
    sample_rate: int,
    phase_mag_threshold: float = 1e-6,
) -> tuple[DFTValidationReport, np.ndarray, np.ndarray]:
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

    # Phase is meaningless where the magnitude is ~0, so only compare significant bins
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
