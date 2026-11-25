"""2-FSK Acoustic Receiver (Option A — Acoustic Messenger).

Signal path for a recording:
  1. Find where the transmission starts (manual `start_time`, or `sync.detect_start`).
  2. Cut the recording into one segment per bit interval.
  3. Hanning-window + FFT each segment and read the magnitude near f0 and near f1.
  4. Decide '1' if |X(f1)| > |X(f0)|, else '0' (Slide 5), then compute BER (Slide 6).

Why only magnitudes? The air path delays the sound by an unknown t0, and the
time-shifting property (SaS-w9-1, 3.5.2; Lyons 3-19) says a delay only multiplies
the spectrum by e^{-jωt0}: the phase changes, the magnitude |X| does not.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import BitDecision, DecodeResult, ModulationConfig, TransmissionMetadata
from src.dft import window
from src.sync import detect_start

GUARD_FRACTION = 0.05       # skip 5% at each bit edge: timing error and room echo live there
MIN_FFT_SIZE = 4096         # zero-pad so the peak is sampled finely (Lyons 3.11: no new resolution, less scalloping)
PEAK_SEARCH_HZ = 40.0       # look for each tone's peak within ±40 Hz ...
PEAK_SEARCH_MAX_SEP = 0.35  # ... but never wider than 35% of |f1 - f0|, so the two searches never overlap


def decode_transmission(
    signal: np.ndarray, sample_rate: int, meta: TransmissionMetadata, start_time: Optional[float] = None
) -> DecodeResult:
    """Decode a recording of a known transmission and score it against the bits that were sent."""
    return decode_signal(signal, sample_rate, len(meta.bits), meta.config, start_time=start_time, expected_bits=meta.bits)


def decode_signal(
    signal: np.ndarray,
    sample_rate: int,
    num_bits: int,
    config: Optional[ModulationConfig] = None,
    *,
    start_time: Optional[float] = None,
    expected_bits: Optional[str] = None,
) -> DecodeResult:
    """Decode `num_bits` from a mono recording by comparing |X(f0)| and |X(f1)| per bit."""
    cfg = config if config is not None else ModulationConfig()
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"Expected 1-D mono audio array, got shape {samples.shape}")
    if num_bits <= 0:
        raise ValueError(f"num_bits must be positive, got {num_bits}")

    sync_method = "manual" if start_time is not None else "auto"
    if start_time is None:
        start_time = detect_start(samples, sample_rate, cfg)

    decisions = []
    for index in range(num_bits):
        bit_start = start_time + index * cfg.bit_duration
        mag_f0, mag_f1 = measure_bit(samples, sample_rate, cfg, bit_start)
        decisions.append(BitDecision(
            index=index,
            start_time=bit_start,
            end_time=bit_start + cfg.bit_duration,
            mag_f0=mag_f0,
            mag_f1=mag_f1,
            decision="1" if mag_f1 > mag_f0 else "0",
            expected=expected_bits[index] if expected_bits is not None else None,
        ))
    decoded_bits = "".join(d.decision for d in decisions)
    bit_errors, ber = compute_ber(decoded_bits, expected_bits) if expected_bits is not None else (None, None)
    return DecodeResult(
        decoded_bits=decoded_bits,
        start_time=start_time,
        config=cfg,
        bit_decisions=decisions,
        expected_bits=expected_bits,
        bit_errors=bit_errors,
        ber=ber,
        sync_method=sync_method,
    )


def measure_bit(samples: np.ndarray, sample_rate: int, cfg: ModulationConfig, bit_start: float) -> tuple[float, float]:
    """Magnitudes near f0 and f1 in the bit interval starting at `bit_start` (one row of the Slide 5 table)."""
    segment = bit_segment(samples, sample_rate, bit_start, cfg.bit_duration)
    return tone_magnitudes(*bit_spectrum(segment, sample_rate), cfg.f0, cfg.f1)


def bit_segment(signal: np.ndarray, sample_rate: int, start_time: float, bit_duration: float) -> np.ndarray:
    """Return the samples of one bit interval, trimmed by `GUARD_FRACTION` at both edges."""
    trim = GUARD_FRACTION * bit_duration
    first = max(0, int(round((start_time + trim) * sample_rate)))
    last = min(signal.size, int(round((start_time + bit_duration - trim) * sample_rate)))
    return signal[first:last]


def bit_spectrum(segment: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Hanning-windowed, zero-padded FFT magnitude of one bit segment.

    X_w(m) = Σ w(n)·x(n)·e^{-j2πnm/N} (Lyons 3-26) with the lecture's Hanning
    window w(n) = 0.5 - 0.5cos(2πn/N). A bit rarely holds a whole number of tone
    cycles, so without a window its energy would leak into the other tone's bins
    (SaS-w11-3, DFT leakage). Scaled by 2/Σw(n) so a tone of amplitude A peaks
    at ≈ A. Returns `(freqs, magnitudes)` with freqs = m·fs/N_fft (Lyons 3-5).

    A segment too short to analyse (the recording ended early) has zero magnitude,
    so its bit counts as an error instead of crashing the trial.
    """
    if segment.size < 4:
        return np.zeros(1), np.zeros(1)
    w = window("hanning", segment.size)
    n_fft = max(MIN_FFT_SIZE, 1 << int(np.ceil(np.log2(segment.size))))
    mags = 2.0 * np.abs(np.fft.rfft(segment * w, n=n_fft)) / w.sum()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, mags


def tone_magnitudes(freqs: np.ndarray, mags: np.ndarray, f0: float, f1: float) -> tuple[float, float]:
    """Peak spectral magnitude near f0 and near f1 (the two numbers of the Slide 5 table)."""
    half_width = min(PEAK_SEARCH_HZ, PEAK_SEARCH_MAX_SEP * abs(f1 - f0))

    def peak_near(f: float) -> float:
        near = np.abs(freqs - f) <= half_width
        return float(mags[near].max()) if near.any() else float(mags[np.argmin(np.abs(freqs - f))])

    return peak_near(f0), peak_near(f1)


def compute_ber(decoded_bits: str, expected_bits: str) -> tuple[int, float]:
    """Return `(bit_errors, BER)` with BER = incorrect bits / transmitted bits (Slide 6)."""
    if len(decoded_bits) != len(expected_bits) or not expected_bits:
        raise ValueError(f"Cannot compare {len(decoded_bits)} decoded bits with {len(expected_bits)} expected bits.")
    errors = sum(d != e for d, e in zip(decoded_bits, expected_bits))
    return errors, errors / len(expected_bits)
