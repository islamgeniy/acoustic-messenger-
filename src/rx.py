"""2-FSK Acoustic Receiver (Option A — Acoustic Messenger).

Signal path for a recording:
  1. Find where the transmission starts (manual `start_time`, or `detect_start`).
  2. Cut the recording into one segment per bit interval.
  3. Hann-window + FFT each segment and read the magnitude near f0 and near f1.
  4. Decide '1' if |X(f1)| > |X(f0)|, else '0' (Slide 5), then compute BER (Slide 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.audio_io import load_wav
from src.config import BitDecision, DecodeResult, ModulationConfig

# --- Bit measurement --------------------------------------------------------
GUARD_FRACTION = 0.05     # skip 5% at each bit edge: timing error and room echo live there
MIN_FFT_SIZE = 4096       # zero-pad short segments so the peak is sampled finely (no new resolution)
PEAK_SEARCH_HZ = 40.0     # look for each tone's peak within ±40 Hz ...
PEAK_SEARCH_MAX_SEP = 0.35  # ... but never wider than 35% of |f1 - f0|, so the two searches never overlap

# --- Automatic start detection ----------------------------------------------
BAND_MARGIN_HZ = 200.0    # band-pass [min(f0,f1) - 200, max(f0,f1) + 200] Hz removes out-of-band noise
NOISE_PERCENTILE = 5      # quietest 5% of frames = background noise (needs >= 5% silence in the recording)
NOISE_FACTOR = 5.0        # signal must be 5x (14 dB) louder than the noise ...
PEAK_FRACTION = 0.22      # ... and at least 22% of the loudest frame (rejects weak noise bursts)
SUSTAIN_BITS = 0.35       # and stay above threshold for 35% of a bit (rejects short clicks)
REFINE_FRACTION = 0.30    # final step: first sample whose |x| reaches 30% of the tone's RMS


def compute_ber(decoded_bits: str, expected_bits: str) -> tuple[int, float]:
    """Return `(bit_errors, BER)` with BER = incorrect bits / transmitted bits (Slide 6)."""
    if len(decoded_bits) != len(expected_bits) or not expected_bits:
        raise ValueError(f"Cannot compare {len(decoded_bits)} decoded bits with {len(expected_bits)} expected bits.")
    errors = sum(d != e for d, e in zip(decoded_bits, expected_bits))
    return errors, errors / len(expected_bits)


def bandpass(signal: np.ndarray, sample_rate: int, f0: float, f1: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass that keeps only the band containing both tones."""
    low = max(50.0, min(f0, f1) - BAND_MARGIN_HZ)
    high = min(0.95 * sample_rate / 2.0, max(f0, f1) + BAND_MARGIN_HZ)
    sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal)


def detect_start(signal: np.ndarray, sample_rate: int, config: ModulationConfig) -> float:
    """Estimate the transmission start time (seconds) from in-band energy.

    1. Band-pass the recording, then compute a sliding RMS over short frames.
    2. Coarse start: first frame where the RMS stays above the threshold for
       `SUSTAIN_BITS` of a bit (threshold = louder of 5x noise or 22% of peak).
    3. Refine to the first sample, near that frame, whose amplitude reaches
       30% of the tone's RMS.
    """
    x = bandpass(signal, sample_rate, config.f0, config.f1)
    spb = int(round(config.bit_duration * sample_rate))  # samples per bit at the *recording* rate
    frame = max(16, int(round(min(0.01 * sample_rate, spb / 4))))  # ≤ 10 ms and ≤ a quarter bit
    if x.size <= frame:
        return 0.0

    # rms[n] = RMS of x[n : n + frame], via a cumulative sum (one pass, no loop)
    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    rms = np.sqrt((csum[frame:] - csum[:-frame]) / frame)

    threshold = max(NOISE_FACTOR * np.percentile(rms, NOISE_PERCENTILE), PEAK_FRACTION * rms.max())
    sustain = max(1, int(SUSTAIN_BITS * spb))
    above = np.concatenate(([0], np.cumsum(rms >= threshold)))
    sustained = (above[sustain:] - above[:-sustain]) == sustain  # all `sustain` frames from n are above
    coarse = int(np.argmax(sustained)) if sustained.any() else int(np.argmax(rms))

    tone_rms = np.sqrt(np.mean(x[coarse : coarse + spb] ** 2))
    lo = max(0, coarse - frame)
    hits = np.flatnonzero(np.abs(x[lo : coarse + frame]) >= REFINE_FRACTION * tone_rms)
    start = lo + int(hits[0]) if hits.size else coarse
    return start / sample_rate


def bit_segment(signal: np.ndarray, sample_rate: int, start_time: float, bit_duration: float) -> np.ndarray:
    """Return the samples of one bit interval, trimmed by `GUARD_FRACTION` at both edges."""
    trim = GUARD_FRACTION * bit_duration
    s0 = max(0, int(round((start_time + trim) * sample_rate)))
    s1 = min(signal.size, int(round((start_time + bit_duration - trim) * sample_rate)))
    return signal[s0:s1]


def bit_spectrum(segment: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed, zero-padded FFT magnitude of one bit segment.

    Scaled so a tone of amplitude A peaks at ≈ A. Returns `(freqs, magnitudes)`.
    """
    if segment.size < 4:
        return np.zeros(1), np.zeros(1)
    window = np.hanning(segment.size)
    n_fft = max(MIN_FFT_SIZE, 1 << int(np.ceil(np.log2(segment.size))))
    mags = 2.0 * np.abs(np.fft.rfft(segment * window, n=n_fft)) / window.sum()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, mags


def tone_magnitudes(freqs: np.ndarray, mags: np.ndarray, f0: float, f1: float) -> tuple[float, float]:
    """Peak spectral magnitude near f0 and near f1 (the two numbers of the Slide 5 table)."""
    half_width = min(PEAK_SEARCH_HZ, PEAK_SEARCH_MAX_SEP * abs(f1 - f0))

    def peak_near(f: float) -> float:
        near = np.abs(freqs - f) <= half_width
        return float(mags[near].max()) if near.any() else float(mags[np.argmin(np.abs(freqs - f))])

    return peak_near(f0), peak_near(f1)


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
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D mono audio array, got shape {arr.shape}")
    if num_bits <= 0:
        raise ValueError(f"num_bits must be positive, got {num_bits}")

    if start_time is None:
        start_time, sync_method = detect_start(arr, sample_rate, cfg), "auto"
    else:
        sync_method = "manual"

    decisions = []
    for idx in range(num_bits):
        t_start = start_time + idx * cfg.bit_duration
        segment = bit_segment(arr, sample_rate, t_start, cfg.bit_duration)
        mag_f0, mag_f1 = tone_magnitudes(*bit_spectrum(segment, sample_rate), cfg.f0, cfg.f1)
        decisions.append(
            BitDecision(
                index=idx,
                start_time=t_start,
                end_time=t_start + cfg.bit_duration,
                mag_f0=mag_f0,
                mag_f1=mag_f1,
                decision="1" if mag_f1 > mag_f0 else "0",
                expected=expected_bits[idx] if expected_bits is not None else None,
            )
        )

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


def decode_wav(
    wav_path: str | Path,
    num_bits: int,
    config: Optional[ModulationConfig] = None,
    *,
    start_time: Optional[float] = None,
    expected_bits: Optional[str] = None,
) -> tuple[DecodeResult, np.ndarray, int]:
    """Load a WAV recording and decode it. Returns `(result, signal, sample_rate)`."""
    sample_rate, signal = load_wav(wav_path)
    result = decode_signal(signal, sample_rate, num_bits, config, start_time=start_time, expected_bits=expected_bits)
    return result, signal, sample_rate
