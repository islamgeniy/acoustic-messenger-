"""Automatic start detection: find where the 2-FSK transmission begins in a recording.

Optional per Slide 6 ("automatic synchronisation is optional"); a manual start
time always overrides it. Three steps, one function each:

  1. `bandpass`        keep only the band that contains both tones
  2. `coarse_onset`    first moment the in-band energy rises and *stays* high
  3. `refine_onset`    move to the exact first sample of the tone
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.config import ModulationConfig

BAND_MARGIN_HZ = 200.0    # band-pass [min(f0,f1) - 200, max(f0,f1) + 200] Hz removes out-of-band noise
NOISE_PERCENTILE = 5      # quietest 5% of frames = background noise (needs >= 5% silence in the recording)
NOISE_FACTOR = 5.0        # signal must be 5x (14 dB) louder than the noise ...
PEAK_FRACTION = 0.22      # ... and at least 22% of the loudest frame (rejects weak noise bursts)
SUSTAIN_BITS = 0.35       # and stay above threshold for 35% of a bit (rejects short clicks)
REFINE_FRACTION = 0.30    # final step: first sample whose |x| reaches 30% of the tone's RMS
MAX_FRAME_SECONDS = 0.01  # energy frames are at most 10 ms ...
FRAMES_PER_BIT = 4        # ... and at most a quarter of a bit, so they resolve short bits too


def detect_start(signal: np.ndarray, sample_rate: int, config: ModulationConfig) -> float:
    """Estimate the transmission start time in seconds."""
    tones = bandpass(signal, sample_rate, config.f0, config.f1)
    samples_per_bit = int(round(config.bit_duration * sample_rate))  # at the *recording's* rate
    frame = max(16, int(round(min(MAX_FRAME_SECONDS * sample_rate, samples_per_bit / FRAMES_PER_BIT))))
    if tones.size <= frame:
        raise ValueError(f"Recording of {tones.size} samples is too short to detect a start.")

    coarse = coarse_onset(sliding_rms(tones, frame), samples_per_bit)
    return refine_onset(tones, coarse, frame, samples_per_bit) / sample_rate


def bandpass(signal: np.ndarray, sample_rate: int, f0: float, f1: float) -> np.ndarray:
    """Frequency-selective band-pass filter (SaS-w6, 3.9.2) keeping only the band with both tones.

    A 4th-order Butterworth approximates the ideal band-pass H(jω) = 1 inside
    [ω_c1, ω_c2] and 0 outside; filtering forwards and backwards cancels its phase delay.
    """
    low = max(50.0, min(f0, f1) - BAND_MARGIN_HZ)
    high = min(0.95 * sample_rate / 2.0, max(f0, f1) + BAND_MARGIN_HZ)
    sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal)


def sliding_rms(x: np.ndarray, frame: int) -> np.ndarray:
    """rms[n] = RMS of x[n : n + frame] = √(average power), SaS-w1-1 §1.1.2; one cumulative sum, no loop."""
    energy = np.concatenate(([0.0], np.cumsum(x * x)))
    return np.sqrt((energy[frame:] - energy[:-frame]) / frame)


def coarse_onset(rms: np.ndarray, samples_per_bit: int) -> int:
    """First index from which the RMS stays above the threshold for `SUSTAIN_BITS` of a bit.

    Threshold = the louder of 5x the background noise and 22% of the loudest frame.
    Falls back to the loudest frame if nothing is sustained long enough.
    """
    threshold = max(NOISE_FACTOR * np.percentile(rms, NOISE_PERCENTILE), PEAK_FRACTION * rms.max())
    sustain = max(1, int(SUSTAIN_BITS * samples_per_bit))
    above_count = np.concatenate(([0], np.cumsum(rms >= threshold)))
    sustained = (above_count[sustain:] - above_count[:-sustain]) == sustain
    return int(np.argmax(sustained)) if sustained.any() else int(np.argmax(rms))


def refine_onset(tones: np.ndarray, coarse: int, frame: int, samples_per_bit: int) -> int:
    """First sample within one frame of `coarse` whose amplitude reaches 30% of the tone's RMS."""
    tone_rms = np.sqrt(np.mean(tones[coarse : coarse + samples_per_bit] ** 2))
    search_from = max(0, coarse - frame)
    hits = np.flatnonzero(np.abs(tones[search_from : coarse + frame]) >= REFINE_FRACTION * tone_rms)
    return search_from + int(hits[0]) if hits.size else coarse
