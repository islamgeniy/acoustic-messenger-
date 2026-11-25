"""2-FSK Acoustic Receiver & Demodulator (Option A — Acoustic Messenger).

Reads a recorded WAV file (from a microphone or loopback), aligns to the start
of the transmission (manual `--start-time` or automatic energy-onset / preamble
sync), slices each bit interval, compares FFT magnitudes near `f0` and `f1`,
estimates per-bit SNR (dB), and reports the Bit Error Rate (BER).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, correlate, hilbert, sosfiltfilt

from src.audio_io import load_wav
from src.config import BitDecision, DecodeResult, ModulationConfig


def compute_ber(decoded_bits: str, expected_bits: str) -> Tuple[int, float]:
    """Compute `(incorrect_bits, BER)` per Slide 6: BER = incorrect / transmitted."""
    if not expected_bits:
        raise ValueError("expected_bits cannot be empty.")

    n_tx = len(expected_bits)
    overlap = min(len(decoded_bits), n_tx)
    mismatches = sum(1 for i in range(overlap) if decoded_bits[i] != expected_bits[i])
    # Any missing or extra bits count as errors against transmitted length
    mismatches += abs(n_tx - len(decoded_bits))
    ber = mismatches / float(n_tx)
    return mismatches, ber


def _bandpass_inband(
    signal: np.ndarray,
    sample_rate: int,
    f0: float,
    f1: float,
    margin_hz: float = 200.0,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass around `[min(f0,f1), max(f0,f1)]`."""
    nyq = sample_rate / 2.0
    low = max(50.0, min(f0, f1) - margin_hz)
    high = min(nyq * 0.95, max(f0, f1) + margin_hz)
    if low >= high or signal.size < 32:
        return signal
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def detect_start_energy(
    signal: np.ndarray,
    sample_rate: int,
    config: ModulationConfig,
) -> float:
    """Automatic energy-onset detector returning estimated start time in seconds."""
    if signal.size == 0:
        return 0.0

    filtered = _bandpass_inband(signal, sample_rate, config.f0, config.f1)
    frame_len = max(16, int(round(min(0.01, config.bit_duration * 0.25) * sample_rate)))
    hop = max(4, frame_len // 4)

    if filtered.size <= frame_len:
        return 0.0

    # Fast sliding-window RMS using cumulative sum of squares
    sq = filtered * filtered
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    starts = np.arange(0, filtered.size - frame_len + 1, hop)
    rms = np.sqrt(np.maximum(0.0, (csum[starts + frame_len] - csum[starts]) / frame_len))

    peak_rms = float(np.max(rms))
    if peak_rms <= 1e-12:
        return 0.0

    noise_floor = float(np.percentile(rms, 20))
    threshold = max(noise_floor * 5.0, peak_rms * 0.22)

    above = rms >= threshold
    # Require signal to remain above threshold for at least ~35% of one bit
    sustain_frames = max(1, int(round((config.bit_duration * 0.35 * sample_rate) / hop)))
    onset_frame: Optional[int] = None
    run = 0
    for idx, flag in enumerate(above):
        if flag:
            run += 1
            if run >= sustain_frames:
                onset_frame = idx - sustain_frames + 1
                break
        else:
            run = 0

    if onset_frame is None:
        onset_frame = int(np.argmax(rms))

    # Refine to the sample where short-window energy crosses 30% of local burst RMS
    coarse_sample = int(starts[onset_frame])
    local_end = min(filtered.size, coarse_sample + int(round(config.bit_duration * sample_rate)))
    local_rms = float(np.sqrt(np.mean(filtered[coarse_sample:local_end] ** 2)))
    back_start = max(0, coarse_sample - frame_len)
    refine_slice = np.abs(filtered[back_start : coarse_sample + frame_len])
    hits = np.flatnonzero(refine_slice >= 0.30 * local_rms)
    refined_sample = back_start + int(hits[0]) if hits.size > 0 else coarse_sample

    return refined_sample / float(sample_rate)


def detect_start_preamble(
    signal: np.ndarray,
    sample_rate: int,
    config: ModulationConfig,
) -> float:
    """Phase-invariant matched-filter detector for `config.preamble_bits`.

    Returns the start time (in seconds) of the *payload* immediately following
    the preamble.
    """
    from src.tx import modulate_fsk

    preamble_cfg = ModulationConfig(
        sample_rate=sample_rate,
        f0=config.f0,
        f1=config.f1,
        bit_duration=config.bit_duration,
        amplitude=1.0,
        lead_silence=0.0,
        trail_silence=0.0,
        use_preamble=False,
    )
    template = modulate_fsk(config.preamble_bits, preamble_cfg, allow_short=True)
    if signal.size < template.size:
        return 0.0

    filtered = _bandpass_inband(signal, sample_rate, config.f0, config.f1)
    corr = correlate(filtered, template, mode="valid", method="fft")
    envelope = np.abs(hilbert(corr))
    preamble_start_sample = int(np.argmax(envelope))
    payload_start_sample = preamble_start_sample + template.size
    return payload_start_sample / float(sample_rate)


def measure_bit_spectrum(
    segment: np.ndarray,
    sample_rate: int,
    f0: float,
    f1: float,
    *,
    window: str = "hann",
    band_hz: float = 40.0,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Compute FFT magnitude near `f0`, near `f1`, and estimated SNR (dB).

    Returns `(mag_f0, mag_f1, snr_db, freqs, mag_spectrum)`.
    """
    n_seg = segment.size
    if n_seg < 4:
        return 0.0, 0.0, -np.inf, np.array([0.0]), np.array([0.0])

    if window == "hann":
        win = np.hanning(n_seg)
        coherent_gain = float(np.mean(win))
    elif window == "rect":
        win = np.ones(n_seg, dtype=np.float64)
        coherent_gain = 1.0
    else:
        raise ValueError(f"Unsupported window {window!r}; choose 'hann' or 'rect'.")

    windowed = segment * win
    # Zero-pad to a fine grid so narrow Δf peaks are accurately sampled
    n_fft = max(4096, 1 << int(np.ceil(np.log2(n_seg))))
    spec = np.fft.rfft(windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    mags = (2.0 * np.abs(spec)) / (n_seg * coherent_gain)

    # Keep search band strictly smaller than half the tone separation
    tone_sep = abs(f1 - f0)
    eff_band = min(band_hz, 0.35 * tone_sep)

    def _band_peak(target_hz: float) -> float:
        mask = np.abs(freqs - target_hz) <= eff_band
        if not np.any(mask):
            idx = int(np.argmin(np.abs(freqs - target_hz)))
            return float(mags[idx])
        return float(np.max(mags[mask]))

    mag_f0 = _band_peak(f0)
    mag_f1 = _band_peak(f1)

    # Estimate out-of-band noise floor in [300 Hz, min(4500 Hz, 0.45*fs)]
    # excluding tones f0, f1 and their 2nd harmonics
    nyq_lim = min(4500.0, 0.45 * sample_rate)
    excl = max(120.0, 2.0 / max(n_seg / float(sample_rate), 1e-4))
    noise_mask = (
        (freqs >= 300.0)
        & (freqs <= nyq_lim)
        & (np.abs(freqs - f0) > excl)
        & (np.abs(freqs - f1) > excl)
        & (np.abs(freqs - 2.0 * f0) > excl)
        & (np.abs(freqs - 2.0 * f1) > excl)
    )

    sig_mag = max(mag_f0, mag_f1)
    if np.any(noise_mask):
        # Median magnitude of complex Gaussian noise bins scaled to RMS noise bin
        noise_mag = float(np.median(mags[noise_mask]))
        snr_db = float(20.0 * np.log10((sig_mag + 1e-12) / (noise_mag + 1e-12)))
    else:
        snr_db = 0.0

    return mag_f0, mag_f1, snr_db, freqs, mags


def decode_signal(
    signal: np.ndarray,
    sample_rate: int,
    num_bits: int,
    config: Optional[ModulationConfig] = None,
    *,
    start_time: Optional[float] = None,
    expected_bits: Optional[str] = None,
    window: str = "hann",
    guard_fraction: float = 0.05,
    band_hz: float = 40.0,
    recording_path: Optional[str] = None,
) -> DecodeResult:
    """Decode `num_bits` from a 1-D acoustic recording using per-bit FFT comparison."""
    cfg = config if config is not None else ModulationConfig(sample_rate=sample_rate)
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D mono audio array, got shape {arr.shape}")
    if num_bits <= 0:
        raise ValueError(f"num_bits must be positive, got {num_bits}")
    if not (0.0 <= guard_fraction < 0.45):
        raise ValueError(f"guard_fraction must be in [0, 0.45), got {guard_fraction}")

    if start_time is not None:
        payload_start = float(start_time)
        if cfg.use_preamble:
            payload_start += len(cfg.preamble_bits) * cfg.bit_duration
        sync_method = "manual"
    elif cfg.use_preamble:
        payload_start = detect_start_preamble(arr, sample_rate, cfg)
        sync_method = "auto_preamble"
    else:
        payload_start = detect_start_energy(arr, sample_rate, cfg)
        sync_method = "auto_energy"

    decisions: list[BitDecision] = []
    decoded_chars: list[str] = []
    snr_values: list[float] = []

    for idx in range(num_bits):
        t_start = payload_start + idx * cfg.bit_duration
        t_end = t_start + cfg.bit_duration
        trim = guard_fraction * cfg.bit_duration

        s0 = max(0, int(round((t_start + trim) * sample_rate)))
        s1 = min(arr.size, int(round((t_end - trim) * sample_rate)))
        segment = arr[s0:s1]

        mag_f0, mag_f1, snr_db, _, _ = measure_bit_spectrum(
            segment,
            sample_rate,
            cfg.f0,
            cfg.f1,
            window=window,
            band_hz=band_hz,
        )

        bit = "1" if mag_f1 > mag_f0 else "0"
        exp_bit = expected_bits[idx] if (expected_bits and idx < len(expected_bits)) else None

        decisions.append(
            BitDecision(
                index=idx,
                start_time=t_start,
                end_time=t_end,
                mag_f0=mag_f0,
                mag_f1=mag_f1,
                decision=bit,
                expected=exp_bit,
                snr_db=snr_db,
            )
        )
        decoded_chars.append(bit)
        if np.isfinite(snr_db):
            snr_values.append(snr_db)

    decoded_bits = "".join(decoded_chars)
    bit_errors: Optional[int] = None
    ber: Optional[float] = None
    if expected_bits is not None:
        bit_errors, ber = compute_ber(decoded_bits, expected_bits)

    mean_snr = float(np.mean(snr_values)) if snr_values else None

    return DecodeResult(
        decoded_bits=decoded_bits,
        start_time=payload_start,
        config=cfg,
        bit_decisions=decisions,
        expected_bits=expected_bits,
        bit_errors=bit_errors,
        ber=ber,
        mean_snr_db=mean_snr,
        recording_path=recording_path,
        sync_method=sync_method,
    )


def decode_wav(
    wav_path: str | Path,
    num_bits: int,
    config: Optional[ModulationConfig] = None,
    *,
    start_time: Optional[float] = None,
    expected_bits: Optional[str] = None,
    window: str = "hann",
    guard_fraction: float = 0.05,
    band_hz: float = 40.0,
) -> DecodeResult:
    """Load a WAV recording and decode its 2-FSK binary message."""
    fs, signal = load_wav(wav_path, remove_dc=True)
    return decode_signal(
        signal,
        fs,
        num_bits=num_bits,
        config=config,
        start_time=start_time,
        expected_bits=expected_bits,
        window=window,
        guard_fraction=guard_fraction,
        band_hz=band_hz,
        recording_path=str(wav_path),
    )
