"""2-FSK Acoustic Transmitter (Option A — Acoustic Messenger).

Encodes binary messages (raw bitstrings or UTF-8/ASCII text) into continuous-phase
2-FSK audio waveforms using two tones (`f0` for bit '0', `f1` for bit '1') and
saves mono 16-bit PCM WAV files paired with JSON metadata for reproducibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import secrets
from typing import Optional, Tuple

import numpy as np

from src.audio_io import save_metadata, save_wav
from src.config import (
    MIN_REQUIRED_BITS,
    ModulationConfig,
    TransmissionMetadata,
)


def validate_bits(bits: str, allow_short: bool = False) -> str:
    """Validate and normalize a bitstring (strips whitespace and underscores)."""
    if not isinstance(bits, str):
        raise TypeError(f"Expected bitstring (str), got {type(bits).__name__}")

    cleaned = "".join(bits.split()).replace("_", "")
    if not cleaned:
        raise ValueError("Bitstring cannot be empty.")

    invalid = set(cleaned) - {"0", "1"}
    if invalid:
        raise ValueError(
            f"Bitstring must contain only '0' and '1'; found invalid chars: {sorted(invalid)}"
        )

    if not allow_short and len(cleaned) < MIN_REQUIRED_BITS:
        raise ValueError(
            f"Message has {len(cleaned)} bits, but Option A requires at least "
            f"{MIN_REQUIRED_BITS} bits. Pass allow_short=True for short sanity tests."
        )

    return cleaned


def validate_config(config: ModulationConfig) -> None:
    """Validate modulation parameters against Nyquist and physical constraints."""
    if config.sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {config.sample_rate}")

    nyquist = config.sample_rate / 2.0
    for name, freq in (("f0", config.f0), ("f1", config.f1)):
        if not (0.0 < freq < nyquist):
            raise ValueError(
                f"{name}={freq} Hz must be strictly between 0 and Nyquist ({nyquist} Hz)."
            )

    if np.isclose(config.f0, config.f1):
        raise ValueError(f"f0 and f1 must be distinct; both are {config.f0} Hz.")

    if config.bit_duration <= 0.0:
        raise ValueError(f"bit_duration must be positive, got {config.bit_duration}")

    if config.samples_per_bit < 2:
        raise ValueError(
            f"bit_duration={config.bit_duration}s at fs={config.sample_rate}Hz gives "
            f"fewer than 2 samples per bit."
        )

    if not (0.0 < config.amplitude <= 1.0):
        raise ValueError(f"amplitude must be in (0, 1], got {config.amplitude}")

    if config.lead_silence < 0.0 or config.trail_silence < 0.0:
        raise ValueError("lead_silence and trail_silence must be non-negative.")

    if config.use_preamble:
        validate_bits(config.preamble_bits, allow_short=True)


def text_to_bits(text: str, encoding: str = "utf-8") -> str:
    """Convert text into an 8-bit-per-byte MSB-first binary string."""
    if not text:
        raise ValueError("Input text cannot be empty.")
    raw = text.encode(encoding)
    return "".join(f"{byte:08b}" for byte in raw)


def bits_to_text(bits: str, encoding: str = "utf-8") -> str:
    """Convert an 8-bit-aligned binary string back to text."""
    cleaned = validate_bits(bits, allow_short=True)
    if len(cleaned) % 8 != 0:
        raise ValueError(
            f"Bitstring length ({len(cleaned)}) must be a multiple of 8 to decode text."
        )
    raw = bytes(int(cleaned[i : i + 8], 2) for i in range(0, len(cleaned), 8))
    return raw.decode(encoding, errors="replace")


def generate_random_bits(n_bits: int = MIN_REQUIRED_BITS, seed: Optional[int] = None) -> str:
    """Generate a binary string of length `n_bits` (deterministic if `seed` is set)."""
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")

    if seed is not None:
        rng = np.random.default_rng(seed)
        return "".join(str(int(b)) for b in rng.integers(0, 2, size=n_bits))

    sys_rng = secrets.SystemRandom()
    return "".join(sys_rng.choice(("0", "1")) for _ in range(n_bits))


def modulate_fsk(
    bits: str,
    config: Optional[ModulationConfig] = None,
    *,
    allow_short: bool = False,
    phase_continuous: bool = True,
) -> np.ndarray:
    """Synthesize a mono 2-FSK acoustic waveform for `bits`.

    Consecutive identical bits produce one uninterrupted tone (e.g., two '1's
    at `bit_duration=0.1 s` form a single continuous 0.2 s tone at `f1`).
    With `phase_continuous=True` (default), instantaneous phase is integrated
    across all samples so there are no broadband phase-jump clicks at bit
    boundaries even when `f * bit_duration` is non-integer.
    """
    cfg = config if config is not None else ModulationConfig()
    validate_config(cfg)
    payload_bits = validate_bits(bits, allow_short=allow_short)

    tx_bits = (
        validate_bits(cfg.preamble_bits, allow_short=True) + payload_bits
        if cfg.use_preamble
        else payload_bits
    )

    spb = cfg.samples_per_bit
    bit_freqs = np.fromiter(
        (cfg.f1 if ch == "1" else cfg.f0 for ch in tx_bits),
        dtype=np.float64,
        count=len(tx_bits),
    )

    t_bit = np.arange(spb, dtype=np.float64) / float(cfg.sample_rate)
    segments = []
    phase_offset = 0.0
    two_pi = 2.0 * np.pi

    for freq in bit_freqs:
        if phase_continuous:
            phase = phase_offset + two_pi * freq * t_bit
            segments.append(cfg.amplitude * np.sin(phase))
            # Exact phase at start of next bit, wrapped to [0, 2*pi) to avoid drift
            phase_offset = (phase_offset + two_pi * freq * spb / float(cfg.sample_rate)) % two_pi
        else:
            segments.append(cfg.amplitude * np.sin(two_pi * freq * t_bit))

    active_signal = np.concatenate(segments)

    lead_samples = int(round(cfg.lead_silence * cfg.sample_rate))
    trail_samples = int(round(cfg.trail_silence * cfg.sample_rate))

    parts = []
    if lead_samples > 0:
        parts.append(np.zeros(lead_samples, dtype=np.float64))
    parts.append(active_signal)
    if trail_samples > 0:
        parts.append(np.zeros(trail_samples, dtype=np.float64))

    return np.concatenate(parts)


def encode_to_wav(
    output_wav_path: str | Path,
    *,
    bits: Optional[str] = None,
    text: Optional[str] = None,
    config: Optional[ModulationConfig] = None,
    allow_short: bool = False,
    phase_continuous: bool = True,
    notes: str = "",
) -> Tuple[Path, Path, TransmissionMetadata, np.ndarray]:
    """Encode a bitstring or text message to a WAV file and companion JSON metadata.

    Returns `(wav_path, meta_path, metadata, signal)`.
    """
    if (bits is None) == (text is None):
        raise ValueError("Provide exactly one of `bits` or `text`.")

    cfg = config if config is not None else ModulationConfig()
    raw_bits = text_to_bits(text) if text is not None else bits
    assert raw_bits is not None
    payload_bits = validate_bits(raw_bits, allow_short=allow_short)

    signal = modulate_fsk(
        payload_bits,
        cfg,
        allow_short=allow_short,
        phase_continuous=phase_continuous,
    )

    wav_path = save_wav(output_wav_path, signal, cfg.sample_rate)
    meta_path = wav_path.with_suffix(".json")

    metadata = TransmissionMetadata(
        bits=payload_bits,
        config=cfg,
        text=text,
        wav_path=wav_path.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    save_metadata(meta_path, metadata)

    return wav_path, meta_path, metadata, signal
