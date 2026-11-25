"""2-FSK Acoustic Transmitter (Option A — Acoustic Messenger).

Encodes binary messages (raw bitstrings or text) into continuous-phase 2-FSK
audio using two tones (`f0` for bit '0', `f1` for bit '1') and saves mono
16-bit PCM WAV files paired with JSON metadata for reproducibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.audio_io import save_metadata, save_wav
from src.config import MIN_REQUIRED_BITS, ModulationConfig, TransmissionMetadata


def validate_bits(bits: str, allow_short: bool = False) -> str:
    """Validate a bitstring and strip whitespace (e.g. "1011 0010" -> "10110010")."""
    cleaned = "".join(bits.split())
    if not cleaned:
        raise ValueError("Bitstring cannot be empty.")

    invalid = set(cleaned) - {"0", "1"}
    if invalid:
        raise ValueError(f"Bitstring must contain only '0' and '1'; found invalid chars: {sorted(invalid)}")

    if not allow_short and len(cleaned) < MIN_REQUIRED_BITS:
        raise ValueError(
            f"Message has {len(cleaned)} bits, but Option A requires at least "
            f"{MIN_REQUIRED_BITS} bits. Pass allow_short=True for short sanity tests."
        )
    return cleaned


def validate_config(config: ModulationConfig) -> None:
    """Validate modulation parameters against Nyquist and physical constraints."""
    nyquist = config.sample_rate / 2.0
    for name, freq in (("f0", config.f0), ("f1", config.f1)):
        if not (0.0 < freq < nyquist):
            raise ValueError(f"{name}={freq} Hz must be strictly between 0 and Nyquist ({nyquist} Hz).")

    if config.f0 == config.f1:
        raise ValueError(f"f0 and f1 must be distinct; both are {config.f0} Hz.")

    if config.samples_per_bit < 2:
        raise ValueError(
            f"bit_duration={config.bit_duration}s at fs={config.sample_rate}Hz gives fewer than 2 samples per bit."
        )

    if not (0.0 < config.amplitude <= 1.0):
        raise ValueError(f"amplitude must be in (0, 1], got {config.amplitude}")

    if config.lead_silence < 0.0 or config.trail_silence < 0.0:
        raise ValueError("lead_silence and trail_silence must be non-negative.")


def text_to_bits(text: str) -> str:
    """Convert text into 8 bits per UTF-8 byte, most significant bit first ("I" -> "01001001")."""
    if not text:
        raise ValueError("Input text cannot be empty.")
    return "".join(f"{byte:08b}" for byte in text.encode("utf-8"))


def bits_to_text(bits: str) -> str:
    """Convert an 8-bit-aligned bitstring back to text (inverse of `text_to_bits`)."""
    cleaned = validate_bits(bits, allow_short=True)
    if len(cleaned) % 8 != 0:
        raise ValueError(f"Bitstring length ({len(cleaned)}) must be a multiple of 8 to decode text.")
    raw = bytes(int(cleaned[i : i + 8], 2) for i in range(0, len(cleaned), 8))
    return raw.decode("utf-8", errors="replace")


def modulate_fsk(
    bits: str,
    config: Optional[ModulationConfig] = None,
    *,
    allow_short: bool = False,
) -> np.ndarray:
    """Synthesize a mono continuous-phase 2-FSK waveform for `bits`.

    Each bit holds frequency f0 ('0') or f1 ('1') for `samples_per_bit` samples.
    The phase is the running sum of the instantaneous frequency,

        phase[n] = 2*pi/fs * sum_{k<n} f[k],

    so it never jumps at bit boundaries (no clicks), and consecutive identical
    bits form one uninterrupted tone (Slide 4: "11" -> one 0.2 s tone).
    """
    cfg = config if config is not None else ModulationConfig()
    validate_config(cfg)
    payload = validate_bits(bits, allow_short=allow_short)

    bit_freqs = np.array([cfg.f1 if b == "1" else cfg.f0 for b in payload])
    inst_freq = np.repeat(bit_freqs, cfg.samples_per_bit)
    phase = 2.0 * np.pi * (np.cumsum(inst_freq) - inst_freq) / cfg.sample_rate
    tone = cfg.amplitude * np.sin(phase)

    lead = int(round(cfg.lead_silence * cfg.sample_rate))
    trail = int(round(cfg.trail_silence * cfg.sample_rate))
    return np.pad(tone, (lead, trail))


def encode_to_wav(
    output_wav_path: str | Path,
    *,
    bits: Optional[str] = None,
    text: Optional[str] = None,
    config: Optional[ModulationConfig] = None,
    allow_short: bool = False,
    notes: str = "",
) -> tuple[Path, Path, TransmissionMetadata, np.ndarray]:
    """Encode a bitstring or text message to a WAV file plus companion JSON metadata.

    Returns `(wav_path, meta_path, metadata, signal)`.
    """
    if (bits is None) == (text is None):
        raise ValueError("Provide exactly one of `bits` or `text`.")

    cfg = config if config is not None else ModulationConfig()
    payload = validate_bits(text_to_bits(text) if text is not None else bits, allow_short=allow_short)
    signal = modulate_fsk(payload, cfg, allow_short=allow_short)

    wav_path = save_wav(output_wav_path, signal, cfg.sample_rate)
    metadata = TransmissionMetadata(
        bits=payload,
        config=cfg,
        text=text,
        wav_path=wav_path.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    meta_path = save_metadata(wav_path.with_suffix(".json"), metadata)
    return wav_path, meta_path, metadata, signal
