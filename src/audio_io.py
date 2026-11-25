"""Audio I/O and metadata helpers for Acoustic Messenger.

Mono 16-bit PCM WAV reading/writing, JSON metadata persistence, and live
playback/recording through the default speaker and microphone (`sounddevice`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.config import TransmissionMetadata


def save_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> Path:
    """Save a 1-D float signal in [-1, 1] as 16-bit mono PCM WAV (values outside would clip)."""
    out_path = Path(path)
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D mono signal, got shape {arr.shape}")
    peak = float(np.max(np.abs(arr), initial=0.0))
    if peak > 1.0:
        raise ValueError(f"Signal peak {peak:.3f} exceeds 1.0 and would clip; scale it down first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_path), int(sample_rate), np.round(arr * 32767.0).astype(np.int16))
    return out_path


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Load a WAV file and return `(sample_rate, mono float64 signal in [-1, 1])`.

    Stereo recordings are averaged to mono; integer PCM is scaled to [-1, 1].
    """
    sample_rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if info.min == 0:  # 8-bit WAV is unsigned, centred on 128
            mid = (info.max + 1) / 2.0
            return int(sample_rate), (data.astype(np.float64) - mid) / mid
        return int(sample_rate), data.astype(np.float64) / -float(info.min)
    return int(sample_rate), data.astype(np.float64)


def save_metadata(path: str | Path, metadata: TransmissionMetadata) -> Path:
    """Write `TransmissionMetadata` as formatted JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return out_path


def load_metadata(path: str | Path) -> TransmissionMetadata:
    """Read `TransmissionMetadata` from a JSON file."""
    return TransmissionMetadata.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def play_audio(signal: np.ndarray, sample_rate: int) -> None:
    """Play a mono float signal through the default speaker and wait until it finishes."""
    import sounddevice as sd  # imported here so the rest of the project works without audio hardware

    sd.play(np.asarray(signal, dtype=np.float32), samplerate=int(sample_rate))
    sd.wait()


def record_audio(duration_sec: float, sample_rate: int) -> np.ndarray:
    """Record mono audio from the default microphone for `duration_sec` seconds."""
    import sounddevice as sd

    recording = sd.rec(int(round(duration_sec * sample_rate)), samplerate=int(sample_rate), channels=1, dtype="float32")
    sd.wait()
    return recording[:, 0].astype(np.float64)
