"""Audio I/O and metadata helpers for Acoustic Messenger.

Provides mono 16-bit PCM WAV reading/writing, JSON metadata persistence, and
optional live playback/recording via `sounddevice`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile

from src.config import TransmissionMetadata


def save_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> Path:
    """Save a 1-D floating-point audio signal in [-1, 1] as 16-bit mono PCM WAV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D mono signal, got shape {arr.shape}")

    peak = np.max(np.abs(arr)) if arr.size > 0 else 0.0
    if peak > 1.0:
        arr = arr / peak

    pcm16 = np.clip(np.round(arr * 32767.0), -32768, 32767).astype(np.int16)
    wavfile.write(str(out_path), int(sample_rate), pcm16)
    return out_path


def load_wav(
    path: str | Path,
    remove_dc: bool = True,
) -> Tuple[int, np.ndarray]:
    """Load a WAV file and return `(sample_rate, mono_float64_signal)` in [-1, 1]."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"WAV file not found: {in_path}")

    sample_rate, data = wavfile.read(str(in_path))

    # Convert stereo / multi-channel audio to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        # Handle unsigned 8-bit PCM vs signed 16/32-bit PCM
        if info.min == 0:
            mid = (info.max + 1) / 2.0
            signal = (data.astype(np.float64) - mid) / mid
        else:
            scale = float(max(abs(info.min), abs(info.max)))
            signal = data.astype(np.float64) / scale
    else:
        signal = data.astype(np.float64)

    if remove_dc and signal.size > 0:
        signal = signal - np.mean(signal)

    return int(sample_rate), signal


def save_metadata(path: str | Path, metadata: TransmissionMetadata) -> Path:
    """Persist `TransmissionMetadata` to a formatted JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata.to_dict(), fh, indent=2)
    return out_path


def load_metadata(path: str | Path) -> TransmissionMetadata:
    """Load `TransmissionMetadata` from a JSON file."""
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return TransmissionMetadata.from_dict(data)


def play_audio(signal: np.ndarray, sample_rate: int) -> None:
    """Play a mono float signal through the default speaker (requires sounddevice)."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Live playback requires `sounddevice`. Install with `pip install sounddevice`."
        ) from exc

    sd.play(np.asarray(signal, dtype=np.float32), samplerate=int(sample_rate))
    sd.wait()


def record_audio(duration_sec: float, sample_rate: int) -> np.ndarray:
    """Record mono audio from the default microphone (requires sounddevice)."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Live recording requires `sounddevice`. Install with `pip install sounddevice`."
        ) from exc

    num_samples = int(round(duration_sec * sample_rate))
    recording = sd.rec(
        num_samples,
        samplerate=int(sample_rate),
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return recording.squeeze().astype(np.float64)
