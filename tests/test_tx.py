"""Unit tests for the 2-FSK acoustic transmitter (`src/tx.py` & CLI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.audio_io import load_metadata, load_wav, save_wav
from src.config import DEFAULT_TEST_BITS, ModulationConfig
from src.tx import (
    bits_to_text,
    encode_to_wav,
    modulate_fsk,
    text_to_bits,
    validate_bits,
)


def _dominant_freq_hz(segment: np.ndarray, sample_rate: int) -> float:
    """Helper returning the peak positive FFT frequency of a time segment."""
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    mags = np.abs(np.fft.rfft(segment))
    return float(freqs[int(np.argmax(mags))])


def test_validate_bits_rules() -> None:
    assert validate_bits(DEFAULT_TEST_BITS) == DEFAULT_TEST_BITS
    assert validate_bits("1011 0010 11100101 01101001 10010110") == DEFAULT_TEST_BITS
    assert validate_bits("1011", min_bits=4) == "1011"

    with pytest.raises(ValueError, match="at least 32"):
        validate_bits("1011")

    with pytest.raises(ValueError, match="invalid chars"):
        validate_bits("1012" * 8)

    with pytest.raises(ValueError, match="empty"):
        validate_bits("   ")


def test_text_bit_roundtrip() -> None:
    bits = text_to_bits("IUT!")
    assert len(bits) == 32
    assert bits_to_text(bits) == "IUT!"

    bits_long = text_to_bits("DFT Squad")
    assert len(bits_long) == 72
    assert bits_to_text(bits_long) == "DFT Squad"


def test_slide4_example_1011_and_continuous_tone() -> None:
    """Slide 4: 1011 -> 2000, 1000, 2000, 2000 Hz, final two 1s form one 0.2 s tone."""
    cfg = ModulationConfig(
        sample_rate=48_000,
        f0=1_000.0,
        f1=2_000.0,
        bit_duration=0.1,
        lead_silence=0.0,
        trail_silence=0.0,
        amplitude=0.8,
    )
    sig = modulate_fsk("1011", cfg, min_bits=4)
    spb = cfg.samples_per_bit

    assert sig.size == 4 * spb
    assert np.max(np.abs(sig)) <= cfg.amplitude + 1e-12

    expected_freqs = [2_000.0, 1_000.0, 2_000.0, 2_000.0]
    for i, f_exp in enumerate(expected_freqs):
        seg = sig[i * spb : (i + 1) * spb]
        assert _dominant_freq_hz(seg, cfg.sample_rate) == pytest.approx(f_exp, abs=1.0)

    # Final two '1's (samples 2*spb .. 4*spb) must form one continuous 2,000 Hz sinusoid
    last_two = sig[2 * spb : 4 * spb]
    t = np.arange(last_two.size, dtype=np.float64) / cfg.sample_rate
    ideal = cfg.amplitude * np.sin(2.0 * np.pi * 2_000.0 * t)
    assert np.max(np.abs(last_two - ideal)) < 1e-10


def test_phase_continuity_for_non_integer_cycles() -> None:
    """Verify no sudden amplitude jump at bit boundaries even for narrow/non-integer Δf."""
    cfg = ModulationConfig(
        sample_rate=48_000,
        f0=1_000.0,
        f1=1_015.0,  # 101.5 cycles in 0.1 s (non-integer)
        bit_duration=0.1,
        lead_silence=0.0,
        trail_silence=0.0,
        amplitude=0.8,
    )
    sig = modulate_fsk(DEFAULT_TEST_BITS, cfg)
    max_step = float(np.max(np.abs(np.diff(sig))))
    # Maximum single-sample step for A*sin(2*pi*f*t) is A * 2*pi*f_max / fs
    theoretical_max_step = cfg.amplitude * 2.0 * np.pi * cfg.f1 / cfg.sample_rate
    assert max_step <= theoretical_max_step * 1.01


def test_32bit_silence_lengths() -> None:
    cfg = ModulationConfig(sample_rate=48_000, bit_duration=0.1, lead_silence=0.5, trail_silence=0.25)
    sig = modulate_fsk(DEFAULT_TEST_BITS, cfg)
    expected_samples = int(round((0.5 + 0.25 + 32 * 0.1) * 48_000))
    assert sig.size == expected_samples
    # Leading and trailing silence must be exact zeros
    assert np.all(sig[: int(round(0.5 * 48_000))] == 0.0)
    assert np.all(sig[-int(round(0.25 * 48_000)) :] == 0.0)


def test_encode_to_wav_and_metadata_roundtrip(tmp_path: Path) -> None:
    cfg = ModulationConfig(f0=1_000.0, f1=1_200.0, bit_duration=0.1)
    out_wav = tmp_path / "exp2_df200.wav"
    wav_path, meta_path, meta, sig = encode_to_wav(
        out_wav,
        text="IUT!",
        config=cfg,
        notes="Exp 2 tone separation test",
    )

    assert wav_path.exists()
    assert meta_path.exists()
    assert meta.bits == text_to_bits("IUT!")

    fs, loaded_sig = load_wav(wav_path)
    loaded_meta = load_metadata(meta_path)
    assert fs == cfg.sample_rate
    assert loaded_sig.size == sig.size
    assert np.max(np.abs(loaded_sig - sig)) < 1e-4
    assert loaded_meta.bits == meta.bits
    assert loaded_meta.config.f1 == 1_200.0
    assert loaded_meta.text == "IUT!"


def test_cli_encode_message(tmp_path: Path, run_script) -> None:
    out_wav = tmp_path / "cli_test.wav"
    stdout = run_script("encode_message.py", "--bits", DEFAULT_TEST_BITS, "--f0", 1000, "--f1", 2000, "-o", out_wav)
    assert "2-FSK Acoustic Transmitter" in stdout
    assert out_wav.exists()
    assert out_wav.with_suffix(".json").exists()


def test_save_wav_refuses_to_clip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clip"):
        save_wav(tmp_path / "loud.wav", np.array([0.0, 1.5, -0.2]), 48_000)
