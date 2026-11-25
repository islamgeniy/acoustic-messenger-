"""Software loopback and robustness tests for the 2-FSK receiver (`src/rx.py`)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.signal import resample_poly

from src.config import DEFAULT_TEST_BITS, ModulationConfig
from src.rx import compute_ber, decode_signal, decode_wav
from src.tx import bits_to_text, encode_to_wav, modulate_fsk, text_to_bits


def test_compute_ber_exact() -> None:
    errs, ber = compute_ber("1011", "1011")
    assert errs == 0 and ber == 0.0

    errs, ber = compute_ber("1001", "1011")
    assert errs == 1 and ber == pytest.approx(0.25)

    with pytest.raises(ValueError):
        compute_ber("101", "1011")


def test_slide4_1011_clean_loopback() -> None:
    cfg = ModulationConfig(lead_silence=0.2, trail_silence=0.2)
    sig = modulate_fsk("1011", cfg, allow_short=True)

    res = decode_signal(sig, cfg.sample_rate, num_bits=4, config=cfg, start_time=0.2, expected_bits="1011")
    assert res.decoded_bits == "1011"
    assert res.bit_errors == 0
    assert res.ber == 0.0
    # Verify magnitudes clearly favor the transmitted tone on every bit (Slide 5)
    for d in res.bit_decisions:
        if d.expected == "1":
            assert d.mag_f1 > 10.0 * d.mag_f0
        else:
            assert d.mag_f0 > 10.0 * d.mag_f1


def test_32bit_auto_sync_loopback() -> None:
    cfg = ModulationConfig(lead_silence=0.5, trail_silence=0.5)
    sig = modulate_fsk(DEFAULT_TEST_BITS, cfg)

    res = decode_signal(sig, cfg.sample_rate, num_bits=32, config=cfg, expected_bits=DEFAULT_TEST_BITS)
    assert res.sync_method == "auto"
    assert res.start_time == pytest.approx(0.5, abs=0.01)
    assert res.decoded_bits == DEFAULT_TEST_BITS
    assert res.ber == 0.0
    assert res.mean_margin_db > 30.0


def test_cross_sample_rate_48k_tx_to_44k1_rx() -> None:
    """Verify decoding works when Laptop A plays at 48 kHz and Laptop B records at 44.1 kHz."""
    tx_cfg = ModulationConfig(sample_rate=48_000, lead_silence=0.3, trail_silence=0.3)
    bits = text_to_bits("IUT!")
    sig_48k = modulate_fsk(bits, tx_cfg)
    # Resample 48000 -> 44100 (ratio 147 / 160)
    sig_44k1 = resample_poly(sig_48k, 147, 160)

    res = decode_signal(sig_44k1, 44_100, num_bits=len(bits), config=tx_cfg, expected_bits=bits)
    assert res.ber == 0.0
    assert bits_to_text(res.decoded_bits) == "IUT!"


def test_narrow_tone_separation_and_degradation() -> None:
    """At Tb=0.1 s (Δf_bin=10 Hz), Δf=50 Hz decodes cleanly; a very weak signal in heavy noise fails."""
    cfg_50 = ModulationConfig(f0=1000.0, f1=1050.0, bit_duration=0.1, lead_silence=0.2, trail_silence=0.2)
    sig_50 = modulate_fsk(DEFAULT_TEST_BITS, cfg_50)
    res_50 = decode_signal(sig_50, cfg_50.sample_rate, num_bits=32, config=cfg_50, start_time=0.2, expected_bits=DEFAULT_TEST_BITS)
    assert res_50.ber == 0.0

    # Signal attenuated to 5% and buried in heavy noise -> bit errors expected
    cfg_base = ModulationConfig(lead_silence=0.2, trail_silence=0.2)
    sig_base = modulate_fsk(DEFAULT_TEST_BITS, cfg_base)
    rng = np.random.default_rng(7)
    very_noisy = 0.05 * sig_base + 0.5 * rng.standard_normal(sig_base.size)
    res_bad = decode_signal(very_noisy, cfg_base.sample_rate, num_bits=32, config=cfg_base, start_time=0.2, expected_bits=DEFAULT_TEST_BITS)
    assert res_bad.ber is not None and res_bad.ber > 0.0


def test_decode_wav_and_cli_on_baseline(tmp_path: Path) -> None:
    out_wav = tmp_path / "loopback.wav"
    wav_path, meta_path, meta, _ = encode_to_wav(out_wav, bits=DEFAULT_TEST_BITS)

    res, _, _ = decode_wav(wav_path, len(meta.bits), meta.config, expected_bits=meta.bits)
    assert res.ber == 0.0

    script = Path(__file__).resolve().parent.parent / "scripts" / "decode_recording.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(wav_path), "--meta", str(meta_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "BER             : 0.0000 (0.00%)" in proc.stdout


def test_auto_sync_with_short_silence() -> None:
    """Auto-sync must not need much silence: 0.1 s on each side is only 6% of the recording."""
    cfg = ModulationConfig(lead_silence=0.1, trail_silence=0.1)
    sig = modulate_fsk(DEFAULT_TEST_BITS, cfg)
    noisy = 0.6 * sig + 0.005 * np.random.default_rng(0).standard_normal(sig.size)
    res = decode_signal(noisy, cfg.sample_rate, num_bits=32, config=cfg, expected_bits=DEFAULT_TEST_BITS)
    assert res.start_time == pytest.approx(0.1, abs=0.002)
    assert res.ber == 0.0
