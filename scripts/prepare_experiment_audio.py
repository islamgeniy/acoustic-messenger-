#!/usr/bin/env python3
"""Generate all transmitted WAV/JSON files and a manifest template for the 4 controlled experiments."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_TEST_BITS, RECORDED_DIR, TRANSMITTED_DIR, ModulationConfig
from src.tx import encode_to_wav


def main() -> int:
    generated: list[Path] = []

    # Baseline TX (used for Baseline, Exp 3 Background Interference, and Exp 4 Distance & Position)
    wav, meta, _, _ = encode_to_wav(
        TRANSMITTED_DIR / "baseline_32bit.wav",
        bits=DEFAULT_TEST_BITS,
        config=ModulationConfig(f0=1000.0, f1=2000.0, bit_duration=0.10),
        notes="Baseline 32-bit 2-FSK transmission (f0=1000 Hz, f1=2000 Hz, Tb=0.10 s)",
    )
    generated.append(wav)

    # Exp 1: Bit Duration (Shuhratov Abdulloh)
    exp1_settings = [
        ("100 ms", 0.10, "exp1_tb100ms"),
        ("50 ms", 0.05, "exp1_tb50ms"),
        ("20 ms", 0.02, "exp1_tb20ms"),
        ("10 ms", 0.01, "exp1_tb10ms"),
    ]
    for label, tb, stem in exp1_settings:
        w, _, _, _ = encode_to_wav(
            TRANSMITTED_DIR / f"{stem}.wav",
            bits=DEFAULT_TEST_BITS,
            config=ModulationConfig(f0=1000.0, f1=2000.0, bit_duration=tb),
            notes=f"Exp 1 Bit Duration: Tb={label}",
        )
        generated.append(w)

    # Exp 2: Tone Separation (Xojambetov Islam)
    exp2_settings = [
        ("1000 Hz", 2000.0, "exp2_df1000"),
        ("200 Hz", 1200.0, "exp2_df200"),
        ("50 Hz", 1050.0, "exp2_df50"),
        ("10 Hz", 1010.0, "exp2_df10"),
    ]
    for label, f1, stem in exp2_settings:
        w, _, _, _ = encode_to_wav(
            TRANSMITTED_DIR / f"{stem}.wav",
            bits=DEFAULT_TEST_BITS,
            config=ModulationConfig(f0=1000.0, f1=f1, bit_duration=0.10),
            notes=f"Exp 2 Tone Separation: f0=1000 Hz, f1={f1:.0f} Hz (Δf={label})",
        )
        generated.append(w)

    # Build ready-to-use manifest template pointing to real recording paths
    manifest_entries: list[dict] = []

    for label, _, stem in exp1_settings:
        for t in range(1, 4):
            manifest_entries.append(
                {
                    "experiment": "exp1_bit_duration",
                    "leader": "Shuhratov Abdulloh",
                    "factor": "Bit Duration (Tb)",
                    "setting": label,
                    "trial_num": t,
                    "tx_meta": f"../transmitted/{stem}.json",
                    "rx_wav": f"exp1_bit_duration/{stem}_t{t}.wav",
                    "start_time": None,
                    "notes": "",
                }
            )

    for label, _, stem in exp2_settings:
        for t in range(1, 4):
            manifest_entries.append(
                {
                    "experiment": "exp2_tone_separation",
                    "leader": "Xojambetov Islam",
                    "factor": "Tone Separation (Δf)",
                    "setting": label,
                    "trial_num": t,
                    "tx_meta": f"../transmitted/{stem}.json",
                    "rx_wav": f"exp2_tone_separation/{stem}_t{t}.wav",
                    "start_time": None,
                    "notes": "",
                }
            )

    exp3_settings = [
        ("Quiet baseline", "quiet"),
        ("Out-of-band speech/music", "oob_noise"),
        ("In-band tone/music (~1-2 kHz)", "inband_noise"),
    ]
    for label, slug in exp3_settings:
        for t in range(1, 4):
            manifest_entries.append(
                {
                    "experiment": "exp3_background_interference",
                    "leader": "Mirobidov Mirakbar",
                    "factor": "Background Interference",
                    "setting": label,
                    "trial_num": t,
                    "tx_meta": "../transmitted/baseline_32bit.json",
                    "rx_wav": f"exp3_background_interference/{slug}_t{t}.wav",
                    "start_time": None,
                    "notes": "",
                }
            )

    exp4_settings = [
        ("0.5m LOS", "0p5m_los"),
        ("2.0m LOS", "2p0m_los"),
        ("5.0m LOS", "5p0m_los"),
        ("2.0m NLOS (blocked)", "2p0m_nlos"),
    ]
    for label, slug in exp4_settings:
        for t in range(1, 4):
            manifest_entries.append(
                {
                    "experiment": "exp4_distance_position",
                    "leader": "Malikov Amirxon",
                    "factor": "Distance & Position",
                    "setting": label,
                    "trial_num": t,
                    "tx_meta": "../transmitted/baseline_32bit.json",
                    "rx_wav": f"exp4_distance_position/{slug}_t{t}.wav",
                    "start_time": None,
                    "notes": "",
                }
            )

    template_path = RECORDED_DIR / "manifest_template.json"
    template_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")

    print(f"Generated {len(generated)} transmitted WAV + JSON pairs in {TRANSMITTED_DIR}")
    print(f"Generated manifest template ({len(manifest_entries)} trial slots) -> {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
