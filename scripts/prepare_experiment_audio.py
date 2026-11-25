#!/usr/bin/env python3
"""Generate every transmitted WAV/JSON pair and a manifest template for the controlled experiments.

Each experiment varies one factor (Slide 14). Exp 1 and Exp 2 need their own
transmissions; Exp 3 and Exp 4 replay the baseline and change the room instead.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_TEST_BITS, RECORDED_DIR, TRANSMITTED_DIR, ModulationConfig
from src.tx import encode_to_wav

TRIALS_PER_SETTING = 3
BASELINE = "baseline_32bit"

# TX file stem -> (ModulationConfig overrides, notes saved in its JSON)
TRANSMISSIONS = {
    BASELINE: ({}, "Baseline 32-bit 2-FSK transmission (f0=1000 Hz, f1=2000 Hz, Tb=0.10 s)"),
    "exp1_tb100ms": ({"bit_duration": 0.10}, "Exp 1 Bit Duration: Tb=100 ms"),
    "exp1_tb50ms": ({"bit_duration": 0.05}, "Exp 1 Bit Duration: Tb=50 ms"),
    "exp1_tb20ms": ({"bit_duration": 0.02}, "Exp 1 Bit Duration: Tb=20 ms"),
    "exp1_tb10ms": ({"bit_duration": 0.01}, "Exp 1 Bit Duration: Tb=10 ms"),
    "exp2_df1000": ({"f1": 2000.0}, "Exp 2 Tone Separation: f0=1000 Hz, f1=2000 Hz (Δf=1000 Hz)"),
    "exp2_df200": ({"f1": 1200.0}, "Exp 2 Tone Separation: f0=1000 Hz, f1=1200 Hz (Δf=200 Hz)"),
    "exp2_df50": ({"f1": 1050.0}, "Exp 2 Tone Separation: f0=1000 Hz, f1=1050 Hz (Δf=50 Hz)"),
    "exp2_df10": ({"f1": 1010.0}, "Exp 2 Tone Separation: f0=1000 Hz, f1=1010 Hz (Δf=10 Hz)"),
}

# (experiment folder, leader, factor, [(setting label, TX stem, recording file stem), ...])
EXPERIMENTS = [
    ("exp1_bit_duration", "Shuhratov Abdulloh", "Bit Duration (Tb)", [
        ("100 ms", "exp1_tb100ms", "exp1_tb100ms"),
        ("50 ms", "exp1_tb50ms", "exp1_tb50ms"),
        ("20 ms", "exp1_tb20ms", "exp1_tb20ms"),
        ("10 ms", "exp1_tb10ms", "exp1_tb10ms"),
    ]),
    ("exp2_tone_separation", "Xojambetov Islam", "Tone Separation (Δf)", [
        ("1000 Hz", "exp2_df1000", "exp2_df1000"),
        ("200 Hz", "exp2_df200", "exp2_df200"),
        ("50 Hz", "exp2_df50", "exp2_df50"),
        ("10 Hz", "exp2_df10", "exp2_df10"),
    ]),
    ("exp3_background_interference", "Mirobidov Mirakbar", "Background Interference", [
        ("Quiet baseline", BASELINE, "quiet"),
        ("Out-of-band speech/music", BASELINE, "oob_noise"),
        ("In-band tone/music (~1-2 kHz)", BASELINE, "inband_noise"),
    ]),
    ("exp4_distance_position", "Malikov Amirxon", "Distance & Position", [
        ("0.5m LOS", BASELINE, "0p5m_los"),
        ("2.0m LOS", BASELINE, "2p0m_los"),
        ("5.0m LOS", BASELINE, "5p0m_los"),
        ("2.0m NLOS (blocked)", BASELINE, "2p0m_nlos"),
    ]),
]


def main() -> int:
    for stem, (overrides, notes) in TRANSMISSIONS.items():
        encode_to_wav(TRANSMITTED_DIR / f"{stem}.wav", bits=DEFAULT_TEST_BITS,
                      config=ModulationConfig(**overrides), notes=notes)

    manifest = [
        {
            "experiment": experiment,
            "leader": leader,
            "factor": factor,
            "setting": setting,
            "trial_num": trial,
            "tx_meta": f"../transmitted/{tx_stem}.json",
            "rx_wav": f"{experiment}/{rx_stem}_t{trial}.wav",
            "start_time": None,
            "notes": "",
        }
        for experiment, leader, factor, settings in EXPERIMENTS
        for setting, tx_stem, rx_stem in settings
        for trial in range(1, TRIALS_PER_SETTING + 1)
    ]
    template_path = RECORDED_DIR / "manifest_template.json"
    template_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generated {len(TRANSMISSIONS)} transmitted WAV + JSON pairs in {TRANSMITTED_DIR}")
    print(f"Generated manifest template ({len(manifest)} trial slots) -> {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
