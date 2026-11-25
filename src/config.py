"""Shared configuration and data contracts for Acoustic Messenger (Option A).

The transmitter, receiver, and experiment code all exchange these dataclasses,
so each module can be developed on its own branch without merge conflicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRANSMITTED_DIR = DATA_DIR / "transmitted"
RECORDED_DIR = DATA_DIR / "recorded"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TRIALS_CSV = RESULTS_DIR / "trials.csv"

# ---------------------------------------------------------------------------
# Baseline acoustic 2-FSK parameters (Slide 4)
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 48_000        # Hz (standard laptop audio rate)
DEFAULT_F0 = 1_000.0                # Hz for bit '0'
DEFAULT_F1 = 2_000.0                # Hz for bit '1'
DEFAULT_BIT_DURATION = 0.1          # seconds per bit
DEFAULT_AMPLITUDE = 0.8             # peak amplitude (<= 1.0 leaves headroom against clipping)
DEFAULT_LEAD_SILENCE = 0.5          # seconds of silence before the signal
DEFAULT_TRAIL_SILENCE = 0.5         # seconds of silence after the signal
MIN_REQUIRED_BITS = 32              # Slide 6: transmit at least 32 bits

# Default 32-bit test payload (mix of 0/1 transitions and runs)
DEFAULT_TEST_BITS = "10110010111001010110100110010110"


@dataclass
class ModulationConfig:
    """Parameters defining a 2-FSK acoustic transmission."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    f0: float = DEFAULT_F0
    f1: float = DEFAULT_F1
    bit_duration: float = DEFAULT_BIT_DURATION
    amplitude: float = DEFAULT_AMPLITUDE
    lead_silence: float = DEFAULT_LEAD_SILENCE
    trail_silence: float = DEFAULT_TRAIL_SILENCE

    @property
    def samples_per_bit(self) -> int:
        return int(round(self.sample_rate * self.bit_duration))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModulationConfig":
        known = {f.name for f in fields(cls)}  # ignore keys from older metadata files
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TransmissionMetadata:
    """Companion JSON metadata saved alongside every transmitted WAV file."""

    bits: str
    config: ModulationConfig
    text: Optional[str] = None
    wav_path: Optional[str] = None
    created_at: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # nested ModulationConfig becomes a dict too

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransmissionMetadata":
        return cls(**{**data, "config": ModulationConfig.from_dict(data["config"])})


@dataclass
class BitDecision:
    """One row of the Slide 5 table: magnitudes near f0 and f1 and the decision."""

    index: int
    start_time: float
    end_time: float
    mag_f0: float
    mag_f1: float
    decision: str
    expected: Optional[str] = None

    @property
    def is_correct(self) -> Optional[bool]:
        return None if self.expected is None else self.decision == self.expected

    @property
    def margin_db(self) -> float:
        """How much stronger the winning tone is than the other one, in dB (0 dB = coin flip)."""
        strong, weak = max(self.mag_f0, self.mag_f1), min(self.mag_f0, self.mag_f1)
        return float(20.0 * np.log10((strong + 1e-12) / (weak + 1e-12)))


@dataclass
class DecodeResult:
    """Output returned by the receiver (`src/rx.py`) for one recording."""

    decoded_bits: str
    start_time: float
    config: ModulationConfig
    bit_decisions: list[BitDecision] = field(default_factory=list)
    expected_bits: Optional[str] = None
    bit_errors: Optional[int] = None
    ber: Optional[float] = None
    sync_method: str = "manual"     # "manual" (given start time) or "auto" (detect_start)

    @property
    def mean_margin_db(self) -> float:
        return float(np.mean([d.margin_db for d in self.bit_decisions]))


@dataclass
class TrialRecord:
    """One row of `results/trials.csv` (failed trials are kept, Slide 6)."""

    experiment: str
    leader: str
    factor: str
    setting: str
    trial_num: int
    tx_wav: str
    rx_wav: str
    start_time: float
    total_bits: int
    bit_errors: int
    ber: float
    mean_margin_db: float
    passed: bool
    notes: str = ""
