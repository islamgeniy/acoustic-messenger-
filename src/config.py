"""Shared configuration and data contracts for Acoustic Messenger (Option A).

All four feature branches (`dft-validation`, `tx-encoder`, `rx-decoder`,
`eval-plots`) import these defaults and dataclasses so PRs integrate cleanly
without merge conflicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

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
# Baseline acoustic 2-FSK parameters (from project specification)
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 48_000        # Hz (standard laptop audio rate)
DEFAULT_F0 = 1_000.0                # Hz for bit '0'
DEFAULT_F1 = 2_000.0                # Hz for bit '1'
DEFAULT_BIT_DURATION = 0.1          # seconds per bit
DEFAULT_AMPLITUDE = 0.8             # peak normalized amplitude (<= 1.0)
DEFAULT_LEAD_SILENCE = 0.5          # seconds of silence before signal
DEFAULT_TRAIL_SILENCE = 0.5         # seconds of silence after signal
MIN_REQUIRED_BITS = 32              # minimum message length for system tests

# Default 32-bit baseline test payload (balanced 0/1 transitions + runs)
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
    use_preamble: bool = False
    preamble_bits: str = "10101010"

    @property
    def samples_per_bit(self) -> int:
        return int(round(self.sample_rate * self.bit_duration))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModulationConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


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
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransmissionMetadata":
        cfg = ModulationConfig.from_dict(data["config"])
        return cls(
            bits=data["bits"],
            config=cfg,
            text=data.get("text"),
            wav_path=data.get("wav_path"),
            created_at=data.get("created_at"),
            notes=data.get("notes", ""),
        )


@dataclass
class BitDecision:
    """Per-bit spectral energy / magnitude decision (Slide 5 table format)."""

    index: int
    start_time: float
    end_time: float
    mag_f0: float
    mag_f1: float
    decision: str
    expected: Optional[str] = None
    snr_db: Optional[float] = None

    @property
    def is_correct(self) -> Optional[bool]:
        if self.expected is None:
            return None
        return self.decision == self.expected


@dataclass
class DecodeResult:
    """Output returned by the receiver (`src/rx.py`) for a decoded recording."""

    decoded_bits: str
    start_time: float
    config: ModulationConfig
    bit_decisions: list[BitDecision] = field(default_factory=list)
    expected_bits: Optional[str] = None
    bit_errors: Optional[int] = None
    ber: Optional[float] = None
    mean_snr_db: Optional[float] = None
    recording_path: Optional[str] = None
    sync_method: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        return payload


@dataclass
class TrialRecord:
    """Single row in `results/trials.csv` for controlled experiment tracking."""

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
    mean_snr_db: Optional[float] = None
    passed: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
