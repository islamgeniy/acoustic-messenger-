"""Tests for the non-audio parts of the guided lab session (`scripts/lab_session.py`)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lab_session  # noqa: E402
from src.config import RECORDED_DIR  # noqa: E402


def test_slots_cover_three_trials_per_setting_and_point_to_real_files() -> None:
    slots = lab_session.load_slots(None)
    assert len(slots) == 45
    for slot in slots:
        assert (RECORDED_DIR / slot["tx_meta"]).exists()
        assert (RECORDED_DIR / slot["tx_meta"]).with_suffix(".wav").exists()

    exp2 = lab_session.load_slots("exp2_tone_separation")
    assert len(exp2) == 12 and {s["trial_num"] for s in exp2} == {1, 2, 3}
    assert all(s["experiment"] in lab_session.SETUP for s in slots)


def test_upsert_manifest_replaces_same_recording() -> None:
    first = {"rx_wav": "exp1/a_t1.wav", "notes": ""}
    other = {"rx_wav": "exp1/a_t2.wav", "notes": ""}
    redo = {"rx_wav": "exp1/a_t1.wav", "notes": "door open"}
    entries = lab_session.upsert_manifest(lab_session.upsert_manifest([first], other), redo)
    assert entries == [other, redo]


def test_rx_session_records_decodes_and_logs(tmp_path: Path, monkeypatch) -> None:
    """Simulated Laptop B session: fake microphone + scripted key presses, no audio hardware."""
    import json
    import shutil

    import numpy as np

    from src.audio_io import load_wav
    from src.experiments import load_manifest

    recorded, transmitted = tmp_path / "recorded", tmp_path / "transmitted"
    recorded.mkdir()
    shutil.copytree(RECORDED_DIR.parent / "transmitted", transmitted)
    monkeypatch.setattr(lab_session, "RECORDED_DIR", recorded)
    monkeypatch.setattr(lab_session, "MANIFEST", recorded / "manifest.json")
    monkeypatch.setattr(lab_session, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(lab_session, "COUNTDOWN_SECONDS", 0)

    slots = lab_session.load_slots("exp2_tone_separation")[:2]

    def fake_record(duration: float, fs: int, device=None) -> np.ndarray:
        # The "room": the slot's TX audio arrives 0.7 s late, quieter, with noise
        _, tx = load_wav((recorded / slots[fake_record.calls]["tx_meta"]).with_suffix(".wav"))
        fake_record.calls += 1
        out = 0.005 * np.random.default_rng(fake_record.calls).standard_normal(int(duration * fs))
        start = int(0.7 * fs)
        out[start : start + tx.size] += 0.3 * tx[: out.size - start]
        return out

    fake_record.calls = 0
    monkeypatch.setattr(lab_session, "record_audio", fake_record)
    answers = iter(["", "", "", "door open"])  # start, keep | start, keep with a note
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    lab_session.run_rx(slots, device=None, sample_rate=48_000, redo=False, start=1)

    manifest = json.loads((recorded / "manifest.json").read_text())
    assert [e["rx_wav"] for e in manifest] == [s["rx_wav"] for s in slots]
    assert manifest[1]["notes"] == "door open"
    for spec in load_manifest(recorded / "manifest.json"):  # the analysis can read what the session wrote
        assert spec.rx_wav.exists()
