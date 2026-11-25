"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def run_script() -> Callable[..., str]:
    """Run `scripts/<name>` with the given arguments like a user would; return its stdout (fails on a non-zero exit)."""

    def run(name: str, *args: object) -> str:
        command = [sys.executable, str(SCRIPTS_DIR / name), *map(str, args)]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout

    return run
