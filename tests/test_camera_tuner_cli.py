"""Tests for the lightweight camera tuner command-line entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_help_does_not_load_rendering_stack(help_flag: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment.pop("MUJOCO_GL", None)

    result = subprocess.run(
        [sys.executable, "scripts/tune_multiview_cameras.py", help_flag],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert "-h, --help" in result.stdout
    assert "--source-root" in result.stdout
    assert "--render-size" in result.stdout
    assert "--model-cache-limit-gb" in result.stdout
    assert "robosuite WARNING" not in result.stdout + result.stderr
