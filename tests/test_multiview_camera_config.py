"""Generator CLI integration tests for task camera calibration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from multiview_collect_demo.task_camera_config import OPERATION_CAMERA_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_replay_module():
    module_path = REPO_ROOT / "scripts" / "multiview_collect_demo.py"
    spec = importlib.util.spec_from_file_location("multiview_replay_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_operation_cameras_skips_even_an_invalid_task_config(
    tmp_path: Path,
) -> None:
    replay = _load_replay_module()
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("not: [valid", encoding="utf-8")
    args = replay.parse_args(
        [
            "--no-operation-cameras",
            "--no-trajectory-cameras",
            "--task-camera-config",
            str(invalid_config),
        ]
    )

    config = replay.build_replay_config(args)

    assert config.operation_camera_config is None
    assert config.task_camera_config is None
    assert config.operation_camera_names == ()


def test_default_generator_has_16_rgb_cameras_and_four_operation_views() -> None:
    replay = _load_replay_module()
    args = replay.parse_args([])

    config = replay.build_replay_config(args)

    assert len(config.camera_names) == 16
    assert set(config.operation_camera_names) == set(OPERATION_CAMERA_NAMES)
    assert "operation_leftbackview" not in config.camera_names
    assert "operation_rightbackview" not in config.camera_names
