"""Tests for deterministic task enumeration and camera YAML persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multiview_collect_demo.task_camera_config import (
    OPERATION_CAMERA_NAMES,
    SUITE_NAMES,
    CameraPose,
    TaskCameraConfig,
    ordered_tasks,
)


def _poses(scale: float = 1.0) -> dict[str, CameraPose]:
    return {
        camera_name: CameraPose.from_mapping(
            {
                "position": [scale + index, 2.0, 3.0],
                "quaternion_wxyz": [2.0, 0.0, 0.0, 0.0],
            }
        )
        for index, camera_name in enumerate(OPERATION_CAMERA_NAMES)
    }


def test_ordered_tasks_has_four_suites_and_40_stable_tasks() -> None:
    tasks = ordered_tasks()

    assert len(tasks) == 40
    assert tuple(dict.fromkeys(task.suite for task in tasks)) == SUITE_NAMES
    assert [sum(task.suite == suite for task in tasks) for suite in SUITE_NAMES] == [
        10,
        10,
        10,
        10,
    ]
    assert [task.index for task in tasks] == list(range(40))


def test_config_normalizes_and_atomically_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "nested" / "cameras.yaml"
    config = TaskCameraConfig.load(config_path)
    task = ordered_tasks()[0]
    replace_calls = []
    original_replace = __import__("os").replace

    def recording_replace(source: Path, target: Path) -> None:
        replace_calls.append((Path(source), Path(target)))
        original_replace(source, target)

    monkeypatch.setattr("os.replace", recording_replace)
    config.set_task(task.suite, task.task_name, _poses())
    config.save()

    assert replace_calls and replace_calls[0][1] == config_path
    assert not replace_calls[0][0].exists()
    reloaded = TaskCameraConfig.load(config_path)
    assert reloaded.confirmed_count == 1
    assert reloaded.get_task(task.suite, task.task_name)[
        OPERATION_CAMERA_NAMES[0]
    ].quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert reloaded.get_task(SUITE_NAMES[1], ordered_tasks()[10].task_name) is None


def test_partial_camera_entry_is_rejected(tmp_path: Path) -> None:
    task = ordered_tasks()[0]
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "suites": {
                    task.suite: {
                        task.task_name: {
                            OPERATION_CAMERA_NAMES[0]: {
                                "position": [0, 0, 0],
                                "quaternion_wxyz": [1, 0, 0, 0],
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly the four"):
        TaskCameraConfig.load(config_path)


def test_zero_quaternion_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        CameraPose.from_mapping(
            {"position": [0, 0, 0], "quaternion_wxyz": [0, 0, 0, 0]}
        )
