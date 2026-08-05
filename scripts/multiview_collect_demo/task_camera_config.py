"""Task-level operation camera configuration shared by replay and tuning tools."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from libero.libero.benchmark.libero_suite_task_map import libero_task_map

SCHEMA_VERSION = 1
SUITE_NAMES = (
    "libero_spatial",
    "libero_goal",
    "libero_object",
    "libero_10",
)
OPERATION_CAMERA_NAMES = (
    "operation_backview",
    "operation_leftview",
    "operation_rightview",
    "operation_topview",
)
DEFAULT_TASK_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "task_operation_cameras.yaml"
)


@dataclass(frozen=True)
class CameraPose:
    """One camera pose in world coordinates using a MuJoCo wxyz quaternion."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraPose":
        """Parses and validates a serialized camera pose."""

        if not isinstance(value, Mapping):
            raise ValueError("Camera pose must be a mapping")
        if set(value) != {"position", "quaternion_wxyz"}:
            raise ValueError(
                "Camera pose must contain exactly 'position' and 'quaternion_wxyz'"
            )
        position = _float_tuple(value["position"], 3, "position")
        quaternion = _float_tuple(value["quaternion_wxyz"], 4, "quaternion_wxyz")
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm <= 1e-12:
            raise ValueError("Camera quaternion must have non-zero length")
        normalized = tuple(component / norm for component in quaternion)
        return cls(position=position, quaternion_wxyz=normalized)

    def to_mapping(self) -> dict[str, list[float]]:
        """Returns the stable YAML representation of this pose."""

        return {
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
        }


@dataclass(frozen=True)
class TaskRecord:
    """A task in the deterministic 40-task tuning sequence."""

    index: int
    suite: str
    task_name: str

    @property
    def task_id(self) -> str:
        """Returns the stable task identifier used by the web API."""

        return f"{self.suite}/{self.task_name}"

    def dataset_path(self, source_root: Path) -> Path:
        """Returns the official demonstration path for this task."""

        return source_root / self.suite / f"{self.task_name}_demo.hdf5"


def ordered_tasks() -> tuple[TaskRecord, ...]:
    """Returns the four 10-task suites in the tuner-defined stable order."""

    records = []
    for suite in SUITE_NAMES:
        for task_name in libero_task_map[suite]:
            records.append(
                TaskRecord(index=len(records), suite=suite, task_name=task_name)
            )
    if len(records) != 40:
        raise RuntimeError(f"Expected 40 tuning tasks, found {len(records)}")
    return tuple(records)


def _float_tuple(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must contain only finite numbers")
    return result


class TaskCameraConfig:
    """Validated task-camera poses backed by an atomically written YAML file."""

    def __init__(
        self,
        path: Path,
        poses: Optional[dict[str, dict[str, dict[str, CameraPose]]]] = None,
    ) -> None:
        self.path = Path(path)
        self._poses = poses or {suite: {} for suite in SUITE_NAMES}

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "TaskCameraConfig":
        """Loads a config file, treating a missing file as an empty config."""

        config_path = Path(path)
        if not config_path.exists():
            return cls(config_path)
        with config_path.open("r", encoding="utf-8") as file_obj:
            raw = yaml.safe_load(file_obj) or {}
        return cls(config_path, cls._parse_document(raw))

    @staticmethod
    def _parse_document(
        raw: Any,
    ) -> dict[str, dict[str, dict[str, CameraPose]]]:
        if not isinstance(raw, Mapping):
            raise ValueError("Task camera config must be a mapping")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported task camera schema_version: {raw.get('schema_version')!r}"
            )
        suites = raw.get("suites")
        if not isinstance(suites, Mapping):
            raise ValueError("Task camera config must contain a 'suites' mapping")
        unknown_suites = set(suites) - set(SUITE_NAMES)
        if unknown_suites:
            raise ValueError(f"Unknown suites in camera config: {sorted(unknown_suites)}")

        known_tasks = {suite: set(libero_task_map[suite]) for suite in SUITE_NAMES}
        parsed = {suite: {} for suite in SUITE_NAMES}
        for suite, task_entries in suites.items():
            if not isinstance(task_entries, Mapping):
                raise ValueError(f"Suite '{suite}' must contain a task mapping")
            for task_name, camera_entries in task_entries.items():
                if task_name not in known_tasks[suite]:
                    raise ValueError(f"Unknown task '{suite}/{task_name}'")
                if not isinstance(camera_entries, Mapping):
                    raise ValueError(f"Task '{suite}/{task_name}' must be a mapping")
                if set(camera_entries) != set(OPERATION_CAMERA_NAMES):
                    raise ValueError(
                        f"Task '{suite}/{task_name}' must contain exactly the four "
                        "operation cameras"
                    )
                parsed[suite][task_name] = {
                    camera_name: CameraPose.from_mapping(camera_entries[camera_name])
                    for camera_name in OPERATION_CAMERA_NAMES
                }
        return parsed

    def get_task(
        self, suite: str, task_name: str
    ) -> Optional[dict[str, CameraPose]]:
        """Returns a confirmed task pose mapping, or ``None`` if unconfirmed."""

        task_poses = self._poses.get(suite, {}).get(task_name)
        return dict(task_poses) if task_poses is not None else None

    def is_confirmed(self, suite: str, task_name: str) -> bool:
        """Returns whether the task has a complete confirmed camera entry."""

        return task_name in self._poses.get(suite, {})

    @property
    def confirmed_count(self) -> int:
        """Returns the number of confirmed task entries."""

        return sum(len(tasks) for tasks in self._poses.values())

    def set_task(
        self,
        suite: str,
        task_name: str,
        camera_poses: Mapping[str, CameraPose | Mapping[str, Any]],
    ) -> None:
        """Validates and stores all four poses for one task in memory."""

        serialized_poses = {
            camera_name: (
                pose.to_mapping() if isinstance(pose, CameraPose) else pose
            )
            for camera_name, pose in camera_poses.items()
        }
        document = {
            "schema_version": SCHEMA_VERSION,
            "suites": {suite: {task_name: serialized_poses}},
        }
        parsed = self._parse_document(document)
        self._poses[suite][task_name] = parsed[suite][task_name]

    def save(self) -> None:
        """Atomically writes the complete config next to its destination."""

        document = {
            "schema_version": SCHEMA_VERSION,
            "suites": {
                suite: {
                    task_name: {
                        camera_name: camera_poses[camera_name].to_mapping()
                        for camera_name in OPERATION_CAMERA_NAMES
                    }
                    for task_name, camera_poses in tasks.items()
                }
                for suite, tasks in self._poses.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_path_str = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temp_path = Path(temp_path_str)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_obj:
                yaml.safe_dump(document, file_obj, sort_keys=False)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
