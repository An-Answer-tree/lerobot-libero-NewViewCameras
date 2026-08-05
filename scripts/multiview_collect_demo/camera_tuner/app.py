"""Flask application and state controller for task camera calibration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from flask import Flask, Response, jsonify, request, send_file

from libero.libero import get_libero_path
from multiview_collect_demo.camera_tuner.session import MujocoCameraSession
from multiview_collect_demo.task_camera_config import (
    DEFAULT_TASK_CAMERA_CONFIG_PATH,
    OPERATION_CAMERA_NAMES,
    CameraPose,
    TaskCameraConfig,
    TaskRecord,
    ordered_tasks,
)


class CameraTunerController:
    """Coordinates task progress, the MuJoCo session, and atomic confirmation."""

    def __init__(
        self,
        source_root: Path,
        config: TaskCameraConfig,
        session_factory: Callable[[], Any],
        validate_datasets: bool = True,
    ) -> None:
        self.source_root = Path(source_root)
        self.config = config
        self.tasks = ordered_tasks()
        if validate_datasets:
            self._validate_dataset_paths()
        self.session = session_factory()
        self.revision = 0
        self.current_index = self._first_unconfirmed_index()
        self.dirty = False
        self.completed = self.config.confirmed_count == len(self.tasks)
        self._load_task(self.current_index)

    def close(self) -> None:
        """Releases the current rendering session."""

        self.session.close()

    def _validate_dataset_paths(self) -> None:
        missing = [
            task.dataset_path(self.source_root)
            for task in self.tasks
            if not task.dataset_path(self.source_root).is_file()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                f"Missing {len(missing)} required LIBERO demonstration files:\n{formatted}"
            )

    def _first_unconfirmed_index(self) -> int:
        for task in self.tasks:
            if not self.config.is_confirmed(task.suite, task.task_name):
                return task.index
        return len(self.tasks) - 1

    def _load_task(self, index: int) -> None:
        if not 0 <= index < len(self.tasks):
            raise ValueError(f"Task index out of range: {index}")
        task = self.tasks[index]
        self.session.load_task(
            task,
            task.dataset_path(self.source_root),
            self.config.get_task(task.suite, task.task_name),
        )
        self.current_index = index
        self.dirty = False
        self.revision += 1

    def load_task(self, task_selector: Any) -> dict[str, Any]:
        """Loads a task by numeric index or stable ``suite/task`` id."""

        if isinstance(task_selector, bool):
            raise ValueError("Invalid task selector")
        if isinstance(task_selector, int):
            index = task_selector
        elif isinstance(task_selector, str):
            matches = [task.index for task in self.tasks if task.task_id == task_selector]
            if not matches:
                raise ValueError(f"Unknown task: {task_selector}")
            index = matches[0]
        else:
            raise ValueError("Task selector must be an index or task id")
        self._load_task(index)
        return self.state()

    def switch_demo(self, demo_index: int) -> dict[str, Any]:
        """Switches the current HDF5 demo and preserves camera poses."""

        self.session.switch_demo(_integer(demo_index, "demo_index"))
        self.revision += 1
        return self.state()

    def switch_frame(self, frame_index: int) -> dict[str, Any]:
        """Restores one recorded simulator state."""

        self.session.switch_frame(_integer(frame_index, "frame_index"))
        self.revision += 1
        return self.state()

    def adjust_camera(
        self,
        camera_name: str,
        translation: Sequence[float],
        rotation_degrees: Sequence[float],
    ) -> dict[str, Any]:
        """Applies one local pose adjustment and returns normalized state."""

        if camera_name not in OPERATION_CAMERA_NAMES:
            raise ValueError(f"Unknown operation camera: {camera_name}")
        translation_array = _finite_vector(translation, "translation")
        rotation_array = _finite_vector(rotation_degrees, "rotation_degrees")
        self.session.adjust_camera(camera_name, translation_array, rotation_array)
        self.dirty = True
        self.revision += 1
        return self.state()

    def reset_camera(self, camera_name: str) -> dict[str, Any]:
        """Restores one camera to its task-load pose."""

        if camera_name not in OPERATION_CAMERA_NAMES:
            raise ValueError(f"Unknown operation camera: {camera_name}")
        self.session.reset_camera(camera_name)
        self.dirty = True
        self.revision += 1
        return self.state()

    def confirm(self) -> dict[str, Any]:
        """Atomically saves all four cameras and opens the next unconfirmed task."""

        self._save_current_task()
        self.dirty = False

        next_index = self._next_unconfirmed_index(self.current_index)
        if next_index is None:
            self.completed = True
            self.revision += 1
        else:
            self.completed = False
            self._load_task(next_index)
        return self.state()

    def save_current(self) -> dict[str, Any]:
        """Atomically saves all four cameras without leaving the current task."""

        self._save_current_task()
        self.dirty = False
        self.completed = self.config.confirmed_count == len(self.tasks)
        self.revision += 1
        return self.state()

    def _save_current_task(self) -> None:
        task = self.tasks[self.current_index]
        poses = {
            camera_name: self.session.camera_poses[camera_name]
            for camera_name in OPERATION_CAMERA_NAMES
        }
        self.config.set_task(task.suite, task.task_name, poses)
        self.config.save()

    def _next_unconfirmed_index(self, current_index: int) -> Optional[int]:
        ordered_indices = list(range(current_index + 1, len(self.tasks))) + list(
            range(0, current_index)
        )
        for index in ordered_indices:
            task = self.tasks[index]
            if not self.config.is_confirmed(task.suite, task.task_name):
                return index
        return None

    def state(self) -> dict[str, Any]:
        """Returns the complete serializable UI state."""

        current_task = self.tasks[self.current_index]
        return {
            "revision": self.revision,
            "dirty": self.dirty,
            "completed": self.completed,
            "progress": {
                "confirmed": self.config.confirmed_count,
                "total": len(self.tasks),
            },
            "current_task": self._task_payload(current_task),
            "tasks": [self._task_payload(task) for task in self.tasks],
            "cameras": {
                camera_name: self.session.camera_poses[camera_name].to_mapping()
                for camera_name in OPERATION_CAMERA_NAMES
            },
            "camera_names": list(OPERATION_CAMERA_NAMES),
            "demo": {
                "index": self.session.demo_index,
                "keys": list(self.session.demo_keys),
                "lengths": list(self.session.demo_lengths),
                "frame_index": self.session.frame_index,
                "frame_count": self.session.frame_count,
            },
        }

    def _task_payload(self, task: TaskRecord) -> dict[str, Any]:
        return {
            "index": task.index,
            "id": task.task_id,
            "suite": task.suite,
            "name": task.task_name,
            "confirmed": self.config.is_confirmed(task.suite, task.task_name),
            "current": task.index == self.current_index,
        }


def _finite_vector(value: Sequence[float], label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    vector = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite numbers")
    return vector


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return parsed


def _json_body() -> Mapping[str, Any]:
    body = request.get_json(silent=True)
    if not isinstance(body, Mapping):
        raise ValueError("Request body must be a JSON object")
    return body


def create_app(
    controller: Optional[CameraTunerController] = None,
    *,
    source_root: Optional[Path] = None,
    config_path: Path = DEFAULT_TASK_CAMERA_CONFIG_PATH,
    render_size: int = 512,
    validate_datasets: bool = True,
) -> Flask:
    """Creates the Flask app, optionally with a test-provided controller."""

    if controller is None:
        resolved_source_root = Path(
            os.path.abspath(
                os.path.expanduser(os.fspath(source_root or get_libero_path("datasets")))
            )
        )
        controller = CameraTunerController(
            source_root=resolved_source_root,
            config=TaskCameraConfig.load(config_path),
            session_factory=lambda: MujocoCameraSession(render_size=render_size),
            validate_datasets=validate_datasets,
        )

    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, static_folder=os.fspath(static_dir), static_url_path="/static")
    app.config["CAMERA_TUNER_CONTROLLER"] = controller

    @app.get("/")
    def index() -> Response:
        return app.send_static_file("index.html")

    @app.get("/api/bootstrap")
    def bootstrap() -> Response:
        return jsonify(controller.state())

    @app.post("/api/task")
    def load_task() -> Response:
        body = _json_body()
        selector = body.get("task_id", body.get("index"))
        return jsonify(controller.load_task(selector))

    @app.post("/api/demo")
    def switch_demo() -> Response:
        body = _json_body()
        return jsonify(controller.switch_demo(body.get("demo_index")))

    @app.post("/api/frame")
    def switch_frame() -> Response:
        body = _json_body()
        return jsonify(controller.switch_frame(body.get("frame_index")))

    @app.post("/api/adjust")
    def adjust_camera() -> Response:
        body = _json_body()
        return jsonify(
            controller.adjust_camera(
                str(body.get("camera", "")),
                body.get("translation", [0.0, 0.0, 0.0]),
                body.get("rotation_degrees", [0.0, 0.0, 0.0]),
            )
        )

    @app.post("/api/camera/reset")
    def reset_camera() -> Response:
        body = _json_body()
        return jsonify(controller.reset_camera(str(body.get("camera", ""))))

    @app.get("/api/render/<camera_name>.jpg")
    def render_camera(camera_name: str) -> Response:
        image = controller.session.render_jpeg(camera_name)
        response = Response(image, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Camera-Revision"] = str(controller.revision)
        return response

    @app.post("/api/confirm")
    def confirm() -> Response:
        return jsonify(controller.confirm())

    @app.post("/api/save")
    def save_current() -> Response:
        return jsonify(controller.save_current())

    @app.get("/api/config/download")
    def download_config() -> Response:
        if not controller.config.path.is_file():
            controller.config.save()
        response = send_file(
            controller.config.path,
            mimetype="application/x-yaml",
            as_attachment=True,
            download_name=controller.config.path.name,
            conditional=False,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(FileNotFoundError)
    def handle_file_error(error: FileNotFoundError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 404

    return app
