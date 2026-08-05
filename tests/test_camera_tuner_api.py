"""Flask API tests using a deterministic non-MuJoCo session."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from multiview_collect_demo.camera_tuner.app import (
    CameraTunerController,
    create_app,
)
from multiview_collect_demo.task_camera_config import (
    OPERATION_CAMERA_NAMES,
    CameraPose,
    TaskCameraConfig,
)


def _default_poses(offset: float = 0.0) -> dict[str, CameraPose]:
    return {
        camera_name: CameraPose.from_mapping(
            {
                "position": [offset + index, 0.0, 1.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        )
        for index, camera_name in enumerate(OPERATION_CAMERA_NAMES)
    }


class FakeSession:
    def __init__(self) -> None:
        self.camera_poses = _default_poses()
        self.initial_camera_poses = dict(self.camera_poses)
        self.demo_keys = ["demo_0", "demo_1"]
        self.demo_lengths = [4, 7]
        self.demo_index = 0
        self.frame_index = 0
        self.closed = False

    @property
    def frame_count(self) -> int:
        return self.demo_lengths[self.demo_index]

    def load_task(self, task, dataset_path, saved_poses=None) -> None:
        del dataset_path
        self.camera_poses = dict(saved_poses or _default_poses(float(task.index)))
        self.initial_camera_poses = dict(self.camera_poses)
        self.demo_index = 0
        self.frame_index = 0

    def switch_demo(self, demo_index: int) -> None:
        if not 0 <= demo_index < len(self.demo_keys):
            raise ValueError("Demo index out of range")
        self.demo_index = demo_index
        self.frame_index = 0

    def switch_frame(self, frame_index: int) -> None:
        if not 0 <= frame_index < self.frame_count:
            raise ValueError("Frame index out of range")
        self.frame_index = frame_index

    def adjust_camera(self, camera_name, translation, rotation_degrees):
        pose = self.camera_poses[camera_name]
        rotation = Rotation.from_quat(
            [
                pose.quaternion_wxyz[1],
                pose.quaternion_wxyz[2],
                pose.quaternion_wxyz[3],
                pose.quaternion_wxyz[0],
            ]
        ) * Rotation.from_euler("xyz", rotation_degrees, degrees=True)
        quat_xyzw = rotation.as_quat()
        updated = CameraPose.from_mapping(
            {
                "position": (np.asarray(pose.position) + translation).tolist(),
                "quaternion_wxyz": [
                    quat_xyzw[3],
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                ],
            }
        )
        self.camera_poses[camera_name] = updated
        return updated

    def reset_camera(self, camera_name):
        self.camera_poses[camera_name] = self.initial_camera_poses[camera_name]
        return self.camera_poses[camera_name]

    def render_jpeg(self, camera_name: str) -> bytes:
        if camera_name not in self.camera_poses:
            raise ValueError(f"Unknown operation camera: {camera_name}")
        camera_index = OPERATION_CAMERA_NAMES.index(camera_name)
        colors = [(45, 135, 105), (72, 112, 170), (184, 123, 62), (148, 82, 105)]
        x_axis = np.linspace(0, 45, 128, dtype=np.uint8)[None, :, None]
        pose = self.camera_poses[camera_name]
        pose_offset = int(
            round((sum(pose.position) + sum(pose.quaternion_wxyz)) * 12)
        )
        base = np.asarray(colors[camera_index], dtype=np.int16)[None, None, :]
        image = np.broadcast_to(base, (128, 128, 3)).copy()
        image = np.clip(image + x_axis + pose_offset, 0, 255).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(output, format="JPEG")
        return output.getvalue()

    def close(self) -> None:
        self.closed = True


def _build_controller(config_path: Path) -> CameraTunerController:
    return CameraTunerController(
        source_root=config_path.parent / "datasets",
        config=TaskCameraConfig.load(config_path),
        session_factory=FakeSession,
        validate_datasets=False,
    )


def test_api_task_adjust_demo_frame_render_and_errors(tmp_path: Path) -> None:
    controller = _build_controller(tmp_path / "cameras.yaml")
    client = create_app(controller).test_client()

    initial = client.get("/api/bootstrap").get_json()
    assert initial["current_task"]["index"] == 0
    assert initial["progress"] == {"confirmed": 0, "total": 40}
    revision = initial["revision"]

    adjusted = client.post(
        "/api/adjust",
        json={
            "camera": "operation_backview",
            "translation": [0.25, 0, 0],
            "rotation_degrees": [0, 0, 0],
        },
    ).get_json()
    assert adjusted["revision"] > revision
    assert adjusted["dirty"] is True
    assert adjusted["cameras"]["operation_backview"]["position"][0] == 0.25

    demo_state = client.post("/api/demo", json={"demo_index": 1}).get_json()
    assert demo_state["demo"]["frame_count"] == 7
    assert demo_state["cameras"]["operation_backview"]["position"][0] == 0.25
    frame_state = client.post("/api/frame", json={"frame_index": 6}).get_json()
    assert frame_state["demo"]["frame_index"] == 6

    render = client.get("/api/render/operation_backview.jpg")
    assert render.status_code == 200
    assert render.mimetype == "image/jpeg"
    assert render.headers["X-Camera-Revision"] == str(frame_state["revision"])
    assert client.get("/api/render/not-a-camera.jpg").status_code == 400
    assert client.post("/api/task", json={"task_id": "bad/task"}).status_code == 400
    assert client.post(
        "/api/adjust", json={"camera": "operation_backview", "translation": [1, 2]}
    ).status_code == 400


def test_confirm_advances_and_restart_resumes_first_unconfirmed(tmp_path: Path) -> None:
    config_path = tmp_path / "cameras.yaml"
    controller = _build_controller(config_path)
    client = create_app(controller).test_client()

    confirmed = client.post("/api/confirm", json={}).get_json()

    assert confirmed["progress"]["confirmed"] == 1
    assert confirmed["current_task"]["index"] == 1
    assert config_path.is_file()
    reloaded = TaskCameraConfig.load(config_path)
    assert reloaded.confirmed_count == 1

    restarted = _build_controller(config_path)
    assert restarted.current_index == 1
    first_task = restarted.tasks[0]
    assert restarted.config.is_confirmed(first_task.suite, first_task.task_name)
