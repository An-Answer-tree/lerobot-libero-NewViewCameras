"""Single-threaded MuJoCo session used by the camera tuning web app."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Optional

import h5py
import mujoco
import numpy as np
from PIL import Image
from robosuite.utils.binding_utils import MjRenderContext, MjSim
from scipy.spatial.transform import Rotation

import replay_dataset_utils as replay_utils
from multiview_collect_demo.camera_injection import (
    OperationCameraConfig,
    install_model_xml_remapper,
)
from multiview_collect_demo.task_camera_config import (
    CameraPose,
    OPERATION_CAMERA_NAMES,
    TaskRecord,
)

SCENE4_MISSING_GEOM_NAME = "new_salad_dressing_1_g0"
SCENE4_SOURCE_OBJECT_NAME = "salad_dressing_1"
SCENE4_TARGET_OBJECT_NAME = "new_salad_dressing_1"


def _vector_text(values: np.ndarray) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _patch_scene4_xml(model_xml: str, reset_error: Exception) -> Optional[str]:
    if SCENE4_MISSING_GEOM_NAME not in str(reset_error):
        return None
    if f"{SCENE4_SOURCE_OBJECT_NAME}_g0" not in model_xml:
        return None
    return model_xml.replace(
        SCENE4_SOURCE_OBJECT_NAME, SCENE4_TARGET_OBJECT_NAME
    )


def attach_operation_cameras_to_mocap(xml_str: str) -> tuple[str, dict[str, CameraPose]]:
    """Moves all four operation cameras under independent mocap bodies."""

    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML missing <worldbody>, cannot attach tuning cameras")

    camera_elements = {
        camera.get("name"): camera
        for camera in worldbody.findall("camera")
        if camera.get("name") in OPERATION_CAMERA_NAMES
    }
    missing = set(OPERATION_CAMERA_NAMES) - set(camera_elements)
    if missing:
        raise ValueError(f"Operation cameras missing from XML: {sorted(missing)}")

    poses = {}
    for camera_name in OPERATION_CAMERA_NAMES:
        camera = camera_elements[camera_name]
        position = np.asarray(
            [float(component) for component in camera.get("pos", "").split()],
            dtype=np.float64,
        )
        quaternion = np.asarray(
            [float(component) for component in camera.get("quat", "1 0 0 0").split()],
            dtype=np.float64,
        )
        pose = CameraPose.from_mapping(
            {"position": position.tolist(), "quaternion_wxyz": quaternion.tolist()}
        )
        poses[camera_name] = pose

        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": f"camera_tuner_{camera_name}_mocap",
                "mocap": "true",
                "pos": _vector_text(np.asarray(pose.position)),
                "quat": _vector_text(np.asarray(pose.quaternion_wxyz)),
            },
        )
        camera_attrs = {
            "name": camera_name,
            "mode": "fixed",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        }
        if camera.get("fovy") is not None:
            camera_attrs["fovy"] = camera.get("fovy")
        ET.SubElement(body, "camera", camera_attrs)
        worldbody.remove(camera)

    return ET.tostring(root, encoding="unicode"), poses


class MujocoCameraSession:
    """Owns the current HDF5 demo, MuJoCo simulation, and camera poses."""

    def __init__(
        self,
        render_size: int = 512,
        model_cache_dir: Optional[Path] = Path("/tmp/libero-camera-tuner-models"),
        model_cache_limit_gb: float = 12.0,
    ) -> None:
        self.render_size = int(render_size)
        self.model_cache_dir = (
            Path(model_cache_dir) if model_cache_dir is not None else None
        )
        self.model_cache_limit_bytes = int(model_cache_limit_gb * 1024**3)
        self.sim: Optional[MjSim] = None
        self.task: Optional[TaskRecord] = None
        self.dataset_path: Optional[Path] = None
        self.demo_keys: list[str] = []
        self.demo_lengths: list[int] = []
        self.demo_index = 0
        self.frame_index = 0
        self.current_states: Optional[np.ndarray] = None
        self.camera_poses: dict[str, CameraPose] = {}
        self.initial_camera_poses: dict[str, CameraPose] = {}

    def close(self) -> None:
        """Releases the active simulation and render context."""

        if self.sim is not None:
            self.sim.free()
            self.sim = None

    def load_task(
        self,
        task: TaskRecord,
        dataset_path: Path,
        saved_poses: Optional[Mapping[str, CameraPose]] = None,
    ) -> None:
        """Loads task metadata and opens its first demonstration at frame zero."""

        self.close()
        self.task = task
        self.dataset_path = Path(dataset_path)
        with h5py.File(self.dataset_path, "r") as dataset:
            data_group = dataset["data"]
            self.demo_keys = replay_utils.sorted_demo_keys(data_group)
            if not self.demo_keys:
                raise ValueError(f"Dataset has no demo groups: {self.dataset_path}")
            self.demo_lengths = [
                int(len(data_group[demo_key]["states"])) for demo_key in self.demo_keys
            ]

        self.camera_poses = {}
        try:
            self._load_demo(0, preserve_camera_poses=False)
        except Exception:
            self.close()
            raise
        self.initial_camera_poses = dict(self.camera_poses)
        if saved_poses:
            self.camera_poses = dict(saved_poses)
            self._apply_camera_poses()

    def _load_demo(self, demo_index: int, preserve_camera_poses: bool = True) -> None:
        if self.dataset_path is None:
            raise RuntimeError("No task is loaded")
        if not 0 <= demo_index < len(self.demo_keys):
            raise ValueError(f"Demo index out of range: {demo_index}")

        previous_poses = dict(self.camera_poses) if preserve_camera_poses else {}
        with h5py.File(self.dataset_path, "r") as dataset:
            demo_group = dataset["data"][self.demo_keys[demo_index]]
            states = np.asarray(demo_group["states"][()])
            model_xml = replay_utils.decode_attr(demo_group.attrs["model_file"])
        if len(states) == 0:
            raise ValueError(f"Demo has no states: {self.demo_keys[demo_index]}")

        install_model_xml_remapper(operation_config=OperationCameraConfig())
        processed_xml = replay_utils.libero_utils.postprocess_model_xml(model_xml, {})
        tunable_xml, xml_poses = attach_operation_cameras_to_mocap(processed_xml)
        self.close()
        try:
            model = self._load_or_compile_model(
                tunable_xml,
                use_cache=demo_index == 0,
            )
        except ValueError as exc:
            patched_xml = _patch_scene4_xml(tunable_xml, exc)
            if patched_xml is None:
                raise
            model = self._load_or_compile_model(
                patched_xml,
                use_cache=demo_index == 0,
            )
        self.sim = MjSim(model)
        try:
            render_context = MjRenderContext(
                self.sim,
                offscreen=True,
                max_width=self.render_size,
                max_height=self.render_size,
            )
            render_context.vopt.geomgroup[0] = 0
            render_context.vopt.geomgroup[1] = 1
        except Exception:
            self.close()
            raise

        self.sim.reset()
        self.current_states = states
        self.demo_index = demo_index
        self.frame_index = 0
        self.camera_poses = previous_poses or xml_poses
        self._restore_frame(0)

    def _load_or_compile_model(
        self,
        model_xml: str,
        *,
        use_cache: bool,
    ) -> mujoco.MjModel:
        cache_path = self._model_cache_path(model_xml) if use_cache else None
        if cache_path is not None and cache_path.is_file():
            try:
                model = mujoco.MjModel.from_binary_path(os.fspath(cache_path))
            except Exception:
                cache_path.unlink(missing_ok=True)
            else:
                cache_path.touch()
                return model

        model = mujoco.MjModel.from_xml_string(model_xml)
        if cache_path is not None:
            self._save_cached_model(model, cache_path)
        return model

    def _model_cache_path(self, model_xml: str) -> Optional[Path]:
        if self.model_cache_dir is None or self.model_cache_limit_bytes <= 0:
            return None
        cache_key = hashlib.sha256(
            f"mujoco-{mujoco.__version__}\0{model_xml}".encode("utf-8")
        ).hexdigest()
        return self.model_cache_dir / f"{cache_key}.mjb"

    def _save_cached_model(self, model: mujoco.MjModel, cache_path: Path) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temp_path_str = tempfile.mkstemp(
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                dir=cache_path.parent,
            )
            os.close(file_descriptor)
            temp_path = Path(temp_path_str)
            try:
                mujoco.mj_saveModel(model, os.fspath(temp_path), None)
                os.replace(temp_path, cache_path)
            finally:
                temp_path.unlink(missing_ok=True)
            self._prune_model_cache(cache_path)
        except OSError:
            return

    def _prune_model_cache(self, keep_path: Path) -> None:
        cache_files = sorted(
            self.model_cache_dir.glob("*.mjb"),
            key=lambda path: path.stat().st_mtime,
        )
        total_size = sum(path.stat().st_size for path in cache_files)
        for cache_file in cache_files:
            if total_size <= self.model_cache_limit_bytes:
                break
            if cache_file == keep_path:
                continue
            file_size = cache_file.stat().st_size
            cache_file.unlink(missing_ok=True)
            total_size -= file_size

    def switch_demo(self, demo_index: int) -> None:
        """Rebuilds the selected demo XML while preserving world camera poses."""

        self._load_demo(int(demo_index), preserve_camera_poses=True)

    def switch_frame(self, frame_index: int) -> None:
        """Restores one recorded simulator state without changing cameras."""

        if self.current_states is None:
            raise RuntimeError("No task is loaded")
        if not 0 <= frame_index < len(self.current_states):
            raise ValueError(f"Frame index out of range: {frame_index}")
        self._restore_frame(int(frame_index))

    def _restore_frame(self, frame_index: int) -> None:
        self.sim.set_state_from_flattened(self.current_states[frame_index])
        self.frame_index = frame_index
        self._apply_camera_poses()

    def _apply_camera_poses(self) -> None:
        for camera_name, pose in self.camera_poses.items():
            body_name = f"camera_tuner_{camera_name}_mocap"
            self.sim.data.set_mocap_pos(body_name, np.asarray(pose.position))
            self.sim.data.set_mocap_quat(
                body_name, np.asarray(pose.quaternion_wxyz)
            )
        self.sim.forward()

    def adjust_camera(
        self,
        camera_name: str,
        translation: np.ndarray,
        rotation_degrees: np.ndarray,
    ) -> CameraPose:
        """Applies local-frame translation and pitch/yaw/roll rotation."""

        self._require_camera(camera_name)
        pose = self.camera_poses[camera_name]
        rotation = Rotation.from_quat(
            [
                pose.quaternion_wxyz[1],
                pose.quaternion_wxyz[2],
                pose.quaternion_wxyz[3],
                pose.quaternion_wxyz[0],
            ]
        )
        world_delta = rotation.apply(np.asarray(translation, dtype=np.float64))
        local_rotation = Rotation.from_euler(
            "xyz", np.asarray(rotation_degrees, dtype=np.float64), degrees=True
        )
        updated_rotation = rotation * local_rotation
        quat_xyzw = updated_rotation.as_quat()
        updated = CameraPose.from_mapping(
            {
                "position": (np.asarray(pose.position) + world_delta).tolist(),
                "quaternion_wxyz": [
                    quat_xyzw[3],
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                ],
            }
        )
        self.camera_poses[camera_name] = updated
        self._apply_camera_poses()
        return updated

    def set_camera_pose(self, camera_name: str, pose: CameraPose) -> CameraPose:
        """Sets one camera to an absolute world pose."""

        self._require_camera(camera_name)
        self.camera_poses[camera_name] = pose
        self._apply_camera_poses()
        return pose

    def reset_camera(self, camera_name: str) -> CameraPose:
        """Restores one camera to its untouched heuristic pose from demo zero."""

        self._require_camera(camera_name)
        self.camera_poses[camera_name] = self.initial_camera_poses[camera_name]
        self._apply_camera_poses()
        return self.camera_poses[camera_name]

    def render_jpeg(self, camera_name: str) -> bytes:
        """Renders one square RGB view as a JPEG byte string."""

        self._require_camera(camera_name)
        image = self.sim.render(
            camera_name=camera_name,
            width=self.render_size,
            height=self.render_size,
            depth=False,
        )
        image = np.flip(np.asarray(image, dtype=np.uint8), axis=0)
        output = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(output, format="JPEG", quality=90)
        return output.getvalue()

    @property
    def frame_count(self) -> int:
        """Returns the current demo length."""

        return self.demo_lengths[self.demo_index]

    def _require_camera(self, camera_name: str) -> None:
        if camera_name not in OPERATION_CAMERA_NAMES:
            raise ValueError(f"Unknown operation camera: {camera_name}")
        if camera_name not in self.camera_poses:
            raise RuntimeError("No task is loaded")
