"""Tests for calibrated and heuristic operation camera injection."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from multiview_collect_demo.camera_injection import (
    DEFAULT_OPERATION_CAMERA_NAMES,
    OperationCameraConfig,
    _generate_operation_camera_specs,
    _inject_operation_cameras,
)
from multiview_collect_demo.camera_tuner.session import (
    MujocoCameraSession,
    attach_operation_cameras_to_mocap,
)
from multiview_collect_demo.task_camera_config import CameraPose

MODEL_XML = """
<mujoco>
  <worldbody>
    <camera name="frontview" pos="1 0 1.2" quat="0.9238795 0 0.3826834 0" fovy="47"/>
    <camera name="agentview" pos="1 0 1.2" quat="0.9238795 0 0.3826834 0"/>
    <camera name="sideview" pos="0 1 1.1" quat="0.7071068 0.5 0.5 0"/>
    <camera name="birdview" pos="0 0 2.4" quat="1 0 0 0"/>
  </worldbody>
</mujoco>
"""


def test_default_operation_camera_set_has_only_four_views() -> None:
    specs = _generate_operation_camera_specs(
        ET.fromstring(MODEL_XML), OperationCameraConfig()
    )

    assert set(DEFAULT_OPERATION_CAMERA_NAMES) == {"top", "left", "right", "back"}
    assert {spec.name for spec in specs} == set(
        DEFAULT_OPERATION_CAMERA_NAMES.values()
    )
    assert "operation_leftbackview" not in _inject_operation_cameras(
        MODEL_XML, OperationCameraConfig()
    )
    assert "operation_rightbackview" not in _inject_operation_cameras(
        MODEL_XML, OperationCameraConfig()
    )


def test_manual_pose_overrides_exact_values_and_unset_views_fall_back() -> None:
    manual_pose = CameraPose.from_mapping(
        {
            "position": [9.25, -2.5, 4.75],
            "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5],
        }
    )
    config = OperationCameraConfig(
        camera_poses={"operation_backview": manual_pose}
    )
    specs = {
        spec.name: spec
        for spec in _generate_operation_camera_specs(ET.fromstring(MODEL_XML), config)
    }

    np.testing.assert_allclose(
        specs["operation_backview"].pos, manual_pose.position, atol=0
    )
    np.testing.assert_allclose(
        specs["operation_backview"].quat, manual_pose.quaternion_wxyz, atol=0
    )
    assert not np.allclose(specs["operation_leftview"].pos, manual_pose.position)


def test_all_operation_cameras_are_attached_to_mocap_bodies() -> None:
    injected = _inject_operation_cameras(MODEL_XML, OperationCameraConfig())
    tunable_xml, poses = attach_operation_cameras_to_mocap(injected)
    root = ET.fromstring(tunable_xml)
    bodies = [
        body
        for body in root.find("worldbody").findall("body")
        if body.get("mocap") == "true"
    ]

    assert len(bodies) == 4
    assert set(poses) == set(DEFAULT_OPERATION_CAMERA_NAMES.values())
    assert {
        body.find("camera").get("name") for body in bodies
    } == set(DEFAULT_OPERATION_CAMERA_NAMES.values())
    assert all(body.find("camera").get("pos") == "0 0 0" for body in bodies)


def test_model_cache_key_and_size_limit(tmp_path: Path) -> None:
    session = MujocoCameraSession(
        model_cache_dir=tmp_path,
        model_cache_limit_gb=20 / 1024**3,
    )
    first_path = session._model_cache_path("<mujoco model='first'/>")
    second_path = session._model_cache_path("<mujoco model='second'/>")

    assert first_path != second_path
    assert first_path.suffix == ".mjb"

    first_path.write_bytes(b"a" * 12)
    second_path.write_bytes(b"b" * 12)
    session._prune_model_cache(second_path)

    assert not first_path.exists()
    assert second_path.exists()
