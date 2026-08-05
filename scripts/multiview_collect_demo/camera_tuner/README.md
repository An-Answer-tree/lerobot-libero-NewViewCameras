# LIBERO task camera tuner

This module calibrates four operation cameras independently for every task in
`libero_spatial`, `libero_goal`, `libero_object`, and `libero_10`. The sequence
is deterministic and contains exactly 40 tasks. Each scene is reconstructed
from an official HDF5 demo's `model_file`, then restored to a selected recorded
state before rendering.

## Setup and launch

Install the repository dependencies in the LIBERO Python 3.10 environment, then
run:

```bash
python scripts/tune_multiview_cameras.py
```

The server validates all 40 source files before creating a MuJoCo session. Use
`--source-root` when the datasets are not under `get_libero_path("datasets")`.
Other useful options are `--config`, `--render-size`, `--host`, and `--port`.
The defaults are a `512x512` render and `127.0.0.1:19985`.

The service intentionally binds to loopback. From a local machine, forward it
through SSH and open `http://127.0.0.1:19985`:

```bash
ssh -L 19985:127.0.0.1:19985 <remote-host>
```

## Controls

- Select one of the four camera tabs or thumbnails before adjusting it.
- `W` / `S` move forward and backward; `A` / `D` move left and right;
  `Q` / `E` move down and up in the active camera's local frame.
- Drag the main preview to change yaw and pitch. Use the mouse wheel to move
  forward or backward.
- Fine, medium, and fast set the translation increment. Holding Shift while
  using a movement key multiplies the current increment by five.
- Reset restores only the active camera to the pose loaded for the task.
- Demo changes rebuild that demo's XML while preserving all current camera
  poses. Frame changes only restore `states[t]`.
- Confirm writes all four poses together and advances to the next unconfirmed
  task. Leaving with unsaved changes triggers a browser warning.

## YAML contract

The default file is
`scripts/multiview_collect_demo/configs/task_operation_cameras.yaml`. It uses
this schema:

```yaml
schema_version: 1
suites:
  libero_spatial:
    pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate:
      operation_backview:
        position: [1.0, 2.0, 3.0]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
      operation_leftview:
        position: [1.0, 2.0, 3.0]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
      operation_rightview:
        position: [1.0, 2.0, 3.0]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
      operation_topview:
        position: [1.0, 2.0, 3.0]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
```

A task entry is valid only when all four named cameras are present. Positions
are world coordinates and orientations are normalized MuJoCo `wxyz`
quaternions. Confirmation writes a temporary file in the destination directory,
flushes it, and replaces the YAML atomically. A partial task is never treated as
confirmed.

To resume, launch the same command with the same `--config`. The first
unconfirmed task opens automatically; confirmed tasks remain available in the
left task list.

## Multiview generator integration

The replay generator reads the same file by default:

```bash
python scripts/multiview_collect_demo.py \
  --task-camera-config scripts/multiview_collect_demo/configs/task_operation_cameras.yaml \
  --source-root /path/to/LIBERO-datasets \
  --output-root /path/to/output
```

For a confirmed task, the stored world poses replace the four heuristic poses
exactly. Unconfirmed tasks continue to use scene-derived heuristic poses.
`--no-operation-cameras` has the highest priority and disables both calibrated
and heuristic operation cameras. The default output now contains 16 RGB views:
10 trajectory cameras, four operation cameras, `agentview`, and
`robot0_eye_in_hand`.

## Architecture and validation

The tuner keeps Flask, HDF5/MuJoCo session ownership, shared YAML validation,
and static UI code in separate modules. Flask is deliberately single-threaded,
so all MuJoCo and EGL calls remain on one execution thread. A monotonically
increasing revision travels with every state mutation and render request; the UI
discards an image response if a newer state has already arrived.

The intended validation matrix is:

| Check | Heuristic baseline | Calibrated result | Record |
| --- | --- | --- | --- |
| Scene coverage | Four inferred poses | Four saved poses | JPEGs and pose YAML |
| Demo switch | Rebuilt demo XML | Same world pose | Before/after pose values |
| Frame switch | Restored source state | Same camera pose | Frame index and pose |
| Dataset replay | 16 default views | Exact task overrides | Camera info datasets |
| UI layout | 1280x800 | 1600x900 | Playwright screenshots |

The calibration run should answer whether each view keeps the manipulated
objects visible across representative frames, whether one pose is robust across
demos of the same task, and where the heuristic baseline fails. Record the demo
and frame used for judgment, inspect early/middle/late trajectory states, and
review failures by suite after regenerated datasets are available.

Run focused checks with:

```bash
python -m pytest -q \
  tests/test_task_camera_config.py \
  tests/test_camera_injection.py \
  tests/test_camera_tuner_api.py \
  tests/test_multiview_camera_config.py
```
