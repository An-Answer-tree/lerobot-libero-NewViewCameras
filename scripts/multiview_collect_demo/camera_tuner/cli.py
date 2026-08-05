"""Lightweight command-line entrypoint for the camera tuner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

default_config_path = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "task_operation_cameras.yaml"
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the parser without importing LIBERO or MuJoCo."""

    parser = argparse.ArgumentParser(
        description="Tune task-level LIBERO operation cameras in a web browser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="LIBERO dataset root; uses the configured LIBERO path when omitted",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help="YAML file used to save and resume task camera poses",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19985,
        help="HTTP listen port",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="square preview image size in pixels",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path("/tmp/libero-camera-tuner-models"),
        help="directory for compiled demo_0 MuJoCo models",
    )
    parser.add_argument(
        "--model-cache-limit-gb",
        type=float,
        default=12.0,
        help="maximum compiled-model cache size; zero disables caching",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parses arguments before loading the rendering stack and starts Flask."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    from multiview_collect_demo.camera_tuner.app import create_app

    try:
        app = create_app(
            source_root=args.source_root,
            config_path=args.config,
            render_size=args.render_size,
            model_cache_dir=args.model_cache_dir,
            model_cache_limit_gb=args.model_cache_limit_gb,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    controller = app.config["CAMERA_TUNER_CONTROLLER"]
    try:
        app.run(host=args.host, port=args.port, threaded=False, debug=False)
    finally:
        controller.close()
