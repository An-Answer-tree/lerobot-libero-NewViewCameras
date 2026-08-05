"""Web-based task camera calibration tool."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    """Lazily exposes the Flask factory without loading MuJoCo for CLI help."""

    if name == "create_app":
        from multiview_collect_demo.camera_tuner.app import create_app

        return create_app
    raise AttributeError(name)
