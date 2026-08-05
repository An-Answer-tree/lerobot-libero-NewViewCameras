#!/usr/bin/env python3
"""Launch the LIBERO task camera tuning web app."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
for search_path in (REPO_ROOT, Path(__file__).resolve().parent, ROBOSUITE_ROOT):
    path_str = os.fspath(search_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from multiview_collect_demo.camera_tuner.app import main


if __name__ == "__main__":
    main()
