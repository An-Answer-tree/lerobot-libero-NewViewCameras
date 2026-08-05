"""Test path setup for repository-local scripts and robosuite."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (
    REPO_ROOT,
    REPO_ROOT / "scripts",
    REPO_ROOT / "third_party" / "robosuite",
):
    path_str = str(search_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
