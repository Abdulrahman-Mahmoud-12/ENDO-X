"""Ensures ``backend/`` is importable as the ``app`` package root when
running pytest from the repository root (e.g. ``pytest tests/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
