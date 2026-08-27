"""Framework-light contract for anything that detects polyps in a frame.

Deliberately has zero FastAPI/HTTP imports so it (and anything implementing
it) can be exercised from a plain Python script or pytest, with no web
server running.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.schemas.prediction import Detection


@runtime_checkable
class Detector(Protocol):
    """Contract: given one image, return zero or more polyp detections."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single RGB image (H, W, 3) uint8 array.

        Returns a list of ``Detection`` objects (empty list if none found).
        Implementations are expected to already be loaded/warm — callers
        should not have to know about ``load()`` to satisfy this contract.
        """
        ...
