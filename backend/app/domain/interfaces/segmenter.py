"""Framework-light contract for anything that segments an input image.

Deliberately has zero FastAPI/HTTP imports so it (and anything implementing
it) can be exercised from a plain Python script or pytest, with no web
server running.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.schemas.prediction import SegmentationMask


@runtime_checkable
class Segmenter(Protocol):
    """Contract: given an RGB image or crop, return its mask."""

    def predict(self, image_crop: np.ndarray) -> SegmentationMask:
        """Run segmentation on an RGB image/crop (H, W, 3) uint8 array.

        Returns a ``SegmentationMask`` sized back to ``image_crop``'s
        original dimensions. Implementations are expected to already be
        loaded/warm.
        """
        ...