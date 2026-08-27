"""Segmenter tests — load the real trained checkpoint, run one crop, check mask shape/range.

No FastAPI import anywhere in this chain (Settings -> PolypSegmenter -> SegmentationMask).
Skips (rather than fails) if the weights file isn't present.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.models.segmenter import PolypSegmenter
from app.schemas.prediction import SegmentationMask

settings = get_settings()

pytestmark = pytest.mark.skipif(
    not settings.segmenter_model_path.exists(),
    reason=f"Segmenter weights not found at {settings.segmenter_model_path}; "
    "set SEGMENTER_MODEL_PATH to run this test.",
)


@pytest.fixture(scope="module")
def segmenter() -> PolypSegmenter:
    return PolypSegmenter(
        model_path=settings.segmenter_model_path,
        encoder_name=settings.segmenter_encoder_name,
        architecture=settings.segmenter_architecture,
        device="cpu",
        settings=settings,
    ).load()


@pytest.fixture(scope="module")
def sample_crop() -> np.ndarray:
    """A synthetic RGB crop; swap for a real detected-polyp crop if you have one on disk."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(180, 220, 3), dtype=np.uint8)


def test_segmenter_loads(segmenter: PolypSegmenter) -> None:
    assert segmenter.is_loaded


def test_predict_returns_segmentation_mask(segmenter: PolypSegmenter, sample_crop: np.ndarray) -> None:
    result = segmenter.predict(sample_crop, detection_index=0)
    assert isinstance(result, SegmentationMask)
    assert result.detection_index == 0
    assert result.mask_area_pixels >= 0
    assert isinstance(result.mask_encoding, str) and result.mask_encoding


def test_mask_shape_matches_crop_and_is_binary(segmenter: PolypSegmenter, sample_crop: np.ndarray) -> None:
    result = segmenter.predict(sample_crop, detection_index=0)

    png_bytes = base64.b64decode(result.mask_encoding)
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    assert mask is not None
    assert mask.shape == sample_crop.shape[:2]
    unique_values = set(np.unique(mask).tolist())
    assert unique_values <= {0, 255}
    assert result.mask_area_pixels == int((mask > 0).sum())