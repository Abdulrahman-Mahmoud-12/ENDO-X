"""Detector tests — load the real trained checkpoint, run one image, check schema.

No FastAPI import anywhere in this chain (Settings -> PolypDetector -> Detection).
Skips (rather than fails) if the weights file isn't present, so CI without the
large checkpoint doesn't red the build; it fails loudly in dev if you point
DETECTOR_MODEL_PATH somewhere wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import get_settings
from app.models.detector import PolypDetector
from app.schemas.prediction import Detection

settings = get_settings()

pytestmark = pytest.mark.skipif(
    not settings.detector_model_path.exists(),
    reason=f"Detector weights not found at {settings.detector_model_path}; "
    "set DETECTOR_MODEL_PATH to run this test.",
)


@pytest.fixture(scope="module")
def detector() -> PolypDetector:
    return PolypDetector(
        model_path=settings.detector_model_path, device="cpu", settings=settings
    ).load()


@pytest.fixture(scope="module")
def sample_image() -> np.ndarray:
    """A synthetic RGB frame; swap for a real endoscopy still if you have one on disk."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def test_detector_loads(detector: PolypDetector) -> None:
    assert detector.is_loaded


def test_predict_returns_detection_list(detector: PolypDetector, sample_image: np.ndarray) -> None:
    detections = detector.predict(sample_image)
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)


def test_detection_schema_and_ranges(detector: PolypDetector, sample_image: np.ndarray) -> None:
    detections = detector.predict(sample_image)
    for det in detections:
        assert isinstance(det.class_name, str) and det.class_name
        assert 0.0 <= det.confidence <= 1.0
        assert det.confidence >= settings.detection_confidence_threshold
        bbox = det.bbox
        assert bbox.x_max > bbox.x_min
        assert bbox.y_max > bbox.y_min
        assert bbox.width > 0
        assert bbox.height > 0