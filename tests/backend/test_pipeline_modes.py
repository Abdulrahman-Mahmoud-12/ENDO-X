"""Tests for independent versus detector-region segmentation orchestration."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from app.core.config import Settings
from app.pipeline.base_pipeline import BasePipeline
from app.schemas.prediction import BoundingBox, Detection, SegmentationMask


def _mask(height: int, width: int) -> SegmentationMask:
    pixels = np.full((height, width), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return SegmentationMask(
        mask_area_pixels=height * width,
        mask_encoding=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


class Detector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def predict(self, image: np.ndarray) -> list[Detection]:
        return self.detections


class Segmenter:
    def __init__(self) -> None:
        self.inputs: list[tuple[int, int]] = []

    def predict(self, image: np.ndarray) -> SegmentationMask:
        self.inputs.append(image.shape[:2])
        return _mask(*image.shape[:2])


def test_independent_mode_segments_even_without_detection() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    segmenter = Segmenter()
    result = BasePipeline(
        detector=Detector([]),
        segmenter=segmenter,
        settings=Settings(segmentation_mode="independent"),
    ).run_detect_segment(image)

    assert segmenter.inputs == [(80, 120)]
    assert len(result.segmentations) == 1
    assert result.segmentations[0].detection_index == -1


def test_detected_region_mode_segments_each_detection() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    detection = Detection(confidence=0.9, bbox=BoundingBox(x_min=20, y_min=20, x_max=60, y_max=60))
    segmenter = Segmenter()
    result = BasePipeline(
        detector=Detector([detection]),
        segmenter=segmenter,
        settings=Settings(segmentation_mode="detected_regions", detection_roi_margin=0),
    ).run_detect_segment(image)

    assert segmenter.inputs == [(40, 40)]
    assert result.segmentations[0].detection_index == 0