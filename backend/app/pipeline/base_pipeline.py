"""Shared orchestration for detection and segmentation pipelines.

Depends only on ``domain/interfaces`` types (``Detector``, ``Segmenter``),
never on the concrete ``PolypDetector``/``PolypSegmenter`` classes —
callers (``services/inference_service.py``) inject whatever implements the
Protocol via ``Depends`` on ``app.state``. This keeps pipeline/ decoupled
from the ultralytics/torch-specific model wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.config import Settings, get_settings
from app.domain.interfaces.detector import Detector
from app.domain.interfaces.segmenter import Segmenter
from app.schemas.prediction import Detection, SegmentationMask
from app.utils.image import crop_by_bbox


@dataclass
class PerObjectResult:
    """One detected (+ maybe segmented) object, paired together for convenience."""

    detection: Detection
    segmentation: SegmentationMask | None


@dataclass
class PipelineResult:
    detections: list[Detection] = field(default_factory=list)
    segmentations: list[SegmentationMask] = field(default_factory=list)
    per_object: list[PerObjectResult] = field(default_factory=list)


class BasePipeline:
    """detect -> crop-by-bbox -> segment per detection -> assemble.

    ``detector``/``segmenter`` are typed against the ``Detector``/
    ``Segmenter`` Protocols, so any conforming implementation works, not
    just ``PolypDetector``/``PolypSegmenter``.
    """

    def __init__(
        self,
        detector: Detector,
        segmenter: Segmenter,
        settings: Settings | None = None,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.settings = settings or get_settings()

    def run_detect_segment(self, image: np.ndarray) -> PipelineResult:
        """Run detection and segmentation according to ``segmentation_mode``.

        In ``independent`` mode both models receive the same full image and
        segmentation is still executed when detection returns no boxes.
        ``detected_regions`` preserves the original crop-based behavior for
        checkpoints trained specifically on detector crops.
        """
        detections = self.detector.predict(image)

        if self.settings.segmentation_mode == "independent":
            mask = self.segmenter.predict(image)
            mask.detection_index = -1
            return PipelineResult(
                detections=detections,
                segmentations=[mask],
                per_object=[
                    PerObjectResult(detection=detection, segmentation=None)
                    for detection in detections
                ],
            )

        if self.settings.segmentation_mode != "detected_regions":
            raise ValueError(
                "SEGMENTATION_MODE must be 'independent' or 'detected_regions'"
            )

        segmentations: list[SegmentationMask] = []
        per_object: list[PerObjectResult] = []

        for index, detection in enumerate(detections):
            try:
                crop = crop_by_bbox(image, detection.bbox, margin=self.settings.detection_roi_margin)
            except ValueError:
                per_object.append(PerObjectResult(detection=detection, segmentation=None))
                continue

            mask = self.segmenter.predict(crop)
            mask.detection_index = index

            segmentations.append(mask)
            per_object.append(PerObjectResult(detection=detection, segmentation=mask))

        return PipelineResult(detections=detections, segmentations=segmentations, per_object=per_object)