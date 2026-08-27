"""Image-specific pipeline: single image in, full per-object stats out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.base_pipeline import BasePipeline, PipelineResult


@dataclass
class ObjectStats:
    """Computed geometry stats for one detected(+segmented) object."""

    detection_index: int
    mask_area_pixels: int
    bbox_width: float
    bbox_height: float
    aspect_ratio: float


@dataclass
class ImagePipelineResult:
    pipeline_result: PipelineResult
    object_stats: list[ObjectStats] = field(default_factory=list)
    inference_time: float = 0.0  # seconds

    @property
    def detections(self):
        return self.pipeline_result.detections

    @property
    def segmentations(self):
        return self.pipeline_result.segmentations


class ImagePipeline(BasePipeline):
    """Runs detect -> segment on one image and adds per-object geometry stats."""

    def run(self, image: np.ndarray) -> ImagePipelineResult:
        start = time.perf_counter()
        result = self.run_detect_segment(image)

        stats: list[ObjectStats] = []
        for index, obj in enumerate(result.per_object):
            mask_area = obj.segmentation.mask_area_pixels if obj.segmentation else 0
            stats.append(
                ObjectStats(
                    detection_index=index,
                    mask_area_pixels=mask_area,
                    bbox_width=obj.detection.bbox.width,
                    bbox_height=obj.detection.bbox.height,
                    aspect_ratio=obj.detection.bbox.aspect_ratio,
                )
            )

        inference_time = time.perf_counter() - start
        return ImagePipelineResult(pipeline_result=result, object_stats=stats, inference_time=inference_time)
