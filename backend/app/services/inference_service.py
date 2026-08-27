"""Thin coordinator between the API layer and the pipeline.

Builds an ``ImagePipeline`` from whatever detector/segmenter are on
``app.state`` (populated once at startup by ``core/lifespan.py``) and just
calls it — no detect/crop/segment orchestration lives here, that's all in
``pipeline/``.
"""

from __future__ import annotations

import numpy as np
from fastapi import Depends, Request

from app.core.exceptions import ModelNotLoadedError
from app.pipeline.image_pipeline import ImagePipeline, ImagePipelineResult


def get_image_pipeline(request: Request) -> ImagePipeline:
    """FastAPI dependency: build an ``ImagePipeline`` from ``app.state``.

    Raises ``ModelNotLoadedError`` (mapped to a 503 by core/exceptions.py)
    if lifespan.py didn't manage to load both models, so callers get a
    clean, typed error instead of an ``AttributeError`` on ``None.predict``.
    """
    detector = getattr(request.app.state, "detector", None)
    segmenter = getattr(request.app.state, "segmenter", None)

    if detector is None or segmenter is None:
        missing = [name for name, obj in (("detector", detector), ("segmenter", segmenter)) if obj is None]
        raise ModelNotLoadedError(
            f"Model(s) not loaded: {', '.join(missing)}. Check GET /api/v1/health."
        )

    return ImagePipeline(detector=detector, segmenter=segmenter)


class InferenceService:
    """Wraps ``ImagePipeline`` so ``image_service.py`` has one thing to call."""

    def __init__(self, pipeline: ImagePipeline) -> None:
        self.pipeline = pipeline

    def run_image_inference(self, image: np.ndarray) -> ImagePipelineResult:
        return self.pipeline.run(image)


def get_inference_service(pipeline: ImagePipeline = Depends(get_image_pipeline)) -> InferenceService:
    return InferenceService(pipeline=pipeline)
