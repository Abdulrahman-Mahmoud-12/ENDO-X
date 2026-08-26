"""FastAPI lifespan management.

Loads the trained detector and segmenter exactly once when the application
starts (never per-request), and releases resources on shutdown.

NOTE: the actual model wrapper classes (``PolypDetector`` / ``PolypSegmenter``)
are implemented in Phase 1 (``app/models/detector.py`` and
``app/models/segmenter.py``). This phase wires the loading *lifecycle* —
device resolution, weight-file validation, timing, and app.state population —
so every later phase has a stable place to plug the real model classes into.
Until Phase 1 lands, ``app.state.detector`` / ``app.state.segmenter`` stay
``None`` and the health endpoint reports them as ``not_loaded``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str) -> str:
    """Resolve the 'auto' device setting to a concrete torch device string."""
    if requested_device != "auto":
        return requested_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError: 
        logger.warning("torch is not importable; falling back to CPU")
    return "cpu"


def _load_detector(model_path: str, device: str) -> Any | None:
    """Load the YOLO polyp detector. Returns None until Phase 1 implements it."""
    try:
        from app.models.detector import PolypDetector
    except ModuleNotFoundError:
        logger.warning(
            "app.models.detector.PolypDetector not implemented yet (Phase 1). "
            "Skipping detector load; /health will report it as not_loaded."
        )
        return None

    detector = PolypDetector(model_path=model_path, device=device)
    detector.load()
    return detector


def _load_segmenter(
    model_path: str,
    encoder_name: str,
    architecture: str,
    device: str,
) -> Any | None:
    """Load the U-Net/U-Net++ polyp segmenter. Returns None until Phase 1 implements it."""
    try:
        from app.models.segmenter import PolypSegmenter
    except ModuleNotFoundError:
        logger.warning(
            "app.models.segmenter.PolypSegmenter not implemented yet (Phase 1). "
            "Skipping segmenter load; /health will report it as not_loaded."
        )
        return None

    segmenter = PolypSegmenter(
        model_path=model_path,
        encoder_name=encoder_name,
        architecture=architecture,
        device=device,
    )
    segmenter.load()
    return segmenter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load models on startup, clean up on shutdown."""
    settings = get_settings()
    device = resolve_device(settings.device)
    logger.info("Resolved inference device: %s", device)

    app.state.device = device
    app.state.model_status = {"detector": "not_loaded", "segmenter": "not_loaded"}
    app.state.detector = None
    app.state.segmenter = None

    start = time.perf_counter()

    if not settings.detector_model_path.exists():
        logger.error(
            "Detector weights not found at %s. Set DETECTOR_MODEL_PATH in .env "
            "once the trained checkpoint is in place.",
            settings.detector_model_path,
        )
    else:
        app.state.detector = _load_detector(str(settings.detector_model_path), device)
        app.state.model_status["detector"] = "loaded" if app.state.detector else "not_loaded"

    if not settings.segmenter_model_path.exists():
        logger.error(
            "Segmenter weights not found at %s. Set SEGMENTER_MODEL_PATH in .env "
            "once the trained checkpoint is in place.",
            settings.segmenter_model_path,
        )
    else:
        app.state.segmenter = _load_segmenter(
            str(settings.segmenter_model_path),
            settings.segmenter_encoder_name,
            settings.segmenter_architecture,
            device,
        )
        app.state.model_status["segmenter"] = "loaded" if app.state.segmenter else "not_loaded"

    elapsed = time.perf_counter() - start
    logger.info(
        "Startup complete in %.2fs — detector=%s, segmenter=%s",
        elapsed,
        app.state.model_status["detector"],
        app.state.model_status["segmenter"],
    )

    yield

    logger.info("Shutting down — releasing model resources")
    app.state.detector = None
    app.state.segmenter = None
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
