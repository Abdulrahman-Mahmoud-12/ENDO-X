"""Health-check endpoint reporting API and AI model status."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.schemas.prediction import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(request: Request) -> HealthResponse:
    """Return service status plus whether the detector/segmenter are loaded."""
    settings = get_settings()
    model_status: dict[str, str] = getattr(
        request.app.state, "model_status", {"detector": "not_loaded", "segmenter": "not_loaded"}
    )
    detector_status = model_status.get("detector", "not_loaded")
    segmenter_status = model_status.get("segmenter", "not_loaded")
    overall = "healthy" if detector_status == "loaded" and segmenter_status == "loaded" else "degraded"

    return HealthResponse(
        status=overall,
        app_name=settings.app_name,
        app_version=settings.app_version,
        device=getattr(request.app.state, "device", "unknown"),
        detector=detector_status,
        segmenter=segmenter_status,
    )
