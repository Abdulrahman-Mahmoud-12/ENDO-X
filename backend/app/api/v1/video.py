"""POST /api/v1/predict/video

Thin by design, same rule as image.py: no model/tracker calls, no
per-frame orchestration — that lives in pipeline/ and services/. This
file only validates query params, calls video_service via Depends, and
shapes the response.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.schemas.prediction import VideoPredictionAPIResponse
from app.services.video_service import VideoService, get_video_service

router = APIRouter()


@router.post(
    "/predict/video",
    response_model=VideoPredictionAPIResponse,
)
async def predict_video(
    file: UploadFile = File(..., description="mp4/avi/mov endoscopy clip"),
    sample_rate: int = Query(default=1, ge=1, description="Process every Nth frame"),
    video_service: VideoService = Depends(get_video_service),
) -> VideoPredictionAPIResponse:
    return await video_service.predict_video(file=file, sample_rate=sample_rate)

