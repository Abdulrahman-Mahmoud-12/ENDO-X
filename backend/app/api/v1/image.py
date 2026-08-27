"""POST /api/v1/predict/image

Thin by design: no model calls, no detect/crop/segment orchestration —
that lives in ``pipeline/``. This file only validates query params, calls
``image_service`` via ``Depends``, and shapes the response.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.schemas.prediction import DetectionOut, ImagePredictionAPIResponse, SegmentationOut
from app.services.image_service import ImageService, get_image_service
from app.utils.visualization import extract_polygon_from_mask_b64

router = APIRouter()


@router.post("/predict/image", response_model=ImagePredictionAPIResponse)
async def predict_image(
    file: UploadFile = File(..., description="jpg/jpeg/png endoscopy frame"),
    confidence_threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional per-request override. Can only raise the effective "
            "threshold above Settings.detection_confidence_threshold, not "
            "lower it — see image_service._filter_by_confidence."
        ),
    ),
    return_overlay: bool = Query(default=True, description="Render + save the annotated overlay image"),
    image_service: ImageService = Depends(get_image_service),
) -> ImagePredictionAPIResponse:
    result = await image_service.predict_image(
        file=file,
        confidence_threshold=confidence_threshold,
        return_overlay=return_overlay,
    )

    # DetectionOut.class_name has alias="class" with populate_by_name=True,
    # so constructing it by the Python-safe field name here works fine —
    # it still serializes as "class" in the response JSON.
    detections_out = [
        DetectionOut(
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=[d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
        )
        for d in result.detections
    ]
    segmentations_out = [
        SegmentationOut(
            detection_index=s.detection_index,
            mask_area_px=s.mask_area_pixels,
            polygon=extract_polygon_from_mask_b64(s.mask_encoding),
        )
        for s in result.segmentations
    ]

    return ImagePredictionAPIResponse(
        status="success",
        detections=detections_out,
        segmentations=segmentations_out,
        overlay_image_url=result.annotated_image_url,
        inference_time_ms=result.inference_time * 1000,
    )
