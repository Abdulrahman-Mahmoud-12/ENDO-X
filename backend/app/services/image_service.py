"""Validates the uploaded image, decodes it, runs inference, and assembles
the internal response schema. Detect/crop/segment orchestration lives in
``pipeline/`` — this file only wires that output into ``ImagePredictionResponse``.
"""

from __future__ import annotations

from fastapi import Depends, UploadFile

from app.core.config import Settings, get_settings
from app.core.exceptions import DecodeError, FileTooLargeError, InvalidFileTypeError
from app.schemas.prediction import Detection, ImagePredictionResponse, SegmentationMask
from app.services.inference_service import InferenceService, get_inference_service
from app.utils.image import decode_upload_bytes
from app.utils.visualization import save_overlay_image


async def _read_and_validate(file: UploadFile, settings: Settings) -> bytes:
    """Enforce extension + size limits before any decoding is attempted."""
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if suffix not in settings.allowed_image_extensions_list:
        raise InvalidFileTypeError(
            f"Unsupported file type {suffix or '(none)'!r}; allowed: {settings.allowed_image_extensions_list}"
        )

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise FileTooLargeError(f"Image is {size_mb:.1f}MB, exceeds the {settings.max_image_size_mb}MB limit")
    return data


def _filter_by_confidence(
    detections: list[Detection], threshold: float | None
) -> tuple[list[Detection], dict[int, int]]:
    """Best-effort query-param override.

    Returns the filtered detections plus a map of {old_index: new_index}
    so callers can remap segmentations' ``detection_index`` to stay
    consistent with the filtered list.
    """
    if threshold is None:
        return detections, {i: i for i in range(len(detections))}

    kept = [(i, d) for i, d in enumerate(detections) if d.confidence >= threshold]
    index_map = {old_i: new_i for new_i, (old_i, _) in enumerate(kept)}
    return [d for _, d in kept], index_map


def _remap_segmentations(
    segmentations: list[SegmentationMask], index_map: dict[int, int]
) -> list[SegmentationMask]:
    remapped: list[SegmentationMask] = []
    for seg in segmentations:
        if seg.detection_index not in index_map:
            continue
        remapped.append(seg.model_copy(update={"detection_index": index_map[seg.detection_index]}))
    return remapped


class ImageService:
    def __init__(self, inference_service: InferenceService, settings: Settings) -> None:
        self.inference_service = inference_service
        self.settings = settings

    async def predict_image(
        self,
        file: UploadFile,
        confidence_threshold: float | None,
        return_overlay: bool,
    ) -> ImagePredictionResponse:
        data = await _read_and_validate(file, self.settings)

        try:
            image = decode_upload_bytes(data)
        except ValueError as exc:
            raise DecodeError(str(exc)) from exc

        result = self.inference_service.run_image_inference(image)

        detections, index_map = _filter_by_confidence(result.detections, confidence_threshold)
        segmentations = _remap_segmentations(result.segmentations, index_map)

        overlay_url: str | None = None
        if return_overlay:
            overlay_url = save_overlay_image(
                image=image,
                detections=detections,
                segmentations=segmentations,
                settings=self.settings,
            )

        return ImagePredictionResponse(
            detections=detections,
            segmentations=segmentations,
            num_polyps=len(detections),
            inference_time=result.inference_time,
            annotated_image_url=overlay_url,
        )


def get_image_service(
    inference_service: InferenceService = Depends(get_inference_service),
    settings: Settings = Depends(get_settings),
) -> ImageService:
    return ImageService(inference_service=inference_service, settings=settings)