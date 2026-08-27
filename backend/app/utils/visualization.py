"""Turns raw Detection/SegmentationMask objects into a viewable overlay image.

Framework-light: no FastAPI imports, so it can be unit tested by comparing
pixel/file output rather than spinning up the app.
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.schemas.prediction import Detection, SegmentationMask
from app.utils.image import compute_crop_region

logger = logging.getLogger(__name__)

_BOX_COLOR = (0, 255, 0)  # RGB
_MASK_COLOR = (255, 64, 64)  # RGB


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes + confidence labels. Returns a copy; never mutates ``image``."""
    annotated = image.copy()
    for detection in detections:
        b = detection.bbox
        p1, p2 = (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max))
        cv2.rectangle(annotated, p1, p2, _BOX_COLOR, thickness=2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _BOX_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return annotated


def _decode_mask_png(mask_b64: str) -> np.ndarray:
    raw = base64.b64decode(mask_b64)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Failed to decode segmentation mask PNG")
    return mask


def draw_segmentation_overlay(
    image: np.ndarray,
    detections: list[Detection],
    segmentations: list[SegmentationMask],
    settings: Settings,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend each mask back onto the full image at its detection's crop region.

    Each ``SegmentationMask`` is sized to its *expanded* crop (detection
    bbox + ``settings.detection_roi_margin``, per ``pipeline/base_pipeline.py``)
    rather than the raw detection bbox — the crop region is recomputed here
    with the same margin so placement lines up. If the pipeline's cropping
    ever changes, update this alongside it.
    """
    annotated = image.copy()
    overlay = image.copy()

    for segmentation in segmentations:
        if segmentation.detection_index >= len(detections) or segmentation.detection_index < 0:
            logger.warning(
                "Skipping segmentation with out-of-range detection_index=%s", segmentation.detection_index
            )
            continue
        detection = detections[segmentation.detection_index]

        try:
            mask = _decode_mask_png(segmentation.mask_encoding)
        except ValueError:
            logger.warning("Skipping unreadable mask for detection_index=%s", segmentation.detection_index)
            continue

        x_min, y_min, x_max, y_max = compute_crop_region(
            image.shape, detection.bbox, margin=settings.detection_roi_margin
        )
        region_h, region_w = y_max - y_min, x_max - x_min
        if region_h <= 0 or region_w <= 0:
            continue
        if mask.shape != (region_h, region_w):
            mask = cv2.resize(mask, (region_w, region_h), interpolation=cv2.INTER_NEAREST)

        region = overlay[y_min:y_max, x_min:x_max]
        colored = np.zeros_like(region)
        colored[:] = _MASK_COLOR
        mask_bool = mask > 0
        region[mask_bool] = colored[mask_bool]
        overlay[y_min:y_max, x_min:x_max] = region

    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, dst=annotated)
    return annotated


def render_overlay(
    image: np.ndarray,
    detections: list[Detection],
    segmentations: list[SegmentationMask],
    settings: Settings,
) -> np.ndarray:
    """Combine the mask overlay and box/label drawing into one annotated image."""
    with_masks = draw_segmentation_overlay(image, detections, segmentations, settings)
    return draw_detections(with_masks, detections)


def save_overlay_image(
    image: np.ndarray,
    detections: list[Detection],
    segmentations: list[SegmentationMask],
    settings: Settings,
) -> str:
    """Render the overlay and save it under ``storage/outputs``.

    Returns a URL path relative to how ``main.py`` mounts the outputs dir
    (``app.mount("/storage/outputs", StaticFiles(directory=settings.outputs_dir))``),
    so the frontend can load it directly by URL.
    """
    annotated = render_overlay(image, detections, segmentations, settings)
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    settings.ensure_storage_dirs()
    filename = f"{uuid.uuid4().hex}.png"
    output_path = Path(settings.outputs_dir) / filename

    success = cv2.imwrite(str(output_path), bgr)
    if not success:
        raise RuntimeError(f"Failed to write overlay image to {output_path}")

    return f"/storage/outputs/{filename}"


def extract_polygon_from_mask_b64(mask_b64: str) -> list[list[float]]:
    """Decode a base64 PNG mask and return its largest external contour as [[x, y], ...].

    Used only at the API-response boundary (``schemas/prediction.SegmentationOut.polygon``
    isn't part of the internal ``SegmentationMask`` schema) — the internal
    schema keeps the raw PNG encoding since it's lossless and cheap to
    produce; the polygon is derived from it on demand for clients that want
    contour points instead of a bitmap. Returns ``[]`` if the mask is empty.
    """
    mask = _decode_mask_png(mask_b64)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return [[float(pt[0][0]), float(pt[0][1])] for pt in largest]