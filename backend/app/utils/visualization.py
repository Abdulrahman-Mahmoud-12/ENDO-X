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

_BOX_COLOR = (255, 0, 0)  # RGB

# Segmentation contour/tint color. Cyan (not red/pink) so it reads clearly
# against GI tissue tones and doesn't visually suggest "bleeding" or
# "danger" the way a red overlay would. High-contrast, low-ambiguity color
# choices like this are standard practice for clinical AI overlays.
_CONTOUR_COLOR = (0, 255, 255)  # RGB, cyan
_CONTOUR_THICKNESS = 2
_FILL_COLOR = _CONTOUR_COLOR
_FILL_ALPHA = 0.2  # light tint — low enough that tissue underneath stays visible


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
    alpha: float = _FILL_ALPHA,
) -> np.ndarray:
    """Outline each mask's boundary with a solid contour and lightly tint its interior.

    Previously this fully recolored every mask pixel and alpha-blended the
    whole thing at 0.45 — heavy enough to obscure the tissue underneath,
    which defeats the point of a clinical review overlay (the reviewer
    needs to see the actual polyp, not just a colored blob standing in for
    it). Now: a solid ``_CONTOUR_COLOR`` line traces the segmentation
    boundary (via ``cv2.findContours`` + ``cv2.drawContours``), and only a
    light ``_FILL_ALPHA``-opacity tint fills the interior, so tissue detail
    stays visible through it.

    Each ``SegmentationMask`` is sized to its *expanded* crop (detection
    bbox + ``settings.detection_roi_margin``, per ``pipeline/base_pipeline.py``)
    rather than the raw detection bbox — the crop region is recomputed here
    with the same margin so placement lines up. If the pipeline's cropping
    ever changes, update this alongside it.
    """
    annotated = image.copy()

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

        # Contours are found in the crop's local coordinate space (mask is
        # already cropped to the region), so they need shifting by
        # (x_min, y_min) — via drawContours' `offset` arg — whenever they're
        # drawn onto the full-size `annotated` image.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Light interior tint: filled contours drawn into a small
        # region-sized buffer, then alpha-blended back only over that
        # region — cheaper than blending the whole image, and the low
        # alpha keeps tissue detail visible underneath.
        region = annotated[y_min:y_max, x_min:x_max]
        fill_layer = region.copy()
        cv2.drawContours(fill_layer, contours, -1, _FILL_COLOR, thickness=cv2.FILLED)
        cv2.addWeighted(fill_layer, alpha, region, 1 - alpha, 0, dst=region)
        annotated[y_min:y_max, x_min:x_max] = region

        # Solid boundary line, drawn at full opacity on top of the tint so
        # the contour itself stays crisp instead of also being alpha-blended.
        cv2.drawContours(
            annotated, contours, -1, _CONTOUR_COLOR, thickness=_CONTOUR_THICKNESS, offset=(x_min, y_min)
        )

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
