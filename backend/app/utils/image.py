"""Image IO and geometry helpers shared by the service layer and the pipeline.

Framework-light: no FastAPI imports (the service layer hands this raw
bytes/arrays, not ``UploadFile`` objects), so it can be unit tested without
a running app.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas.prediction import BoundingBox


def decode_upload_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (as read from an UploadFile) into an RGB array.

    Raises ``ValueError`` if the bytes don't decode to a valid image — the
    service layer is expected to translate that into the shared API error
    shape (``core/exceptions.DecodeError``).
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes; unsupported or corrupt file")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_and_pad(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize ``image`` to fit inside a ``target_size`` square, letterboxing the rest.

    Returns ``(padded_image, scale, (pad_x, pad_y))`` so callers can map
    coordinates predicted on the padded image back to the original. Kept
    here as the one shared implementation for any caller that needs a
    fixed-size square input — currently unused by the detector itself
    (ultralytics does its own letterboxing internally at predict time).
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return padded, scale, (pad_x, pad_y)


def compute_crop_region(
    image_shape: tuple[int, ...], bbox: BoundingBox, margin: float = 0.0
) -> tuple[int, int, int, int]:
    """Compute the (clamped) pixel rectangle for ``bbox`` expanded by ``margin``.

    ``margin`` is a fraction of the box's own width/height (matches
    ``Settings.detection_roi_margin``). Split out from ``crop_by_bbox`` so
    callers that need to know *where* a crop came from without slicing the
    array — e.g. placing a segmentation mask back onto the full image in
    ``utils/visualization.py`` — can reuse the exact same math instead of
    duplicating it.
    """
    h, w = image_shape[0], image_shape[1]
    box_w, box_h = bbox.width, bbox.height

    x_min = max(0, int(round(bbox.x_min - box_w * margin)))
    y_min = max(0, int(round(bbox.y_min - box_h * margin)))
    x_max = min(w, int(round(bbox.x_max + box_w * margin)))
    y_max = min(h, int(round(bbox.y_max + box_h * margin)))
    return x_min, y_min, x_max, y_max


def crop_by_bbox(image: np.ndarray, bbox: BoundingBox, margin: float = 0.0) -> np.ndarray:
    """Crop ``image`` to ``bbox``, optionally expanded by ``margin`` (fraction of box size).

    Raises ``ValueError`` for a degenerate region (e.g. a bbox entirely
    outside the image) — callers in ``pipeline/`` are expected to catch
    this per-detection rather than fail the whole request.
    """
    x_min, y_min, x_max, y_max = compute_crop_region(image.shape, bbox, margin)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Degenerate crop region for bbox {bbox!r} with margin={margin}")
    return image[y_min:y_max, x_min:x_max]
