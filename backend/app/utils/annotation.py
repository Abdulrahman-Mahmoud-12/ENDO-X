"""Drawing helpers: turn a raw frame + tracked objects (+ optional masks)
into an annotated frame.

NOT one of the 7 files you listed — added because video_pipeline.py's
"annotate" step needs to live somewhere, and you mentioned Phase 2's
image endpoint already draws annotated_image_url output. I don't have
that drawing code (image.py / whatever util it uses wasn't shared), so
this is a fresh implementation rather than a reuse of it. If Phase 2
already has equivalent box/mask-drawing logic, it's worth consolidating
into one shared util used by both the image and video pipelines instead
of keeping two copies that can drift apart.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from app.schemas.prediction import BoundingBox, SegmentationMask, TrackedObject

_BOX_COLOR = (0, 255, 0)  # RGB, drawn as-is since frames are RGB throughout
_TEXT_COLOR = (0, 0, 0)
_MASK_COLOR = (0, 200, 255)  # RGB


def draw_tracked_objects(frame_rgb: np.ndarray, tracks: list[TrackedObject]) -> np.ndarray:
    """Draw a box + '#id class conf' label for each track. Returns a new
    array; does not mutate `frame_rgb`."""
    annotated = frame_rgb.copy()
    for track in tracks:
        x1, y1, x2, y2 = (
            int(track.bbox.x_min),
            int(track.bbox.y_min),
            int(track.bbox.x_max),
            int(track.bbox.y_max),
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(th + baseline, y1)
        cv2.rectangle(annotated, (x1, label_y - th - baseline), (x1 + tw, label_y), _BOX_COLOR, -1)
        cv2.putText(
            annotated, label, (x1, label_y - baseline // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return annotated


def overlay_mask(
    frame_rgb: np.ndarray,
    mask: SegmentationMask,
    bbox: BoundingBox,
    alpha: float = 0.4,
) -> np.ndarray:
    """Alpha-blend a translucent color over the mask region, placed back
    at `bbox`'s location in the full frame.

    `mask.mask_encoding` is a base64 PNG "cropped to the detection bbox"
    (per SegmentationMask's docstring) — the bbox passed here should be
    the *track's* current bbox, not necessarily the original detection's,
    since IoU-matching (see video_pipeline.py) associates a track back to
    whichever detection produced its mask. The mask is resized to that
    bbox's current size, so mild drift between the two boxes is smoothed
    over rather than causing a crash.
    """
    png_bytes = base64.b64decode(mask.mask_encoding)
    mask_img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, int(bbox.x_min)), max(0, int(bbox.y_min))
    x2, y2 = min(w, int(bbox.x_max)), min(h, int(bbox.y_max))
    if x2 <= x1 or y2 <= y1:
        return frame_rgb

    mask_resized = cv2.resize(mask_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_resized > 0
    if not mask_bool.any():
        return frame_rgb

    region = frame_rgb[y1:y2, x1:x2]
    color_layer = np.zeros_like(region)
    color_layer[:] = _MASK_COLOR
    blended = cv2.addWeighted(region, 1 - alpha, color_layer, alpha, 0)

    annotated = frame_rgb.copy()
    region_out = annotated[y1:y2, x1:x2]
    region_out[mask_bool] = blended[mask_bool]
    annotated[y1:y2, x1:x2] = region_out
    return annotated
