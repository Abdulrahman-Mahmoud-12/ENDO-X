"""ByteTrack-based polyp tracker wrapper.

Framework-light: no FastAPI/HTTP imports, matching detector.py/segmenter.py.

WHY BYTETRACK OVER BOT-SORT
Both are supported by `boxmot` with the same update(dets, img) API, so
switching later is a one-line change here if this turns out wrong. Went
with ByteTrack because:

  1. Single class, no appearance disambiguation needed. BoT-SORT's main
     advantage over ByteTrack is a ReID embedding for re-identifying
     objects after long occlusions or when many similar-looking objects
     are present. Here every object is "polyp" and — worse — polyps
     often *do* look visually similar to each other, so a ReID embedding
     is more likely to mismatch two different polyps as "the same track"
     than to help. Motion + IoU association (ByteTrack) is the safer
     default for this domain.
  2. ByteTrack's headline trick — recovering low-confidence detection
     boxes that a naive tracker would discard — matters a lot here.
     Polyps get partially occluded by the scope, folds, bubbles, or
     specular highlights; a detection that drops from 0.6 to 0.15
     confidence for a couple of frames due to a bubble passing over it
     is exactly the case ByteTrack was designed to keep tracking through
     instead of dropping and reassigning a new ID.
  3. Lower latency / no extra ReID network to run per detection —
     relevant since this already stacks on top of YOLO detection + U-Net
     segmentation per frame, and config.py's `tracker_type` default is
     already "bytetrack", so this doesn't require a config change.

If you're seeing frequent ID switches in practice (e.g. camera whip-pans
common in colonoscopy withdrawal), BoT-SORT with camera motion
compensation (`boxmot.BotSort(..., with_reid=False)`) is the first thing
to try before adding full ReID.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.config import Settings, get_settings
from app.domain.interfaces.tracker import Track, Tracker
from app.schemas.prediction import BoundingBox, Detection, TrackedObject

logger = logging.getLogger(__name__)


class PolypTracker(Tracker):
    """Wraps `boxmot.ByteTrack` for single-class polyp tracking.

    `frame_count` on the returned TrackedObject is computed here, not by
    boxmot (whose `hits`/`age` fields vary by version and don't map
    1:1 onto "consecutive frames tracked" for a re-acquired track) — kept
    as a simple per-track_id counter that increments every time that ID
    appears in `update()`'s output.
    """

    def __init__(self, frame_rate: int = 30, settings: Settings | None = None) -> None:
        self.frame_rate = frame_rate
        self._settings = settings or get_settings()
        self._tracker = None
        self._frame_counts: dict[int, int] = {}

    def load(self) -> "PolypTracker":
        """Initialize the ByteTrack instance. Idempotent — safe to call once at startup."""
        if self._tracker is not None:
            return self

        try:
            from boxmot import ByteTrack
        except ImportError as exc:
            raise RuntimeError(
                "boxmot is not installed; add it to requirements.txt to use "
                "PolypTracker (pip install boxmot)"
            ) from exc

        logger.info("Initializing ByteTrack (frame_rate=%s)", self.frame_rate)
        self._tracker = ByteTrack(frame_rate=self.frame_rate)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._tracker is not None

    def reset(self) -> None:
        """Drop all track state — call between unrelated videos.

        Re-instantiates rather than relying on a private reset() method,
        since boxmot's internal reset API has changed across versions;
        this is more version-stable at the cost of a small re-init cost.
        """
        self._frame_counts.clear()
        if self._tracker is not None:
            from boxmot import ByteTrack

            self._tracker = ByteTrack(frame_rate=self.frame_rate)

    def update(self, detections: list[Detection], image: np.ndarray) -> list[Track]:
        """Advance the tracker by one frame.

        NOTE: boxmot's tracker.update() expects BGR uint8 frames (it may
        internally rely on OpenCV conventions for any visualization/CMC
        steps). This codebase standardizes on RGB per detector.py's
        contract, so we convert here rather than push that concern up
        into video_pipeline.py.
        """
        if self._tracker is None:
            raise RuntimeError("PolypTracker.load() must be called before update()")

        if not detections:
            dets_array = np.empty((0, 6), dtype=float)
        else:
            dets_array = np.array(
                [
                    [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max, d.confidence, 0.0]
                    for d in detections
                ],
                dtype=float,
            )

        image_bgr = image[:, :, ::-1]
        tracked_rows = self._tracker.update(dets_array, image_bgr)

        tracks: list[Track] = []
        for row in tracked_rows:
            # boxmot's output columns: x1, y1, x2, y2, track_id, conf, cls, [extra...]
            x1, y1, x2, y2, track_id, confidence = row[:6]
            track_id = int(track_id)
            self._frame_counts[track_id] = self._frame_counts.get(track_id, 0) + 1

            tracks.append(
                TrackedObject(
                    track_id=track_id,
                    class_name="polyp",
                    confidence=float(confidence),
                    bbox=BoundingBox(x_min=float(x1), y_min=float(y1), x_max=float(x2), y_max=float(y2)),
                    frame_count=self._frame_counts[track_id],
                )
            )

        return tracks
