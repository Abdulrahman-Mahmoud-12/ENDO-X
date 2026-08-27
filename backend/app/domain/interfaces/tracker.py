"""Framework-light contract for anything that tracks polyps across frames.

Deliberately has zero FastAPI/HTTP imports, same rationale as detector.py
and segmenter.py: exercisable from a plain script or pytest.

DESIGN NOTE — reusing TrackedObject as Track
Your brief asked for a `Track` type with at least object_id, bbox,
confidence, age/frame_count, and said to match schemas/prediction.py's
existing `TrackedObject` if compatible. It is compatible: TrackedObject
already has track_id, class_name, confidence, bbox, and frame_count. So
rather than inventing a parallel `Track` dataclass that just duplicates
those fields under different names (object_id vs track_id), `Track` is a
plain alias for `TrackedObject`. No schema extension needed.

DESIGN NOTE — update() signature includes `image`
Your brief's sketch was `update(self, detections) -> list[Track]`. I
extended it to `update(self, detections, image)`. Reason: the ByteTrack
implementation this pairs with (see models/tracker.py) is a thin wrapper
around the `boxmot` library, whose trackers share one uniform
`update(dets, img)` API across motion-only trackers (ByteTrack) and
appearance-based ones (BoT-SORT, StrongSORT, etc.) so they're
interchangeable without changing call sites. ByteTrack itself ignores the
pixel content, but the frame reference has to be threaded through the
Protocol for that interchangeability to hold — and for it to be available
later if you ever swap in an appearance-based tracker. If you'd rather
keep the interface minimal and drop this, PolypTracker.update() is the
only place that would need to change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.schemas.prediction import Detection, TrackedObject

Track = TrackedObject


@runtime_checkable
class Tracker(Protocol):
    """Contract: given this frame's detections, return persistent tracks."""

    def update(self, detections: list[Detection], image: np.ndarray) -> list[Track]:
        """Advance tracker state by one frame and return current tracks.

        ``detections`` are this frame's raw detector output (unordered,
        no identity). ``image`` is the RGB (H, W, 3) uint8 frame they came
        from. Returns the full set of currently-active tracks (not just
        ones matched this frame) with stable ``track_id``s across calls.
        Implementations are expected to already be loaded/warm.
        """
        ...

    def reset(self) -> None:
        """Clear all track state.

        Must be called between unrelated videos processed by the same
        long-lived tracker instance — otherwise track IDs and Kalman state
        from video A would leak into video B's first frames.
        """
        ...
