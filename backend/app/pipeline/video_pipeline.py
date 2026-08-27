"""Video-specific pipeline: per-frame detect -> crop -> segment (all via
BasePipeline, unchanged from image_pipeline.py's approach) -> track ->
annotate, with support for sub-sampling long videos.

LAYERING: only imports domain/interfaces types (Detector, Segmenter,
Tracker) — never PolypDetector/PolypSegmenter/PolypTracker directly.
Concrete implementations are injected by services/video_service.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.domain.interfaces.detector import Detector
from app.domain.interfaces.segmenter import Segmenter
from app.domain.interfaces.tracker import Track, Tracker
from app.pipeline.base_pipeline import BasePipeline, PerObjectResult
from app.schemas.prediction import BoundingBox, SegmentationMask
from app.utils.annotation import draw_tracked_objects, overlay_mask
from app.utils.video import timed

# Below this IoU, don't guess which detection a track's mask came from —
# better to show an unfilled box than a mask stolen from a neighboring
# polyp.
_MASK_MATCH_IOU_FLOOR = 0.35


def _iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    xa1, ya1 = max(box_a.x_min, box_b.x_min), max(box_a.y_min, box_b.y_min)
    xa2, ya2 = min(box_a.x_max, box_b.x_max), min(box_a.y_max, box_b.y_max)
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    if inter <= 0:
        return 0.0
    union = box_a.width * box_a.height + box_b.width * box_b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class FramePipelineResult:
    frame_index: int
    tracks: list[Track] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None
    has_polyp: bool = False
    was_sampled: bool = True  # False if this frame reused the last sampled result
    inference_time: float = 0.0  # seconds; 0.0 for non-sampled frames


class VideoPipeline(BasePipeline):
    """detect -> crop -> segment -> track -> annotate, per frame.

    SAMPLE_RATE BEHAVIOR: `sample_rate` controls how many frames actually
    run the (expensive) detect/segment/track stages. On skipped frames,
    the tracker is NOT called — its Kalman-filter-based state already
    tolerates gaps between updates, so re-feeding it stale/duplicate
    detections would only add noise. Instead, the *last sampled frame's*
    tracks are redrawn onto the skipped frame as-is (same box position),
    so the output video doesn't flicker to empty between samples. This
    means boxes visibly "hold still" for `sample_rate - 1` frames between
    refreshes at high sample_rate values — a deliberate smoothness/cost
    trade-off; interpolating box positions between samples would look
    better but adds complexity I'd rather flag than silently build in.
    """

    def __init__(
        self,
        detector: Detector,
        segmenter: Segmenter,
        tracker: Tracker,
        settings=None,
    ) -> None:
        super().__init__(detector=detector, segmenter=segmenter, settings=settings)
        self.tracker = tracker
        self._last_tracks: list[Track] = []

    def reset(self) -> None:
        """Call once per new video: clears both the tracker's internal
        state and this pipeline's cached last-tracks-for-skipped-frames."""
        self._last_tracks = []
        self.tracker.reset()

    def _match_mask(self, track: Track, per_object: list[PerObjectResult]) -> SegmentationMask | None:
        best_iou = _MASK_MATCH_IOU_FLOOR
        best_mask: SegmentationMask | None = None
        for obj in per_object:
            if obj.segmentation is None:
                continue
            score = _iou(track.bbox, obj.detection.bbox)
            if score > best_iou:
                best_iou, best_mask = score, obj.segmentation
        return best_mask

    def run_frame(self, frame_index: int, frame: np.ndarray, sample_rate: int) -> FramePipelineResult:
        is_sampled = (frame_index % max(1, sample_rate)) == 0

        if not is_sampled:
            annotated = draw_tracked_objects(frame, self._last_tracks)
            return FramePipelineResult(
                frame_index=frame_index,
                tracks=self._last_tracks,
                annotated_frame=annotated,
                has_polyp=len(self._last_tracks) > 0,
                was_sampled=False,
            )

        with timed() as t:
            detect_segment_result = self.run_detect_segment(frame)
            tracks = self.tracker.update(detect_segment_result.detections, frame)

        annotated = draw_tracked_objects(frame, tracks)
        for track in tracks:
            mask = self._match_mask(track, detect_segment_result.per_object)
            if mask is not None:
                annotated = overlay_mask(annotated, mask, track.bbox)

        self._last_tracks = tracks
        return FramePipelineResult(
            frame_index=frame_index,
            tracks=tracks,
            annotated_frame=annotated,
            has_polyp=len(tracks) > 0,
            was_sampled=True,
            inference_time=t.elapsed,
        )
