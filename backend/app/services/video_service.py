"""Validates uploaded video files, executes VideoPipeline frame-by-frame,
re-encodes annotated video output under storage/outputs, and returns
VideoPredictionAPIResponse.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, Request, UploadFile

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    CorruptVideoError,
    FileTooLargeError,
    ModelNotLoadedError,
    UnsupportedVideoFormatError,
)
from app.domain.interfaces.tracker import Track, Tracker
from app.pipeline.video_pipeline import VideoPipeline
from app.schemas.prediction import Detection, TrackedObject, VideoPredictionAPIResponse, VideoSummary
from app.utils.video import FPSAccountant, FrameReader, FrameWriter, validate_video_extension

logger = logging.getLogger(__name__)


class PassthroughTracker:
    """Per-frame pass-through tracker used when no tracking algorithm is loaded.

    Converts raw detections into single-frame TrackedObject instances.
    """

    def update(self, detections: list[Detection], image: np.ndarray) -> list[Track]:
        return [
            TrackedObject(
                track_id=i,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                frame_count=1,
            )
            for i, d in enumerate(detections)
        ]

    def reset(self) -> None:
        pass


def get_video_pipeline(request: Request) -> VideoPipeline:
    """FastAPI dependency: build a ``VideoPipeline`` from ``app.state``.

    Mirrors ``get_image_pipeline`` in ``inference_service.py`` — accesses
    ``detector`` and ``segmenter`` loaded by ``lifespan.py``.
    """
    detector = getattr(request.app.state, "detector", None)
    segmenter = getattr(request.app.state, "segmenter", None)
    tracker = getattr(request.app.state, "tracker", None)

    if detector is None or segmenter is None:
        missing = [name for name, obj in (("detector", detector), ("segmenter", segmenter)) if obj is None]
        raise ModelNotLoadedError(
            f"Model(s) not loaded: {', '.join(missing)}. Check GET /api/v1/health."
        )

    if tracker is None:
        tracker = PassthroughTracker()

    return VideoPipeline(detector=detector, segmenter=segmenter, tracker=tracker)


class VideoService:
    def __init__(self, pipeline: VideoPipeline, settings: Settings) -> None:
        self.pipeline = pipeline
        self.settings = settings

    async def _read_and_validate(self, file: UploadFile) -> Path:
        """Enforce file extension and size limits, saving to a temp file."""
        filename = file.filename or "video.mp4"
        ext = validate_video_extension(filename, self.settings)

        self.settings.ensure_storage_dirs()

        temp_dir = Path(tempfile.gettempdir()) / "endo_x_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / f"upload_{uuid.uuid4().hex}{ext}"

        total_bytes = 0
        max_bytes = self.settings.max_video_size_mb * 1024 * 1024

        try:
            with open(temp_file_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise FileTooLargeError(
                            f"Video exceeds the {self.settings.max_video_size_mb}MB limit"
                        )
                    buffer.write(chunk)
        except Exception:
            if temp_file_path.exists():
                temp_file_path.unlink()
            raise

        return temp_file_path

    async def predict_video(
        self,
        file: UploadFile,
        sample_rate: int = 1,
    ) -> VideoPredictionAPIResponse:
        input_path = await self._read_and_validate(file)

        ext = input_path.suffix.lower()
        output_filename = f"{uuid.uuid4().hex}{ext}"
        output_path = Path(self.settings.outputs_dir) / output_filename

        try:
            total_frames = 0
            frames_with_polyp = 0
            fps_accountant = FPSAccountant()

            self.pipeline.reset()

            try:
                reader = FrameReader(input_path)
                with reader:
                    fps = reader.fps
                    width = reader.width
                    height = reader.height

                    if width <= 0 or height <= 0:
                        raise CorruptVideoError("Video has invalid dimensions or zero frames.")

                    writer = FrameWriter(output_path, fps=fps, frame_size=(width, height))
                    with writer:
                        for frame_idx, frame in enumerate(reader.frames()):
                            result = self.pipeline.run_frame(
                                frame_index=frame_idx,
                                frame=frame,
                                sample_rate=sample_rate,
                            )
                            if result.annotated_frame is not None:
                                writer.write(result.annotated_frame)

                            total_frames += 1
                            if result.has_polyp:
                                frames_with_polyp += 1
                            if result.was_sampled:
                                fps_accountant.record(result.inference_time)

            except UnsupportedVideoFormatError:
                raise
            except Exception as exc:
                if isinstance(exc, (FileTooLargeError, UnsupportedVideoFormatError, CorruptVideoError)):
                    raise
                logger.exception("Error processing video: %s", exc)
                raise CorruptVideoError(f"Failed to process video: {exc}") from exc

            return VideoPredictionAPIResponse(
                status="success",
                output_video_url=f"/storage/outputs/{output_filename}",
                summary=VideoSummary(
                    total_frames=total_frames,
                    frames_with_polyp=frames_with_polyp,
                    avg_fps=fps_accountant.avg_fps,
                    avg_latency_ms=fps_accountant.avg_latency_ms,
                ),
            )

        finally:
            if input_path.exists():
                try:
                    input_path.unlink()
                except OSError as err:
                    logger.warning("Failed to delete temp input video %s: %s", input_path, err)


def get_video_service(
    pipeline: VideoPipeline = Depends(get_video_pipeline),
    settings: Settings = Depends(get_settings),
) -> VideoService:
    return VideoService(pipeline=pipeline, settings=settings)
