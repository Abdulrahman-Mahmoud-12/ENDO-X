"""Pydantic schemas shared by the image, video, and live prediction endpoints.

Defined once here in Phase 0 so every later phase (image/video/live services
and API routers) imports the same data contracts instead of re-declaring
overlapping shapes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates of the original frame."""

    x_min: float = Field(..., description="Left edge, in pixels")
    y_min: float = Field(..., description="Top edge, in pixels")
    x_max: float = Field(..., description="Right edge, in pixels")
    y_max: float = Field(..., description="Bottom edge, in pixels")

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


class Detection(BaseModel):
    """A single detected polyp from the YOLO detector."""

    class_name: str = Field(default="polyp")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class SegmentationMask(BaseModel):
    """Segmentation result for one detected polyp region."""

    detection_index: int = Field(..., description="Index into the parent detections list")
    mask_area_pixels: int = Field(..., ge=0)
    mask_encoding: str = Field(
        ...,
        description="Base64-encoded PNG of the binary mask, cropped to the detection bbox",
    )


class TrackedObject(BaseModel):
    """A detection with a persistent identity across video/live frames."""

    track_id: int
    class_name: str = Field(default="polyp")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    frame_count: int = Field(..., ge=1, description="Number of consecutive frames tracked")


class ImagePredictionResponse(BaseModel):
    """Response payload for POST /api/v1/predict/image."""

    detections: list[Detection] = Field(default_factory=list)
    segmentations: list[SegmentationMask] = Field(default_factory=list)
    num_polyps: int = Field(..., ge=0)
    inference_time: float = Field(..., ge=0.0, description="Seconds")
    annotated_image_url: str | None = Field(
        default=None, description="Relative URL to the annotated result image"
    )


class VideoPredictionResponse(BaseModel):
    """Response payload for POST /api/v1/predict/video."""

    tracked_objects: list[TrackedObject] = Field(default_factory=list)
    total_frames: int = Field(..., ge=0)
    average_fps: float = Field(..., ge=0.0)
    processing_time: float = Field(..., ge=0.0, description="Seconds")
    annotated_video_url: str | None = Field(
        default=None, description="Relative URL to the annotated result video"
    )


class HealthResponse(BaseModel):
    """Response payload for GET /api/v1/health."""

    status: str = Field(..., description="'healthy' or 'degraded'")
    app_name: str
    app_version: str
    device: str
    detector: str = Field(..., description="'loaded' or 'not_loaded'")
    segmenter: str = Field(..., description="'loaded' or 'not_loaded'")


class ErrorResponse(BaseModel):
    """Standard error payload returned by API exception handlers."""

    detail: str
