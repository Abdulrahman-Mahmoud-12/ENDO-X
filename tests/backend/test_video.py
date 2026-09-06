"""End-to-end test for POST /api/v1/predict/video.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.prediction import BoundingBox, Detection, SegmentationMask, TrackedObject


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray) -> list[Detection]:
        x = 20 + self.calls * 2
        self.calls += 1
        return [
            Detection(
                class_name="polyp",
                confidence=0.9,
                bbox=BoundingBox(x_min=x, y_min=20, x_max=x + 40, y_max=60),
            )
        ]


class FakeSegmenter:
    def predict(self, image_crop: np.ndarray, detection_index: int = 0) -> SegmentationMask:
        h, w = max(1, image_crop.shape[0]), max(1, image_crop.shape[1])
        mask = np.full((h, w), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", mask)
        assert ok
        return SegmentationMask(
            detection_index=detection_index,
            mask_area_pixels=int(h * w),
            mask_encoding=base64.b64encode(encoded.tobytes()).decode("ascii"),
        )


class FakeTracker:
    def __init__(self) -> None:
        self._frame_count = 0

    def reset(self) -> None:
        self._frame_count = 0

    def update(self, detections: list[Detection], image: np.ndarray) -> list[TrackedObject]:
        if not detections:
            return []
        self._frame_count += 1
        det = detections[0]
        return [
            TrackedObject(
                track_id=1,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                frame_count=self._frame_count,
            )
        ]


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    video_path = Path("tests/data/polyp_sample1.mp4")
    if not video_path.exists():
        video_path = tmp_path / "sample.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 100))
        for _ in range(15):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
            writer.write(frame)
        writer.release()
    return video_path


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        app.state.detector = FakeDetector()
        app.state.segmenter = FakeSegmenter()
        app.state.tracker = FakeTracker()
        yield c


def test_predict_video_end_to_end(client: TestClient, sample_video: Path) -> None:
    with sample_video.open("rb") as f:
        response = client.post(
            "/api/v1/predict/video",
            files={"file": ("sample.mp4", f, "video/mp4")},
            params={"sample_rate": 1},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["output_video_url"].startswith("/storage/outputs/")

    summary = payload["summary"]
    assert summary["total_frames"] == 15
    assert summary["frames_with_polyp"] == 15
    assert summary["avg_fps"] > 0
    assert summary["avg_latency_ms"] >= 0

    output_filename = Path(payload["output_video_url"]).name
    output_path = Path("app/storage/outputs") / output_filename
    assert output_path.exists(), f"annotated output missing at {output_path}"
    assert output_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(output_path))
    assert cap.isOpened(), "annotated output isn't a playable/valid video container"
    ok, frame = cap.read()
    assert ok and frame is not None
    cap.release()


def test_predict_video_invalid_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict/video",
        files={"file": ("test.txt", b"invalid video data", "text/plain")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error_code"] == "unsupported_video_format"
