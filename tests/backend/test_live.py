"""Contract test for the live WebSocket endpoint using fake AI components."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.live import router as live_router
from app.schemas.prediction import BoundingBox, Detection, SegmentationMask, TrackedObject


class FakeDetector:
    def predict(self, image: np.ndarray) -> list[Detection]:
        return [Detection(confidence=0.9, bbox=BoundingBox(x_min=10, y_min=10, x_max=40, y_max=40))]


class FakeSegmenter:
    def predict(self, image_crop: np.ndarray, detection_index: int = 0) -> SegmentationMask:
        mask = np.full(image_crop.shape[:2], 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", mask)
        assert ok
        import base64

        return SegmentationMask(
            detection_index=detection_index,
            mask_area_pixels=int(mask.size),
            mask_encoding=base64.b64encode(encoded.tobytes()).decode("ascii"),
        )


class FakeTracker:
    def reset(self) -> None:
        pass

    def update(self, detections: list[Detection], image: np.ndarray) -> list[TrackedObject]:
        return [
            TrackedObject(track_id=1, confidence=d.confidence, bbox=d.bbox, frame_count=1)
            for d in detections
        ]


def test_live_websocket_returns_annotated_frame() -> None:
    app = FastAPI()
    app.include_router(live_router)
    app.state.detector = FakeDetector()
    app.state.segmenter = FakeSegmenter()
    app.state.tracker = FakeTracker()

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    with TestClient(app) as client:
        with client.websocket_connect("/predict/live") as websocket:
            websocket.send_bytes(encoded.tobytes())
            response = websocket.receive_json()

    assert response["status"] == "success"
    assert response["detections"][0]["track_id"] == 1
    assert response["frame"]