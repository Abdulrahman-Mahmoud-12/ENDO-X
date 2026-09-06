"""WebSocket endpoint for browser camera inference."""

from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.pipeline.video_pipeline import VideoPipeline
from app.services.video_service import PassthroughTracker

router = APIRouter()


def _decode_frame(payload: bytes) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Could not decode camera frame")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _encode_frame(frame_rgb: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Could not encode annotated camera frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@router.websocket("/predict/live")
async def predict_live(websocket: WebSocket) -> None:
    """Receive JPEG frames and return annotated frames plus inference metrics."""
    await websocket.accept()
    app: Any = websocket.app
    detector = getattr(app.state, "detector", None)
    segmenter = getattr(app.state, "segmenter", None)
    tracker = getattr(app.state, "tracker", None) or PassthroughTracker()

    if detector is None or segmenter is None:
        await websocket.close(code=1013, reason="AI models are not loaded")
        return

    pipeline = VideoPipeline(detector=detector, segmenter=segmenter, tracker=tracker)
    pipeline.reset()
    frame_index = 0
    last_timestamp = time.perf_counter()

    try:
        while True:
            payload = await websocket.receive_bytes()
            started = time.perf_counter()
            frame = _decode_frame(payload)
            result = pipeline.run_frame(frame_index=frame_index, frame=frame, sample_rate=1)
            elapsed_ms = (time.perf_counter() - started) * 1000
            now = time.perf_counter()
            fps = 1 / max(now - last_timestamp, 1e-6)
            last_timestamp = now

            await websocket.send_json(
                {
                    "status": "success",
                    "frame": _encode_frame(result.annotated_frame if result.annotated_frame is not None else frame),
                    "frame_index": frame_index,
                    "fps": round(fps, 2),
                    "latency_ms": round(elapsed_ms, 2),
                    "detections": [
                        {
                            "class": track.class_name,
                            "confidence": track.confidence,
                            "bbox": [
                                track.bbox.x_min,
                                track.bbox.y_min,
                                track.bbox.x_max,
                                track.bbox.y_max,
                            ],
                            "track_id": track.track_id,
                            "frame_count": track.frame_count,
                        }
                        for track in result.tracks
                    ],
                }
            )
            frame_index += 1
    except WebSocketDisconnect:
        return
    except (ValueError, RuntimeError) as exc:
        await websocket.send_json({"status": "error", "message": str(exc)})
        await websocket.close(code=1011)