"""End-to-end test for POST /api/v1/predict/image.

Skips (rather than fails) if real trained weights aren't present in this
environment — this exercises your actual detector/segmenter checkpoints
(see Settings.detector_model_path / segmenter_model_path), not mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "polyp_sample1.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _models_available() -> bool:
    settings = get_settings()
    return settings.detector_model_path.exists() and settings.segmenter_model_path.exists() and FIXTURE_IMAGE.exists()


@pytest.mark.skipif(not _models_available(), reason="Trained detector/segmenter weights not present")
def test_predict_image_end_to_end(client: TestClient) -> None:
    assert FIXTURE_IMAGE.exists(), f"Missing test fixture at {FIXTURE_IMAGE}"

    with open(FIXTURE_IMAGE, "rb") as f:
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("polyp_sample.png", f, "image/png")},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "detections" in body and "segmentations" in body

    # Known-positive frame (visible polyp) — expect at least one detection.
    assert len(body["detections"]) >= 1, "Expected at least one detection on a known-positive frame"
    for detection in body["detections"]:
        assert detection["class"] == "polyp"
        assert 0.0 <= detection["confidence"] <= 1.0
        assert len(detection["bbox"]) == 4

    assert len(body["segmentations"]) >= 1, "Expected at least one segmentation mask"
    for seg in body["segmentations"]:
        assert seg["mask_area_px"] >= 0
        assert isinstance(seg["polygon"], list)

    assert body["overlay_image_url"] is not None
    overlay_response = client.get(body["overlay_image_url"])
    assert overlay_response.status_code == 200

    assert body["inference_time_ms"] > 0


def test_predict_image_rejects_bad_file_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict/image",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error_code"] == "invalid_file_type"


def test_predict_image_404s_on_missing_file_field(client: TestClient) -> None:
    response = client.post("/api/v1/predict/image")
    assert response.status_code == 422
