"""Tests for GET /health.

Unlike the framework-light tests under ``tests/ai/``, this one legitimately
imports FastAPI — it's testing the HTTP layer itself, not the model wrappers.
Builds a minimal app around the real router and drives ``app.state`` directly
so it doesn't depend on the lifespan actually loading model checkpoints.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.health import router as health_router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(make_app())


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_healthy_when_both_models_loaded(client: TestClient) -> None:
    client.app.state.model_status = {"detector": "loaded", "segmenter": "loaded"}
    client.app.state.device = "cpu"

    body = client.get("/health").json()

    assert body["status"] == "healthy"
    assert body["detector"] == "loaded"
    assert body["segmenter"] == "loaded"
    assert body["device"] == "cpu"


@pytest.mark.parametrize(
    "model_status",
    [
        {"detector": "not_loaded", "segmenter": "loaded"},
        {"detector": "loaded", "segmenter": "not_loaded"},
        {"detector": "not_loaded", "segmenter": "not_loaded"},
    ],
)
def test_health_degraded_when_any_model_not_loaded(client: TestClient, model_status: dict[str, str]) -> None:
    client.app.state.model_status = model_status
    client.app.state.device = "cpu"

    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["detector"] == model_status["detector"]
    assert body["segmenter"] == model_status["segmenter"]


def test_health_defaults_when_model_status_never_set(client: TestClient) -> None:
    """If lifespan hasn't run (e.g. app built without it in a test), health
    should report not_loaded/degraded rather than raising."""
    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["detector"] == "not_loaded"
    assert body["segmenter"] == "not_loaded"
    assert body["device"] == "unknown"


def test_health_response_includes_app_metadata(client: TestClient) -> None:
    from app.core.config import get_settings

    settings = get_settings()
    body = client.get("/health").json()

    assert body["app_name"] == settings.app_name
    assert body["app_version"] == settings.app_version


def test_health_response_matches_schema_fields(client: TestClient) -> None:
    from app.schemas.prediction import HealthResponse

    body = client.get("/health").json()
    # Round-trips cleanly through the declared response_model — catches drift
    # between the route's HealthResponse(...) construction and the schema.
    HealthResponse.model_validate(body)