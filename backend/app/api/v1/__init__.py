"""Aggregates all v1 endpoint routers (health, image, video, live, ...)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.image import router as image_router
from app.api.v1.video import router as video_router

api_router_v1 = APIRouter()
api_router_v1.include_router(health_router, tags=["health"])
api_router_v1.include_router(image_router, tags=["image"])
api_router_v1.include_router(video_router, tags=["video"])
