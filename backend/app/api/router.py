"""Aggregates all versioned API routers into one ``api_router`` that
main.py mounts once.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import api_router_v1

api_router = APIRouter()
api_router.include_router(api_router_v1, prefix="/api/v1")
