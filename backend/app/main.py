"""Creates the FastAPI application and registers all API routes.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import get_settings
from app.core.lifespan import lifespan


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def create_app() -> FastAPI:
    """Application factory: builds and configures the FastAPI instance."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Gastrointestinal endoscopy computer vision platform: polyp "
            "detection, segmentation, and tracking for image, video, and "
            "live-camera input."
        ),
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Serve annotated images/videos written to storage/outputs so the
    # frontend can load them directly by URL.
    settings.ensure_storage_dirs()
    app.mount(
        "/storage/outputs",
        StaticFiles(directory=str(settings.outputs_dir)),
        name="outputs",
    )

    return app


app = create_app()
