"""Centralized API error shape and exception handling.

Every error response returned by the API — regardless of which endpoint
raised it — has the exact same JSON shape:

    {"status": "error", "error_code": "...", "message": "..."}

Endpoints and services should raise an ``APIError`` subclass; nothing above
the service layer should construct a raw ``HTTPException`` with an ad-hoc
body.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base class for all handled API errors.

    Subclasses set ``status_code`` and ``error_code`` as class attributes;
    ``message`` is supplied per-instance.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={"status": "error", "error_code": self.error_code, "message": self.message},
        )


class InvalidFileTypeError(APIError):
    """Uploaded file's extension isn't in Settings.allowed_image_extensions."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_file_type"


class FileTooLargeError(APIError):
    """Uploaded file exceeds Settings.max_image_size_mb."""

    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    error_code = "file_too_large"


class DecodeError(APIError):
    """File passed extension/size checks but isn't a valid/decodable image."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "decode_error"


class ModelNotLoadedError(APIError):
    """Detector and/or segmenter weren't loaded by lifespan.py at startup."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "model_not_loaded"


class InferenceError(APIError):
    """The pipeline raised while actually running detect/crop/segment."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "inference_error"


async def _api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    logger.warning(
        "APIError on %s %s: [%s] %s", request.method, request.url.path, exc.error_code, exc.message
    )
    return exc.to_response()


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_code": "internal_error",
            "message": "An unexpected error occurred.",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every ``APIError`` subclass — and anything unhandled — to the
    shared JSON error shape. Call once from main.py's ``create_app()``.
    """
    app.add_exception_handler(APIError, _api_error_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
