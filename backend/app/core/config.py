"""Application settings, loaded from environment variables / .env file.

All configuration used anywhere in the backend must be declared here rather
than read ad-hoc with ``os.environ`` elsewhere, so there is a single source
of truth for what the service depends on at startup.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    """Strongly typed application settings.

    Values are read from environment variables (or a ``.env`` file) using
    the exact field names below, upper-cased. See ``.env.example`` at the
    project root for the full list with documentation.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App metadata ---------------------------------------------------
    app_name: str = "ENDO-X"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    debug: bool = False

    # --- Server ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # --- CORS --------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- Compute device ----------------------------------------------------
    device: str = "auto"

    # --- Model weights -------------------------------------------------
    detector_model_path: Path = BASE_DIR / "models" / "detector" / "best.pt"
    segmenter_model_path: Path = BASE_DIR / "models" / "segmenter" / "best.pth"
    segmenter_encoder_name: str = "resnet34"
    segmenter_architecture: str = "unet"
    segmenter_image_size: int = 256

    # --- Inference thresholds ------------------------------------------
    detection_confidence_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    detection_iou_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    segmentation_mask_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    detection_roi_margin: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- Uploads / storage -----------------------------------------------
    storage_root: Path = Path("app/storage")
    uploads_dir: Path = Path("app/storage/uploads")
    outputs_dir: Path = Path("app/storage/outputs")
    max_image_size_mb: int = 15
    max_video_size_mb: int = 250
    allowed_image_extensions: str = ".jpg,.jpeg,.png"
    allowed_video_extensions: str = ".mp4,.avi,.mov"

    # --- Live inference --------------------------------------------------
    live_target_fps: int = 15

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in logging._nameToLevel:  # noqa: SLF001
            raise ValueError(f"Invalid LOG_LEVEL: {value!r}")
        return normalized

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a clean list, parsed from the comma-separated env var."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_image_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_image_extensions.split(",") if ext.strip()]

    @property
    def allowed_video_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_video_extensions.split(",") if ext.strip()]

    def ensure_storage_dirs(self) -> None:
        """Create the uploads/outputs directories if they do not exist yet."""
        for directory in (self.uploads_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance.

    Cached with ``lru_cache`` so the .env file is only parsed once, and every
    module importing this function shares the same Settings object.
    """
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings