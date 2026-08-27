"""YOLO-based polyp detector wrapper.

Framework-light: no FastAPI/HTTP imports. Loads a native ultralytics ``.pt``
checkpoint (``YOLO(path)``) — NOT a raw ``state_dict`` — and exposes a
``predict()`` that returns our own ``Detection`` schema, not ultralytics'
internal Results object, so nothing above this layer depends on ultralytics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.core.config import Settings, get_settings
from app.domain.interfaces.detector import Detector
from app.schemas.prediction import BoundingBox, Detection

logger = logging.getLogger(__name__)


class PolypDetector(Detector):
    """Wraps a trained Ultralytics YOLO polyp detector.

    Confidence / IoU thresholds are read from ``Settings`` at predict time
    (not hardcoded, not baked in at construction) so a config/.env change
    takes effect without re-instantiating the wrapper.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        settings: Settings | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._settings = settings or get_settings()
        self._model = None

    def load(self) -> "PolypDetector":
        """Load the YOLO weights into memory. Idempotent — safe to call once at startup."""
        if self._model is not None:
            return self

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed; add it to requirements.txt to use PolypDetector"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"Detector weights not found at {self.model_path}")

        logger.info("Loading YOLO detector from %s on device=%s", self.model_path, self.device)
        model = YOLO(str(self.model_path))
        model.to(self.device)
        self._model = model
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single RGB image (H, W, 3) uint8 array.

        ``image`` should be RGB (ultralytics accepts either, but the rest of
        this codebase standardizes on RGB via ``decode_upload_bytes`` /
        ``resize_and_pad``). NOTE: imgsz is left to ultralytics' own default
        resolution behaviour (it reads the training imgsz embedded in the
        checkpoint) — confirm this matches your training run; override with
        an explicit ``imgsz=`` kwarg here if not.
        """
        if self._model is None:
            raise RuntimeError("PolypDetector.load() must be called before predict()")

        settings = self._settings
        results = self._model.predict(
            source=image,
            conf=settings.detection_confidence_threshold,
            iou=settings.detection_iou_threshold,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names if hasattr(result, "names") else {}
        detections: list[Detection] = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0].item())
            cls_id = int(box.cls[0].item()) if box.cls is not None else 0
            class_name = names.get(cls_id, "polyp") if isinstance(names, dict) else "polyp"

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3]),
                )
            )

        return detections