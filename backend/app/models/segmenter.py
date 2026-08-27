"""U-Net polyp segmenter wrapper.

Framework-light: no FastAPI/HTTP imports.

Architecture MUST exactly match training (see ``unet_polyp_segmentation.ipynb``,
cell 18): ``smp.Unet(encoder_name="resnet34", encoder_weights=..., in_channels=3,
classes=1, activation=None)``. Checkpoints are saved as
``{"model_state_dict": ..., "epoch": ..., "best_metric": ..., "config": ...}``
(cell 23), not a bare ``state_dict()``, so loading unwraps that key.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.config import Settings, get_settings
from app.domain.interfaces.segmenter import Segmenter
from app.schemas.prediction import SegmentationMask
from app.services.preprocessing_service import preprocess_segmentation_input

logger = logging.getLogger(__name__)

_SUPPORTED_ARCHITECTURES = {"unet"}


class PolypSegmenter(Segmenter):
    """Wraps a trained U-Net (ResNet34 encoder) polyp segmenter."""

    def __init__(
        self,
        model_path: str | Path,
        encoder_name: str = "resnet34",
        architecture: str = "unet",
        device: str = "cpu",
        settings: Settings | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.encoder_name = encoder_name
        self.architecture = architecture
        self.device = device
        self._settings = settings or get_settings()
        self._model = None

    def load(self) -> "PolypSegmenter":
        """Load the U-Net weights into memory. Idempotent — safe to call once at startup."""
        if self._model is not None:
            return self

        if self.architecture not in _SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported segmenter architecture {self.architecture!r}; the trained "
                f"checkpoint is a plain U-Net. Check SEGMENTER_ARCHITECTURE in .env / config.py."
            )

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "segmentation-models-pytorch is not installed; "
                "add it to requirements.txt to use PolypSegmenter"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"Segmenter weights not found at {self.model_path}")

        logger.info(
            "Loading U-Net segmenter (encoder=%s) from %s on device=%s",
            self.encoder_name,
            self.model_path,
            self.device,
        )

        model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=None,  # weights come from our checkpoint, not fresh ImageNet
            in_channels=3,
            classes=1,
            activation=None,  # raw logits, matches training — sigmoid applied in predict()
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model = model
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @torch.no_grad()
    def predict(self, image_crop: np.ndarray, detection_index: int = 0) -> SegmentationMask:
        """Run segmentation on a single cropped polyp region.

        ``image_crop`` is an RGB uint8 array (H, W, 3) — typically the output
        of ``crop_by_bbox``. The mask is predicted at the training resolution
        then resized back up to the crop's original size before thresholding,
        so ``mask_area_pixels`` is in the crop's native pixel units.
        """
        if self._model is None:
            raise RuntimeError("PolypSegmenter.load() must be called before predict()")

        settings = self._settings
        orig_h, orig_w = image_crop.shape[:2]

        tensor = preprocess_segmentation_input(image_crop, settings=settings).to(self.device)
        logits = self._model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        probs_full_res = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (probs_full_res > settings.segmentation_mask_threshold).astype(np.uint8) * 255

        success, encoded = cv2.imencode(".png", binary_mask)
        if not success:
            raise RuntimeError("Failed to encode segmentation mask as PNG")
        mask_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        return SegmentationMask(
            detection_index=detection_index,
            mask_area_pixels=int((binary_mask > 0).sum()),
            mask_encoding=mask_b64,
        )