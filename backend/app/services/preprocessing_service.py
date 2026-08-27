"""Preprocessing shared by model wrappers.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from app.core.config import Settings

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_segmentation_input(image_crop: np.ndarray, settings: Settings) -> torch.Tensor:
    """Resize + normalize a cropped RGB uint8 array into a (1, 3, H, W) float tensor.

    ``settings.segmenter_image_size`` is read from Settings (not
    hardcoded) so it matches whatever training resolution you configured.
    """
    size = settings.segmenter_image_size
    resized = cv2.resize(image_crop, (size, size), interpolation=cv2.INTER_LINEAR)

    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - _IMAGENET_MEAN) / _IMAGENET_STD

    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).float()