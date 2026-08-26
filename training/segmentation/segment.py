import cv2
import numpy as np
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 256
THRESHOLD = 0.5

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PolypSegmenter:
    def __init__(self, weights_path, device="cpu"):
        self.device = torch.device(device)

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(self.device)

        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(
                IMAGE_SIZE,
                IMAGE_SIZE,
                interpolation=cv2.INTER_LINEAR,
            ),
            A.Normalize(
                mean=MEAN,
                std=STD,
            ),
            ToTensorV2(),
        ])

    @torch.no_grad()
    def predict(self, image):
        original_height, original_width = image.shape[:2]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=rgb_image)
        tensor = transformed["image"].float().unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask = (probability > THRESHOLD).astype(np.uint8)

        mask = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask