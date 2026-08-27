"""Standalone model verification script for ENDO-X pipeline.

Runs dummy or real image tensors through detector and segmenter model weights
to verify model integrity, execution device, and output shapes.
"""

from pathlib import Path
import numpy as np
import torch
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DETECTOR_PATH = BASE_DIR / "models/detector/best.pt"
SEGMENTER_PATH = BASE_DIR / "models/segmenter/best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_detector(image_np: np.ndarray):
    print(f"\n--- Testing Detector on {DEVICE.upper()} ---")
    if not DETECTOR_PATH.exists():
        print(f"❌ Error: Detector weights not found at {DETECTOR_PATH.resolve()}")
        return None

    try:
        from ultralytics import YOLO

        model = YOLO(str(DETECTOR_PATH))
        model.to(DEVICE)
        print("✓ Detector weights loaded successfully.")

        # Run inference
        results = model.predict(image_np, verbose=False, device=DEVICE)
        boxes = results[0].boxes

        print(f"✓ Inference successful. Detected {len(boxes)} box(es).")
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            print(f"   [Box {idx+1}] Class: {cls_id} | Conf: {conf:.4f} | Coordinates: {xyxy.round(1)}")

        return boxes
    except Exception as e:
        print(f"❌ Detector failed with error: {e}")
        return None


def test_segmenter(image_crop_np: np.ndarray):
    print(f"\n--- Testing Segmenter on {DEVICE.upper()} ---")
    if not SEGMENTER_PATH.exists():
        print(f"❌ Error: Segmenter weights not found at {SEGMENTER_PATH.resolve()}")
        return None

    try:
        import segmentation_models_pytorch as smp

        # Reconstruct base architecture matching your config
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        state_dict = torch.load(SEGMENTER_PATH, map_location=DEVICE)
        # Handle state_dict loading if saved via Lightning/PyTorch wrapper
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

        model.to(DEVICE)
        model.eval()
        print("✓ Segmenter weights loaded successfully.")

        # Preprocess crop (Resize -> Transpose to CHW -> Float32 [0,1] Batch)
        resized = cv2.resize(image_crop_np, (256, 256))
        normalized = resized.astype(np.float32) / 255.0
        tensor_input = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor_input)
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (probs > 0.5).astype(np.uint8)

        print(f"✓ Inference successful.")
        print(f"   Output Mask Shape: {binary_mask.shape}")
        print(f"   Positive Mask Area: {np.sum(binary_mask)} pixels")
        return binary_mask
    except Exception as e:
        print(f"❌ Segmenter failed with error: {e}")
        return None


if __name__ == "__main__":
    print(f"Running Environment Check...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Execution Device: {DEVICE}")

    # Use a dummy test image if sample file isn't present
    sample_path = Path("tests/backend/fixtures/polyp_sample1.jpg")
    if sample_path.exists():
        print(f"Loading test frame from {sample_path}...")
        test_img = cv2.imread(str(sample_path))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    else:
        print("Sample image not found; generating random RGB tensor (640x640x3)...")
        test_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # 1. Run Detector
    boxes = test_detector(test_img)

    # 2. Run Segmenter on Crop (or dummy crop if 0 boxes detected)
    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].cpu().numpy())
        crop = test_img[max(0, y1):min(test_img.shape[0], y2), max(0, x1):min(test_img.shape[1], x2)]
    else:
        print("\nNo bounding box found for cropping; using full image for segmenter test...")
        crop = test_img

    if crop.size > 0:
        test_segmenter(crop)

    print("\n--- Model Verification Finished ---")