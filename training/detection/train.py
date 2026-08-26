import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ENDO-X polyp detection model.")

    parser.add_argument(
        "--data",
        type=str,
        default="training/configs/detection.yaml",
        help="Path to the dataset YAML file (defines train/val/test splits and class names). "
             "Run this script from the repository root so this default path resolves correctly.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="Base YOLO model to start training from (a pretrained checkpoint name or path).",
    )
    parser.add_argument("--epochs", type=int, default=70, help="Maximum number of training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Stop training early if validation mAP does not improve for this many epochs.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (pixels, square).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on: '0' for first GPU, 'cpu' for CPU-only training.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader worker processes.")
    parser.add_argument(
        "--project",
        type=str,
        default="training/experiments",
        help="Root output folder where training runs are saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="polyp_detection",
        help="Name of this specific training run (creates a subfolder under --project).",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs, in addition to the best/last weights.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at '{data_path}'. "
            f"Make sure dataset.yaml exists and points to your train/val/test image folders."
        )

    print("=" * 60)
    print("ENDO-X — Polyp Detection Training")
    print("=" * 60)
    print(f"Base model : {args.model}")
    print(f"Dataset    : {data_path}")
    print(f"Epochs     : {args.epochs} (patience={args.patience})")
    print(f"Image size : {args.imgsz}")
    print(f"Batch size : {args.batch}")
    print(f"Device     : {args.device}")
    print("=" * 60)

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mixup=0.0,
        degrees=15.0,
        fliplr=0.5,
        flipud=0.5,
        deterministic=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best weights saved to: {best_weights}")
    print("Next steps:")
    print("  1. Run validate.py to evaluate this model on the test set.")
    print(f"  2. Copy the best weights into models/detector/best.pt for production use:")
    print(f"     copy {best_weights} models\\detector\\best.pt   (Windows)")
    print(f"     cp {best_weights} models/detector/best.pt      (Mac/Linux)")
    print("=" * 60)


if __name__ == "__main__":
    main()