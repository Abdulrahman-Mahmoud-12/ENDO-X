import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the ENDO-X polyp detection model.")

    parser.add_argument(
        "--weights",
        type=str,
        default="models/detector/best.pt",
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="training/configs/detection.yaml",
        help="Path to the dataset YAML file (defines train/val/test splits and class names). "
             "Run this script from the repository root so this default path resolves correctly.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which dataset split to evaluate on. Use 'test' for the final, unbiased report.",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size used during evaluation.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run evaluation on: '0' for first GPU, 'cpu' for CPU-only.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="training/experiments",
        help="Root output folder where evaluation results (plots, JSON) are saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="official_evaluation",
        help="Name of this evaluation run (creates a subfolder under --project).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: '{weights_path}'.")

    print("=" * 60)
    print("ENDO-X — Polyp Detection Validation")
    print("=" * 60)
    print(f"Weights : {weights_path}")
    print(f"Dataset : {args.data}")
    print(f"Split   : {args.split}")
    print("=" * 60)

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=args.data,
        split=args.split,
        project=args.project,
        name=args.name,
        device=args.device,
        batch=args.batch,
        plots=True,
        save_json=True,
    )

    print("\n" + "=" * 60)
    print(f"RESULTS ON '{args.split.upper()}' SPLIT")
    print("=" * 60)
    print(f"Precision            : {metrics.box.mp:.4f}")
    print(f"Recall               : {metrics.box.mr:.4f}")
    print(f"mAP@0.5              : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95         : {metrics.box.map:.4f}")
    print("=" * 60)
    print(f"Plots and reports saved to: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()