import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from training.segmentation.segment import PolypSegmenter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection and U-Net segmentation."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="models/detector/best.pt",
    )

    parser.add_argument(
        "--seg-weights",
        type=str,
        default="models/segmenter/best.pth",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="training/experiments/predictions",
    )

    return parser.parse_args()


def apply_segmentation(frame, box, segmenter):
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = [
        int(v) for v in box.xyxy[0].tolist()
    ]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return frame

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return frame

    mask = segmenter.predict(crop)
    mask_pixels = mask > 0

    if np.any(mask_pixels):
        green = np.zeros_like(crop)
        green[:, :, 1] = 255

        blended = cv2.addWeighted(
            crop,
            0.5,
            green,
            0.5,
            0,
        )

        crop[mask_pixels] = blended[mask_pixels]
        frame[y1:y2, x1:x2] = crop

    confidence = float(box.conf[0])

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    label = f"Polyp {confidence:.2f}"

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return frame


def process_image(
    model,
    segmenter,
    image_path,
    conf,
    imgsz,
    device,
    output_dir,
):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not open image: {image_path}"
        )

    results = model.predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=False,
        verbose=False,
    )

    result = results[0]
    annotated = image.copy()
    detections = 0

    for box in result.boxes:
        confidence = float(box.conf[0])

        if confidence < 0.25:
            continue

        annotated = apply_segmentation(
            annotated,
            box,
            segmenter,
        )

        detections += 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{image_path.stem}_segmented{image_path.suffix}"
    )

    cv2.imwrite(
        str(output_path),
        annotated,
    )

    print("=" * 60)
    print("IMAGE INFERENCE")
    print("=" * 60)
    print(f"Image      : {image_path}")
    print(f"Confidence : {conf}")
    print(f"Detections : {detections}")
    print(f"Output     : {output_path}")
    print("=" * 60)

    return detections


def process_video(
    model,
    segmenter,
    video_path,
    conf,
    imgsz,
    device,
    output_dir,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video file: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{video_path.stem}_segmented.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    frame_index = 0
    total_detections = 0

    print("=" * 60)
    print("VIDEO INFERENCE")
    print("=" * 60)
    print(f"Video      : {video_path}")
    print(f"Frames     : {total_frames}")
    print(f"Resolution : {width}x{height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Confidence : {conf}")
    print(f"Device     : {device}")
    print("=" * 60)

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
            save=False,
            verbose=False,
        )

        result = results[0]
        annotated = frame.copy()

        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < 0.25:
                continue

            annotated = apply_segmentation(
                annotated,
                box,
                segmenter,
            )

            total_detections += 1

        writer.write(annotated)

        frame_index += 1

        if (
            frame_index % 30 == 0
            or frame_index == total_frames
        ):
            print(
                f"Processed {frame_index}/{total_frames} "
                f"frames | Detections: {total_detections}"
            )

    cap.release()
    writer.release()

    print("=" * 60)
    print(f"Output video       : {output_path}")
    print(f"Total detections   : {total_detections}")
    print("=" * 60)

    return total_detections


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    seg_weights_path = Path(args.seg_weights)
    source_path = Path(args.source)
    output_dir = Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Detection weights not found: {weights_path}"
        )

    if not seg_weights_path.exists():
        raise FileNotFoundError(
            f"Segmentation weights not found: {seg_weights_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source not found: {source_path}"
        )

    model = YOLO(str(weights_path))

    segmenter = PolypSegmenter(
        str(seg_weights_path),
        device=args.device,
    )

    start_time = time.time()

    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        total_detections = process_image(
            model,
            segmenter,
            source_path,
            args.conf,
            args.imgsz,
            args.device,
            output_dir,
        )

    elif source_path.suffix.lower() in VIDEO_EXTENSIONS:
        total_detections = process_video(
            model,
            segmenter,
            source_path,
            args.conf,
            args.imgsz,
            args.device,
            output_dir,
        )

    else:
        raise ValueError(
            f"Unsupported file type: "
            f"{source_path.suffix}"
        )

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"Finished in {elapsed:.2f} seconds")
    print(f"Total detections: {total_detections}")
    print("=" * 60)


if __name__ == "__main__":
    main()