import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path(".")
KVASIR_IMAGES_DIR = BASE_DIR / "1000 images" / "Kvasir-SEG" / "Kvasir-SEG" / "images"
POLYP_DIR = BASE_DIR / "4K images" / "PolypDB" / "PolypDB" / "PolypDB_modality_wise"
RAW_LABELS_DIR = BASE_DIR / "01_Raw_Labels"
OUTPUT_DATASET_DIR = BASE_DIR / "YOLO_Dataset_Strict"

def build_image_registry():
    registry = {}
    if KVASIR_IMAGES_DIR.exists():
        for img_path in KVASIR_IMAGES_DIR.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                registry[img_path.stem] = img_path

    modalities = ["BLI", "FICE", "LCI", "NBI", "WLI"]
    for mod in modalities:
        img_folder = POLYP_DIR / mod / "images"
        if img_folder.exists():
            for img_path in img_folder.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    registry[img_path.stem] = img_path
    return registry

def assemble_three_way_split():
    # Create standard YOLO directories including the 'test' folder
    for split in ['train', 'val', 'test']:
        os.makedirs(OUTPUT_DATASET_DIR / 'images' / split, exist_ok=True)
        os.makedirs(OUTPUT_DATASET_DIR / 'labels' / split, exist_ok=True)

    image_registry = build_image_registry()
    valid_pairs = []

    for label_path in RAW_LABELS_DIR.glob("*.txt"):
        stem = label_path.stem
        if stem in image_registry:
            valid_pairs.append((image_registry[stem], label_path))

    print(f"Successfully matched {len(valid_pairs)} image-label pairs.")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(valid_pairs)

    # Strict 3-way split ratios: 70% Train, 15% Validation, 15% Unseen Test
    total_len = len(valid_pairs)
    train_end = int(total_len * 0.70)
    val_end = train_end + int(total_len * 0.15)

    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]

    def copy_files(pairs, split_name):
        for img_path, label_path in pairs:
            shutil.copy(img_path, OUTPUT_DATASET_DIR / 'images' / split_name / img_path.name)
            shutil.copy(label_path, OUTPUT_DATASET_DIR / 'labels' / split_name / label_path.name)

    print(f"Copying {len(train_pairs)} training files...")
    copy_files(train_pairs, 'train')

    print(f"Copying {len(val_pairs)} validation files...")
    copy_files(val_pairs, 'val')

    print(f"Copying {len(test_pairs)} unseen test files...")
    copy_files(test_pairs, 'test')

    print(f"Strict 3-way dataset assembly complete at: {OUTPUT_DATASET_DIR}")

if __name__ == "__main__":
    assemble_three_way_split()