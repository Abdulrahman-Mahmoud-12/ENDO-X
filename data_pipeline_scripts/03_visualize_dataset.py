import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

# Define specific paths for the 4K dataset and the previously generated raw labels
POLYP_DIR = Path("4K images/PolypDB/PolypDB/PolypDB_modality_wise")
RAW_LABELS_DIR = Path("01_Raw_Labels")

def visualize_mask_and_bbox(num_samples=3):
    image_mask_pairs = []
    modalities = ["BLI", "FICE", "LCI", "NBI", "WLI"]
    
    # Traverse the directories to find valid image and corresponding mask pairs dynamically
    for mod in modalities:
        img_folder = POLYP_DIR / mod / "images"
        mask_folder = POLYP_DIR / mod / "masks"
        
        if img_folder.exists() and mask_folder.exists():
            for img_path in img_folder.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    # Construct the expected mask path by searching for the exact stem
                    mask_candidates = list(mask_folder.glob(f"{img_path.stem}.*"))
                    if mask_candidates:
                        image_mask_pairs.append((img_path, mask_candidates[0]))
                    
    if len(image_mask_pairs) < num_samples:
        print("Error: Not enough image-mask pairs found for visualization.")
        return

    # Randomly sample pairs for an unbiased visual check
    selected_pairs = random.sample(image_mask_pairs, num_samples)
    
    # Create a dynamic subplot grid (num_samples rows, 2 columns)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    
    for idx, (img_path, mask_path) in enumerate(selected_pairs):
        # Read and process the original endoscopic image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        
        # Read the corresponding ground truth mask in grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Locate the programmatic label file generated in Stage 1
        label_path = RAW_LABELS_DIR / f"{img_path.stem}.txt"
        
        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Extract YOLO normalized format coordinates
                    x_center, y_center = float(parts[1]), float(parts[2])
                    width, height = float(parts[3]), float(parts[4])
                    
                    # Denormalize mathematically back to absolute pixel dimensions
                    x_center_pix = int(x_center * img_w)
                    y_center_pix = int(y_center * img_h)
                    width_pix = int(width * img_w)
                    height_pix = int(height * img_h)
                    
                    # Compute geometrical corners of the bounding box
                    x_min = int(x_center_pix - (width_pix / 2))
                    y_min = int(y_center_pix - (height_pix / 2))
                    x_max = int(x_center_pix + (width_pix / 2))
                    y_max = int(y_center_pix + (height_pix / 2))
                    
                    # Overlay the bounding box using a distinct green color
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        # Plot the binary mask on the left column
        axes[idx, 0].imshow(mask, cmap='gray')
        modality_name = img_path.parent.parent.name
        axes[idx, 0].set_title(f"Original Mask [{modality_name}]")
        axes[idx, 0].axis("off")
        
        # Plot the final image with the bounding box on the right column
        axes[idx, 1].imshow(img)
        axes[idx, 1].set_title(f"Generated BBox")
        axes[idx, 1].axis("off")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Fetching corresponding masks and bounding boxes...")
    visualize_mask_and_bbox()