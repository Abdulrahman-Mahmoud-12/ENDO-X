import os
import csv
import cv2
from pathlib import Path

# Define base paths according to the established directory tree
BASE_DIR = Path(".")
KVASIR_IMAGES_DIR = BASE_DIR / "1000 images" / "Kvasir-SEG" / "Kvasir-SEG" / "images"
KVASIR_BBOX_DIR = BASE_DIR / "1000 images" / "Kvasir-SEG" / "Kvasir-SEG" / "bbox"
POLYP_DIR = BASE_DIR / "4K images" / "PolypDB" / "PolypDB" / "PolypDB_modality_wise"

# Define the output directory for this specific pipeline stage
RAW_LABELS_DIR = BASE_DIR / "01_Raw_Labels"
os.makedirs(RAW_LABELS_DIR, exist_ok=True)

def extract_polypdb_labels():
    modalities = ["BLI", "FICE", "LCI", "NBI", "WLI"]
    extracted_count = 0
    
    # Define allowed image extensions for robust dataset parsing
    allowed_extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    
    for mod in modalities:
        mask_folder = POLYP_DIR / mod / "masks"
        if not mask_folder.exists():
            continue
            
        # Iterate through all files in the directory
        for mask_file in mask_folder.iterdir():
            # Check if the current file matches our allowed image extensions
            if mask_file.suffix.lower() not in allowed_extensions:
                continue
                
            # Read the mask image in grayscale mode
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            img_height, img_width = mask.shape
            
            # Find contours to locate the polyp boundaries
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            txt_path = RAW_LABELS_DIR / f"{mask_file.stem}.txt"
            has_valid_box = False
            
            with open(txt_path, "w") as f:
                for cnt in contours:
                    # Ignore tiny artifacts by setting a minimum area threshold
                    if cv2.contourArea(cnt) < 15:
                        continue
                        
                    # Calculate the bounding rectangle coordinates
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Convert absolute coordinates to YOLO normalized coordinates
                    x_center = (x + w / 2.0) / img_width
                    y_center = (y + h / 2.0) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height
                    
                    # Write the class index (0) and the normalized coordinates
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    has_valid_box = True
            
            # Remove the file if no valid bounding boxes were detected
            if not has_valid_box:
                os.remove(txt_path)
            else:
                extracted_count += 1
                
    return extracted_count

def extract_kvasir_labels():
    csv_files = list(KVASIR_BBOX_DIR.glob("*.csv"))
    extracted_count = 0
    
    for csv_path in csv_files:
        # Construct the path to the corresponding image to extract its shape
        img_path = KVASIR_IMAGES_DIR / f"{csv_path.stem}.jpg"
        if not img_path.exists():
            continue
            
        # Read the image strictly to get the height and width
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        txt_path = RAW_LABELS_DIR / f"{csv_path.stem}.txt"
        
        with open(txt_path, "w") as f_out, open(csv_path, "r") as f_in:
            reader = csv.DictReader(f_in)
            has_valid_box = False
            
            for row in reader:
                try:
                    # Parse bounding box coordinates from the CSV columns
                    xmin = float(row['xmin'])
                    ymin = float(row['ymin'])
                    xmax = float(row['xmax'])
                    ymax = float(row['ymax'])
                    
                    # Convert coordinates to YOLO normalized format
                    x_center = ((xmin + xmax) / 2.0) / img_width
                    y_center = ((ymin + ymax) / 2.0) / img_height
                    bbox_width = (xmax - xmin) / img_width
                    bbox_height = (ymax - ymin) / img_height
                    
                    # Write the formatted data to the text file
                    f_out.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    has_valid_box = True
                except KeyError:
                    continue
            
            # Increment the counter only if a bounding box was successfully processed
            if has_valid_box:
                extracted_count += 1
            else:
                os.remove(txt_path)
                
    return extracted_count

if __name__ == "__main__":
    print("Starting Stage 1: Data Extraction Pipeline...")
    
    polyp_count = extract_polypdb_labels()
    print(f"Extracted {polyp_count} labels from PolypDB masks.")
    
    kvasir_count = extract_kvasir_labels()
    print(f"Extracted {kvasir_count} labels from Kvasir-SEG CSVs.")
    
    total = polyp_count + kvasir_count
    print(f"Total unified labels generated in '{RAW_LABELS_DIR.name}': {total}")