import os
import matplotlib.pyplot as plt
from pathlib import Path

# Define the path to the unified raw labels generated in Stage 1
RAW_LABELS_DIR = Path("01_Raw_Labels")

def perform_bbox_eda():
    widths = []
    heights = []
    areas = []
    
    # Iterate through all generated text files
    for label_file in RAW_LABELS_DIR.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                # Extract normalized width and height from YOLO format
                w = float(parts[3])
                h = float(parts[4])
                
                # Calculate the normalized area of the bounding box
                area = w * h
                
                widths.append(w)
                heights.append(h)
                areas.append(area)

    print(f"Total Bounding Boxes Analyzed: {len(widths)}")
    
    # Create a figure with two subplots for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram of Bounding Box Areas
    ax1.hist(areas, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Polyp Bounding Box Areas')
    ax1.set_xlabel('Normalized Area (0 to 1)')
    ax1.set_ylabel('Frequency')
    
    # Plot 2: Scatter Plot of Width vs Height to visualize aspect ratios
    ax2.scatter(widths, heights, alpha=0.3, color='coral')
    ax2.set_title('Polyp Dimensions: Width vs Height')
    ax2.set_xlabel('Normalized Width')
    ax2.set_ylabel('Normalized Height')
    
    # Plot a diagonal line representing perfectly square bounding boxes (1:1 aspect ratio)
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1, label='1:1 Ratio')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not RAW_LABELS_DIR.exists():
        print(f"Error: Directory {RAW_LABELS_DIR} not found.")
    else:
        perform_bbox_eda()