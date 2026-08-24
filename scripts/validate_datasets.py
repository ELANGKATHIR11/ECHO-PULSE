import os
import zipfile
import glob
import cv2
import json
import numpy as np
from typing import Dict, Any

def extract_and_validate():
    print("=== EXTRACTING AND VALIDATING SEABED OBJECTS DATASET ===")
    src_dir = "data/downloaded/SeabedObjects"
    extract_dir = "data/extracted/SeabedObjects"
    unified_dir = "data/unified"
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(unified_dir, exist_ok=True)
    
    zip_files = glob.glob(os.path.join(src_dir, "*.zip"))
    print(f"Found {len(zip_files)} zip archives.")
    for zf in zip_files:
        print(f"[*] Extracting: {os.path.basename(zf)}...")
        with zipfile.ZipFile(zf, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
    # Validate extracted images
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.bmp", "*.tif"]:
        image_paths.extend(glob.glob(os.path.join(extract_dir, "**", ext), recursive=True))
        
    print(f"\n[*] Total extracted sonar images: {len(image_paths)}")
    
    valid_count = 0
    corrupt_count = 0
    sample_records = []
    
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            corrupt_count += 1
            continue
            
        h, w = img.shape[:2]
        is_plane = "plane" in img_path.lower()
        class_name = "shipwreck" if not is_plane else "marine_debris"
        
        valid_count += 1
        if valid_count <= 20:
            sample_records.append({
                "id": f"IMG-{idx:04d}",
                "filename": os.path.basename(img_path),
                "dimensions": {"width": w, "height": h},
                "class": class_name,
                "source": "SeabedObjects"
            })
            
    report = {
        "dataset": "SeabedObjects",
        "total_extracted": len(image_paths),
        "valid_images": valid_count,
        "corrupt_images": corrupt_count,
        "sample_records": sample_records
    }
    
    os.makedirs("reports/datasets", exist_ok=True)
    with open("reports/datasets/seabed_objects_validation.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[PASS] Successfully validated {valid_count} real sonar images ({corrupt_count} corrupt).")
    print(f"Validation report saved to reports/datasets/seabed_objects_validation.json")

if __name__ == "__main__":
    extract_and_validate()
