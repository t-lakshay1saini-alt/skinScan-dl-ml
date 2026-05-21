"""
Download and prepare datasets for SkinScan AI.

Downloads HAM10000 from Kaggle and organizes it into the expected structure.
Can be run standalone or imported.

Usage:
  python scripts/download_data.py
"""

import os
import sys
import zipfile
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, HAM10000_DIR, HAM10000_IMAGES_DIR


def download_and_extract():
    """Download HAM10000 from Kaggle and extract to data/ham10000/."""
    os.makedirs(HAM10000_DIR, exist_ok=True)

    zip_path = os.path.join(HAM10000_DIR, "skin-cancer-mnist-ham10000.zip")

    # Download if not already present
    if not os.path.exists(zip_path) and not os.path.exists(HAM10000_IMAGES_DIR):
        print("Downloading HAM10000 from Kaggle...")
        ret = os.system(
            f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 "
            f"-p {HAM10000_DIR}"
        )
        if ret != 0:
            print("ERROR: Kaggle download failed. Ensure kaggle.json is configured.")
            sys.exit(1)

    # Extract zip if present
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(HAM10000_DIR)
        os.remove(zip_path)
        print("Extraction complete.")

    # Consolidate images into single directory
    os.makedirs(HAM10000_IMAGES_DIR, exist_ok=True)
    for part_dir_name in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        part_dir = os.path.join(HAM10000_DIR, part_dir_name)
        if os.path.isdir(part_dir):
            print(f"Moving images from {part_dir_name}...")
            for img_file in os.listdir(part_dir):
                src = os.path.join(part_dir, img_file)
                dst = os.path.join(HAM10000_IMAGES_DIR, img_file)
                if os.path.isfile(src) and not os.path.exists(dst):
                    os.rename(src, dst)
            # Clean up empty directory
            if not os.listdir(part_dir):
                os.rmdir(part_dir)

    # Verify
    n_images = len(glob.glob(os.path.join(HAM10000_IMAGES_DIR, "*.jpg")))
    print(f"\nDataset ready: {n_images} images in {HAM10000_IMAGES_DIR}")

    # Check metadata CSV
    csv_candidates = glob.glob(os.path.join(HAM10000_DIR, "*metadata*"))
    if csv_candidates:
        print(f"Metadata CSV: {csv_candidates[0]}")
    else:
        print("WARNING: No metadata CSV found. Check the download.")

    return n_images


if __name__ == "__main__":
    download_and_extract()
