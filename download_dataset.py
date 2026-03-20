"""
Dataset Download Script for Camouflaged Animals.

Downloads CAMO dataset (camouflaged animals & objects) from Kaggle.
Dataset: tankz/camo-dataset (1,250 images, 8 animal categories)

SETUP: Before running, configure Kaggle API credentials:
  1. Go to https://www.kaggle.com → Account → API → "Create New Token"
  2. Download kaggle.json and place it at:
     C:\\Users\\<YourName>\\.kaggle\\kaggle.json
  3. Run: python download_dataset.py
"""
import os
import sys
import shutil
import zipfile

# The Kaggle dataset to download. Options:
# "tankz/camo-dataset"             — CAMO: 1250 camouflaged animals (recommended, ~130MB)
# "stpete1/camo-plus-plus"         — CAMO++: larger with more images
KAGGLE_DATASET = "tankz/camo-dataset"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "animals")


def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up."""
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        print("=" * 60)
        print("  KAGGLE CREDENTIALS NOT FOUND")
        print("=" * 60)
        print(f"\nPlease set up Kaggle API credentials:")
        print("  1. Go to: https://www.kaggle.com/settings/account")
        print("  2. Scroll to 'API' section → click 'Create New Token'")
        print("  3. Save the downloaded kaggle.json to:")
        print(f"     {kaggle_json}")
        print("\nThen re-run this script.")
        return False
    print("  Kaggle credentials found!")
    return True


def download_camo_dataset():
    """Download the CAMO camouflaged animals dataset from Kaggle."""
    if not check_kaggle_credentials():
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nDownloading dataset: {KAGGLE_DATASET}")
    print(f"Output directory:   {OUT_DIR}")
    print("This may take a few minutes...\n")

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=OUT_DIR,
            unzip=True,
            quiet=False,
        )
        print(f"\nDataset downloaded successfully to: {OUT_DIR}")

    except Exception as e:
        print(f"\nERROR downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"  Extract to: {OUT_DIR}")
        sys.exit(1)

    # Show what was downloaded
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(OUT_DIR):
        level = root.replace(OUT_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:  # Only show first 2 levels
            for f in files[:5]:
                fpath = os.path.join(root, f)
                size_mb = os.path.getsize(fpath) / (1024 ** 2)
                print(f"{indent}  {f} ({size_mb:.1f} MB)")
            if len(files) > 5:
                print(f"{indent}  ... and {len(files)-5} more files")

    print("\nSetup complete! Run the Streamlit app with:")
    print("  venv\\Scripts\\activate")
    print("  streamlit run app.py")


if __name__ == "__main__":
    download_camo_dataset()
