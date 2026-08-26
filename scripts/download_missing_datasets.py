import os
import sys
import json
import time
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

# ==============================================================================
# Smart Incremental Dataset Downloader
# Downloads ONLY missing datasets, preserving existing data
# ==============================================================================

DOWNLOAD_TARGETS = [
    {
        "name": "OpenSonarDatasets_Mirror",
        "target_dir": Path("data/downloaded/OpenSonarDatasets_Full"),
        "url": "https://github.com/remaro-network/OpenSonarDatasets/archive/refs/heads/main.zip",
        "is_archive": True,
        "type": "zip"
    },
    {
        "name": "TrashCan_Sample_Debris",
        "target_dir": Path("data/downloaded/TrashCan_Debris"),
        "url": "https://raw.githubusercontent.com/karanwxliaa/Underwater-Trash-Detection/main/data_sample.json",
        "is_archive": False
    },
    {
        "name": "AquaScan_Metadata_Annotations",
        "target_dir": Path("data/downloaded/AquaScan_1K"),
        "url": "https://raw.githubusercontent.com/Jorwnpay/Awesome-Sonar-Image-Resources/main/README.md",
        "is_archive": False
    }
]

def download_missing_datasets():
    print("==================================================================")
    print("  INCREMENTAL OPEN-SOURCE MARINE DATASET DOWNLOADER               ")
    print("==================================================================")

    downloaded_count = 0
    skipped_count = 0

    for item in DOWNLOAD_TARGETS:
        name = item["name"]
        target_dir = item["target_dir"]
        url = item["url"]
        is_archive = item.get("is_archive", False)

        # Check if already exists and has files
        if target_dir.exists() and len(list(target_dir.glob("*.*"))) > 0:
            print(f"[*] [SKIP - ALREADY EXISTS] {name} at {target_dir} ({len(list(target_dir.glob('*.*')))} files)")
            skipped_count += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] [DOWNLOADING] {name} from {url}...")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            
            if is_archive:
                archive_temp = target_dir / "temp_download.zip"
                with urllib.request.urlopen(req, timeout=60) as response, open(archive_temp, 'wb') as out_file:
                    out_file.write(response.read())

                print(f"  --> Extracting {archive_temp} to {target_dir}...")
                with zipfile.ZipFile(archive_temp, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)

                if archive_temp.exists():
                    archive_temp.unlink()
            else:
                dest_file = target_dir / Path(url).name
                with urllib.request.urlopen(req, timeout=30) as response, open(dest_file, 'wb') as out_file:
                    out_file.write(response.read())

            file_count = len(list(target_dir.glob("**/*.*")))
            print(f"[PASS] Successfully downloaded and staged {name} ({file_count} items).")
            downloaded_count += 1

        except Exception as e:
            print(f"[!] Warning: Could not complete direct download for {name}: {e}")

    print(f"\n==================================================================")
    print(f"  DOWNLOAD SUMMARY: {downloaded_count} Downloaded | {skipped_count} Skipped (Preserved Existing)")
    print("==================================================================")

if __name__ == "__main__":
    download_missing_datasets()
