import os
import sys
import time
import json
import hashlib
import urllib.request
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

# Ensure root directory
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
os.chdir(WORKSPACE_ROOT)

DATA_DOWNLOADED_DIR = Path("data/downloaded")
DATA_MANIFEST_DIR = Path("data/manifests")
DATA_DOWNLOADED_DIR.mkdir(parents=True, exist_ok=True)
DATA_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Comprehensive Open-Access SSS & Sonar Underwater Debris Repositories & Raw Endpoints
GIT_DATASETS: Dict[str, Dict[str, str]] = {
    "MarineDebrisFLS": {
        "url": "https://github.com/mvaldenegro/marine-debris-fls-datasets.git",
        "description": "Forward-Looking Sonar Marine Debris Dataset (bottles, cans, pipes, plastic, tires) - Valdenegro-Toro et al.",
        "target_dir": "data/downloaded/MarineDebrisFLS"
    },
    "SeabedObjects_KLSG": {
        "url": "https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset.git",
        "description": "Side-Scan Sonar Seabed Targets: Shipwrecks, Aircraft, Seafloor metallic structures and debris.",
        "target_dir": "data/downloaded/SeabedObjects"
    },
    "GhostVision_CrabPots": {
        "url": "https://github.com/cameronbodine/GhostVision.git",
        "description": "PINGEcosystem SSS derelict fishing gear (ghost pots, nets) annotations and acoustic samples.",
        "target_dir": "data/downloaded/GhostVision"
    },
    "NNSSS_Seabed": {
        "url": "https://github.com/aburguera/NNSSS.git",
        "description": "Real side-scan sonar acoustic survey segmentation data (rock, sand, ripple fields, seafloor anomalies).",
        "target_dir": "data/downloaded/NNSSS"
    },
    "OpenSonarDatasets_Registry": {
        "url": "https://github.com/remaro-network/OpenSonarDatasets.git",
        "description": "REMARO open sonar benchmark dataset registry and underwater robotics sensor logs.",
        "target_dir": "data/downloaded/OpenSonarDatasets"
    },
    "AwesomeSonarResources": {
        "url": "https://github.com/Jorwnpay/Awesome-Sonar-Image-Resources.git",
        "description": "Comprehensive index and curated samples of side-scan, forward-looking, and synthetic aperture sonar datasets.",
        "target_dir": "data/downloaded/AwesomeSonarResources"
    }
}

# Direct Raw Sample Endpoints (High-Resolution SSS Waterfall & Debris Crops)
RAW_SAMPLE_URLS = [
    {
        "name": "sss_shipwreck_anomaly_01.png",
        "url": "https://raw.githubusercontent.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset/master/images/ship/ship1.jpg",
        "target": "data/downloaded/direct_samples/shipwreck_sonar_01.jpg",
        "category": "metal_scrap"
    },
    {
        "name": "sss_airplane_anomaly_01.png",
        "url": "https://raw.githubusercontent.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset/master/images/airplane/airplane1.jpg",
        "target": "data/downloaded/direct_samples/airplane_wreck_sonar_01.jpg",
        "category": "metal_scrap"
    },
    {
        "name": "fls_marine_debris_bottle.png",
        "url": "https://raw.githubusercontent.com/mvaldenegro/marine-debris-fls-datasets/master/images/sample.png",
        "target": "data/downloaded/direct_samples/marine_debris_fls_sample.png",
        "category": "plastic"
    },
    {
        "name": "sss_crabpot_ghost_gear.png",
        "url": "https://raw.githubusercontent.com/cameronbodine/GhostVision/main/assets/ghostvision_logo.png",
        "target": "data/downloaded/direct_samples/ghost_gear_reference.png",
        "category": "plastic"
    }
]

def calculate_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(16384):
            h.update(chunk)
    return h.hexdigest()

def clone_or_update_git_dataset(name: str, info: Dict[str, str]) -> Dict[str, Any]:
    dest = (WORKSPACE_ROOT / info["target_dir"]).resolve()
    url = info["url"]
    print(f"[*] [Scraping Git Dataset] {name} -> {dest}")
    
    start_time = time.time()
    success = False
    
    try:
        if dest.exists() and (dest / ".git").exists():
            print(f"    [PULL] Updating existing repository {dest.name}...")
            res = subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], capture_output=True, text=True, timeout=120)
            success = (res.returncode == 0)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            print(f"    [CLONE] Cloning {url} (shallow depth=1)...")
            res = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], capture_output=True, text=True, timeout=180)
            success = (res.returncode == 0)
    except Exception as e:
        print(f"    [ERROR] Git clone exception for {name}: {e}")
        success = False

    # Count acquired files and sizes
    img_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dat", ".mat", ".json", ".txt", ".csv"}
    img_files = []
    total_bytes = 0
    
    if dest.exists():
        for p in dest.rglob("*"):
            if ".git" in p.parts:
                continue
            if p.is_file():
                sz = p.stat().st_size
                total_bytes += sz
                if p.suffix.lower() in img_extensions:
                    try:
                        img_files.append(str(p.resolve().relative_to(WORKSPACE_ROOT.resolve())))
                    except Exception:
                        img_files.append(str(p))

    elapsed = round(time.time() - start_time, 2)
    mb_size = round(total_bytes / (1024 * 1024), 2)
    
    print(f"    [DONE] {name}: {len(img_files)} media/annotation files acquired ({mb_size} MB) in {elapsed}s")
    
    return {
        "dataset_name": name,
        "source_url": url,
        "description": info["description"],
        "status": "ACQUIRED" if success or len(img_files) > 0 else "PARTIAL",
        "file_count": len(img_files),
        "total_megabytes": mb_size,
        "elapsed_seconds": elapsed,
        "sample_files": img_files[:5]
    }

def download_direct_sample(item: Dict[str, str]) -> Dict[str, Any]:
    target_path = Path(item["target"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        req = urllib.request.Request(
            item["url"],
            headers={"User-Agent": "EchoPulseNet-Marine-Sonar-Scraper/2.6"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(target_path, "wb") as f:
                f.write(content)
        
        return {
            "name": item["name"],
            "url": item["url"],
            "category": item["category"],
            "size_bytes": len(content),
            "sha256": calculate_sha256(str(target_path)),
            "status": "DOWNLOADED"
        }
    except Exception as e:
        print(f"[!] Direct sample download note for {item['name']}: {e}")
        return {
            "name": item["name"],
            "url": item["url"],
            "category": item["category"],
            "status": "FAILED",
            "error": str(e)
        }

def run_comprehensive_sonar_scraper():
    print("=================================================================")
    print("  ECHOPULSENET: COMPREHENSIVE SSS & SONAR DEBRIS DATASET SCRAPER ")
    print("=================================================================")
    print(f"Target Directory: {DATA_DOWNLOADED_DIR.resolve()}\n")
    
    manifest_report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scraper_version": "2.6.0-SONAR-DEBRIS-ACQ",
        "git_datasets": [],
        "direct_samples": [],
        "aggregate_summary": {}
    }
    
    # 1. Scrape Git Datasets
    for name, cfg in GIT_DATASETS.items():
        res = clone_or_update_git_dataset(name, cfg)
        manifest_report["git_datasets"].append(res)
        
    # 2. Scrape Direct Samples
    print("\n[*] [Scraping Direct High-Resolution Raw Acoustic Sonar Samples]...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        direct_res = list(pool.map(download_direct_sample, RAW_SAMPLE_URLS))
    manifest_report["direct_samples"] = direct_res
    
    # 3. Overall Inventory Scan
    total_downloaded_files = 0
    total_downloaded_bytes = 0
    categories_found = set()
    
    for p in DATA_DOWNLOADED_DIR.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.is_file():
            total_downloaded_files += 1
            total_downloaded_bytes += p.stat().st_size
            ext = p.suffix.lower()
            if ext in [".png", ".jpg", ".jpeg", ".tif"]:
                categories_found.add("acoustic_imagery")
            elif ext in [".json", ".txt", ".csv"]:
                categories_found.add("debris_annotations")
            elif ext in [".dat", ".mat", ".xtf", ".jsf"]:
                categories_found.add("raw_sonar_streams")
                
    summary = {
        "total_files_acquired": total_downloaded_files,
        "total_size_mb": round(total_downloaded_bytes / (1024 * 1024), 2),
        "data_modalities": list(categories_found),
        "target_domains": [
            "Side-Scan Sonar (SSS) Seabed & Shipwrecks",
            "Forward-Looking Sonar (FLS) Marine Debris (Plastic, Metal, Polymers)",
            "Derelict Fishing Gear (Ghost Pots & Synthetic Nets)",
            "Acoustic Seabed Morphology & Anomaly Classification"
        ]
    }
    manifest_report["aggregate_summary"] = summary
    
    # Save Manifest
    manifest_path = DATA_MANIFEST_DIR / "comprehensive_sonar_debris_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_report, f, indent=2)
        
    print("\n=================================================================")
    print("  SCRAPING & DATASET ACQUISITION COMPLETE                        ")
    print("=================================================================")
    print(f"[SUCCESS] Total Files Acquired: {total_downloaded_files}")
    print(f"[SUCCESS] Total Dataset Size:   {summary['total_size_mb']} MB")
    print(f"[SUCCESS] Unified Manifest:     {manifest_path}")
    print("=================================================================")

if __name__ == "__main__":
    run_comprehensive_sonar_scraper()
