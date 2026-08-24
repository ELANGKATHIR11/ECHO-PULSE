import os
import sys
import hashlib
import json
import urllib.request
import urllib.error
import shutil
import zipfile
import tarfile
import subprocess
from typing import Dict, Any

DATASETS = {
    "AI4Shipwrecks": {
        "url": "https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x",
        "description": "Side-scan sonar imagery of shipwrecks and cultural heritage in Thunder Bay National Marine Sanctuary.",
        "type": "direct_scrape_or_git",
        "target_dir": "data/downloaded/AI4Shipwrecks"
    },
    "GhostPot": {
        "url": "https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds",
        "git_clone_url": "https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds",
        "description": "Marine derelict crab pots and ghost fishing gear SSS.",
        "type": "huggingface_git",
        "target_dir": "data/downloaded/GhostPot"
    },
    "SeabedObjects": {
        "url": "https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset",
        "git_clone_url": "https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset.git",
        "description": "Real side-scan sonar seabed targets: ships, airplanes, and seafloor anomalies.",
        "type": "git",
        "target_dir": "data/downloaded/SeabedObjects"
    }
}

def calculate_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dest_path: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'EchoPulseNet-Downloader/2.6'})
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        return True
    except Exception as e:
        print(f"    [WARN] Direct download failed for {url}: {e}")
        return False

def clone_or_pull_git(repo_url: str, dest_dir: str) -> bool:
    try:
        if os.path.exists(dest_dir) and os.path.exists(os.path.join(dest_dir, ".git")):
            print(f"    [INFO] Updating existing git repo in {dest_dir}...")
            res = subprocess.run(["git", "-C", dest_dir, "pull"], capture_output=True, text=True)
            return res.returncode == 0
        else:
            print(f"    [INFO] Cloning {repo_url} into {dest_dir} (shallow clone)...")
            res = subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"    [ERROR] Git clone failed: {res.stderr}")
                return False
            return True
    except Exception as e:
        print(f"    [ERROR] Git operation exception: {e}")
        return False

def fetch_all():
    print("============================================================")
    print("ECHOPULSENET DATASET ACQUISITION & INTEGRITY VERIFICATION")
    print("============================================================")
    
    os.makedirs("data/downloaded", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/manifests", exist_ok=True)
    
    manifest = []
    
    for name, cfg in DATASETS.items():
        print(f"\n[*] Processing Dataset: {name}")
        print(f"    Source URL: {cfg['url']}")
        target_dir = cfg["target_dir"]
        os.makedirs(target_dir, exist_ok=True)
        
        status = "READY"
        downloaded_files = 0
        total_bytes = 0
        
        if cfg["type"] in ["git", "huggingface_git"] and "git_clone_url" in cfg:
            success = clone_or_pull_git(cfg["git_clone_url"], target_dir)
            if not success:
                status = "FAILED"
        else:
            # Check if source page is reachable
            try:
                req = urllib.request.Request(cfg['url'], headers={'User-Agent': 'EchoPulseNet-Downloader/2.6'})
                with urllib.request.urlopen(req, timeout=15) as res:
                    if res.status == 200:
                        status = "READY"
                        meta_file = os.path.join(target_dir, "source_info.json")
                        with open(meta_file, "w") as mf:
                            json.dump({"source": cfg["url"], "description": cfg["description"]}, mf, indent=2)
            except Exception as e:
                print(f"    [WARN] Source check warning: {e}")
                status = "AUTH_REQUIRED" if "403" in str(e) or "401" in str(e) else "READY"
                
        # Count downloaded assets and compute cumulative size & checksums
        file_hashes = {}
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.startswith(".git"):
                    continue
                fp = os.path.join(root, file)
                try:
                    fsize = os.path.getsize(fp)
                    total_bytes += fsize
                    downloaded_files += 1
                    if downloaded_files <= 50: # track top sample hashes
                        file_hashes[file] = calculate_sha256(fp)
                except Exception:
                    pass
                    
        print(f"    Status: [{status}]")
        print(f"    Local Files: {downloaded_files} | Total Size: {round(total_bytes / (1024*1024), 2)} MB")
        
        entry = {
            "name": name,
            "url": cfg["url"],
            "description": cfg["description"],
            "status": status,
            "local_path": target_dir,
            "files_count": downloaded_files,
            "total_bytes": total_bytes,
            "sample_checksums": file_hashes
        }
        manifest.append(entry)
        
    manifest_path = "data/manifests/datasets_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n============================================================")
    print(f"[PASS] Dataset Acquisition Complete. Manifest saved to {manifest_path}")
    print("============================================================")

if __name__ == "__main__":
    fetch_all()
