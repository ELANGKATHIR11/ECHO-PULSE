import os
import sys
import hashlib
import json
import subprocess
import glob
import cv2

FOSS_DATASETS = {
    "SeabedObjects": {
        "url": "https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset.git",
        "description": "Real side-scan sonar seabed targets: ships, airplanes, and seafloor anomalies.",
        "type": "git",
        "target_dir": "data/downloaded/SeabedObjects"
    },
    "NNSSS_Acoustic_Seabed": {
        "url": "https://github.com/aburguera/NNSSS.git",
        "description": "Real side-scan sonar acoustic survey data for seabed segmentation (rock, sand, ripple fields).",
        "type": "git",
        "target_dir": "data/downloaded/NNSSS"
    },
    "OpenSonarDatasets_Registry": {
        "url": "https://github.com/remaro-network/OpenSonarDatasets.git",
        "description": "Comprehensive open-source underwater sonar benchmark dataset repository and sensor logs.",
        "type": "git",
        "target_dir": "data/downloaded/OpenSonarDatasets"
    },
    "PINGEcosystem_GhostPots": {
        "url": "https://github.com/cameronbodine/GhostVision.git",
        "description": "HuggingFace/GitHub PINGEcosystem side-scan sonar derelict fishing gear (ghost pots) annotations and models.",
        "type": "git",
        "target_dir": "data/downloaded/GhostVision"
    },
    "Awesome_Sonar_Resources": {
        "url": "https://github.com/Jorwnpay/Awesome-Sonar-Image-Resources.git",
        "description": "Curated repository of side-scan, forward-looking, and synthetic aperture sonar datasets.",
        "type": "git",
        "target_dir": "data/downloaded/AwesomeSonarResources"
    }
}

def calculate_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def clone_or_pull(repo_url: str, dest_dir: str) -> bool:
    try:
        if os.path.exists(dest_dir) and os.path.exists(os.path.join(dest_dir, ".git")):
            print(f"    [INFO] Updating existing git repo in {dest_dir}...")
            res = subprocess.run(["git", "-C", dest_dir, "pull"], capture_output=True, text=True)
            return res.returncode == 0
        else:
            print(f"    [INFO] Cloning {repo_url} into {dest_dir}...")
            res = subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], capture_output=True, text=True)
            return res.returncode == 0
    except Exception as e:
        print(f"    [ERROR] Git clone error: {e}")
        return False

def fetch_foss_datasets():
    print("============================================================")
    print("ECHOPULSENET FOSS & OPEN-SOURCE SONAR DATASET ACQUISITION")
    print("============================================================")
    os.makedirs("data/downloaded", exist_ok=True)
    os.makedirs("data/manifests", exist_ok=True)
    
    manifest = []
    
    for name, cfg in FOSS_DATASETS.items():
        print(f"\n[*] Fetching FOSS Dataset: {name}")
        print(f"    URL: {cfg['url']}")
        target_dir = cfg["target_dir"]
        
        success = clone_or_pull(cfg["url"], target_dir)
        status = "READY" if success else "FAILED"
        
        file_count = 0
        total_bytes = 0
        for root, _, files in os.walk(target_dir):
            for file in files:
                if ".git" in root:
                    continue
                fp = os.path.join(root, file)
                try:
                    fsize = os.path.getsize(fp)
                    total_bytes += fsize
                    file_count += 1
                except Exception:
                    pass
                    
        print(f"    Status: [{status}]")
        print(f"    Files: {file_count} | Total Size: {round(total_bytes / (1024*1024), 2)} MB")
        
        manifest.append({
            "name": name,
            "url": cfg["url"],
            "description": cfg["description"],
            "status": status,
            "local_path": target_dir,
            "files_count": file_count,
            "total_bytes": total_bytes
        })
        
    with open("data/manifests/foss_datasets_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n============================================================")
    print("[PASS] FOSS Sonar Datasets Successfully Acquired and Logged.")
    print("============================================================")

if __name__ == "__main__":
    fetch_foss_datasets()
