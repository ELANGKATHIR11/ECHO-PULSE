import os
import sys
import subprocess
import json

EXTENDED_REPOS = {
    "Awesome_Sonar_Image_Resources": {
        "url": "https://github.com/Jorwnpay/Awesome-Sonar-Image-Resources.git",
        "description": "Comprehensive index and benchmark of side-scan and forward-looking sonar datasets.",
        "target_dir": "data/downloaded/Awesome-Sonar-Image-Resources"
    },
    "SidescanTools": {
        "url": "https://github.com/sonoware/sidescantools.git",
        "description": "GhostNetBusters side-scan sonar XTF tools, slant-range corrections, and acoustic calibration.",
        "target_dir": "data/downloaded/SidescanTools"
    }
}

def clone_repo(repo_url: str, dest_dir: str):
    if os.path.exists(dest_dir) and os.path.exists(os.path.join(dest_dir, ".git")):
        print(f"[*] Updating {dest_dir}...")
        subprocess.run(["git", "-C", dest_dir, "pull"], capture_output=True, text=True)
    else:
        print(f"[*] Cloning {repo_url} into {dest_dir}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], capture_output=True, text=True)

def fetch_all_extended():
    print("=== DOWNLOADING EXTENDED FOSS SONAR DATASETS AND REPOSITORIES ===")
    os.makedirs("data/downloaded", exist_ok=True)
    
    for name, cfg in EXTENDED_REPOS.items():
        clone_repo(cfg["url"], cfg["target_dir"])
        
    print("[PASS] Extended FOSS sonar resources downloaded successfully.")

if __name__ == "__main__":
    fetch_all_extended()
