"""
EchoPulseNet Project Archiver & Packager
Generates a complete, standalone, self-contained project archive zip file under 512 MB in the user's Downloads directory.
Includes:
  - Complete Frontend source (src/, public/, dist/, index.html, vite.config.ts, tsconfig.json)
  - Complete Backend source (backend/app/, backend/requirements.txt, etc.)
  - Complete Native Desktop Electron App (electron_main.js, package.json, scripts)
  - Complete ML Checkpoints & Models (models_checkpoints/*.pt, *.onnx)
  - Full Representative Datasets across all 4 domains:
      * 4-Channel AVS Vector Sensor dataset (120 packets)
      * Scraped FOSS Marine Hydrophone Audio (.wav files)
      * Scraped FOSS Side-Scan Sonar Imagery (.jpg files)
      * Representative Multi-Category Hydrophone Acoustic Dataset (182 WAV recordings across all 4 macro classes)
      * Side-Scan Sonar Object Detection Challenge dataset (1,071 files)
      * Strata 1D Acoustic Ping Inversion dataset
      * Biofouled Marine Expert Sonar Corpus
      * Sonar 8-Class Benchmark Images & Labels
      * All dataset manifests and ground truths
  - Test suites, deployment scripts, and architectural docs
"""

import os
import sys
import zipfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = Path(os.path.join(os.environ.get("USERPROFILE", "C:\\Users\\elang"), "Downloads"))
ZIP_NAME = "echopulsenet-marine-intelligence-platform-v2.6.0.zip"
TARGET_ZIP = DOWNLOADS_DIR / ZIP_NAME

# Exclusion patterns to keep archive clean, lean, and strictly under 512 MB
EXCLUDE_DIRS = {
    "node_modules", ".git", "__pycache__", ".pytest_cache", "runs", "cache",
    ".idea", ".vscode", "PROJECTS", "downloaded"
}
EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".tmp", ".log", ".DS_Store"
}

def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False

def build_zip():
    print("=" * 80)
    print(f"EchoPulseNet Project Master Packager (Target < 512 MB)")
    print(f"Destination: {TARGET_ZIP}")
    print("=" * 80)

    start_time = time.time()
    total_files = 0
    total_uncompressed = 0

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_ZIP.exists():
        TARGET_ZIP.unlink()

    # Pre-select representative audio files from hydrophone dataset (every 5th file across all 4 categories)
    hp_wavs = sorted((ROOT_DIR / "data" / "hydrophone_acoustic_dataset" / "audio").rglob("*.wav"))
    selected_hp_wavs = set(hp_wavs[::5])
    print(f"[*] Pre-selected {len(selected_hp_wavs)}/{len(hp_wavs)} representative hydrophone audio files across all 4 categories.")

    # Pre-select representative sonar images from hydrophys_8class/sonar/images/train (every 4th file to preserve diversity while fitting under 512MB)
    sonar_train_imgs = sorted((ROOT_DIR / "data" / "hydrophys_8class_dataset" / "sonar" / "images" / "train").glob("*.jpg"))
    selected_sonar_train = set(sonar_train_imgs[::4])
    print(f"[*] Pre-selected {len(selected_sonar_train)}/{len(sonar_train_imgs)} representative training sonar images.")

    with zipfile.ZipFile(TARGET_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_out:

        def add_file(fpath: Path, arc_name: str):
            nonlocal total_files, total_uncompressed
            if should_skip(fpath):
                return
            zip_out.write(fpath, arc_name)
            total_files += 1
            total_uncompressed += fpath.stat().st_size

        def add_directory(dpath: Path, arc_prefix: str):
            if not dpath.exists():
                return
            for root, dirs, files in os.walk(dpath):
                # prune excluded dirs
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    fp = Path(root) / file
                    if should_skip(fp):
                        continue
                    
                    # Hydrophone audio filtering
                    if "hydrophone_acoustic_dataset" in fp.parts and fp.suffix.lower() == ".wav":
                        if fp not in selected_hp_wavs:
                            continue

                    # Hydrophys sonar train images filtering
                    if "hydrophys_8class_dataset" in fp.parts and "train" in fp.parts and fp.suffix.lower() in [".jpg", ".png"]:
                        if fp not in selected_sonar_train:
                            continue

                    rel_path = fp.relative_to(ROOT_DIR)
                    add_file(fp, str(rel_path).replace("\\", "/"))

        # 1. Core source codes
        print("\n[1/6] Packaging Frontend Source (src/, dist/, public/, config)...")
        add_directory(ROOT_DIR / "src", "src")
        add_directory(ROOT_DIR / "public", "public")
        add_directory(ROOT_DIR / "dist", "dist")

        print("[2/6] Packaging Backend AI & Services (backend/, tests/, configs/)...")
        add_directory(ROOT_DIR / "backend", "backend")
        add_directory(ROOT_DIR / "tests", "tests")
        add_directory(ROOT_DIR / "configs", "configs")
        add_directory(ROOT_DIR / "scripts", "scripts")

        print("[3/6] Packaging Desktop Electron App & Configs...")
        for root_file in [
            "electron_main.js", "package.json", "package-lock.json", "index.html",
            "vite.config.ts", "tsconfig.json", "README.md", "LICENSE",
            "ARCHITECTURE_AND_BLOCK_DIAGRAM.md", "ECHOPULSENET_COMPLETE_TECHNICAL_REFERENCE.md",
            "MODELS_AND_ARCHITECTURE_ANALYSIS.md", "MODELS_LICENSE.md", "PHYSICS_TENSOR.md",
            "Launch_EchoPulseNet.bat", "Install_Desktop_Shortcut.bat", ".env.example"
        ]:
            rf = ROOT_DIR / root_file
            if rf.exists():
                add_file(rf, root_file)

        print("[4/6] Packaging ALL Retrained Deep Learning Models & Weights (models_checkpoints/)...")
        add_directory(ROOT_DIR / "models_checkpoints", "models_checkpoints")

        print("[5/6] Packaging Datasets & Manifests (data/)...")
        # Manifests and root json
        for mf in (ROOT_DIR / "data").glob("*.json"):
            add_file(mf, f"data/{mf.name}")
        for mf in (ROOT_DIR / "data" / "manifests").glob("*.json"):
            add_file(mf, f"data/manifests/{mf.name}")

        # AVS Vector Sensor Dataset (Complete 120 packets)
        add_directory(ROOT_DIR / "data" / "avs_vector_dataset", "data/avs_vector_dataset")

        # Scraped FOSS Audio & Sonar Images (Complete)
        add_directory(ROOT_DIR / "data" / "scraped_foss_hydrophone_audio", "data/scraped_foss_hydrophone_audio")
        add_directory(ROOT_DIR / "data" / "scraped_foss_sonar_images", "data/scraped_foss_sonar_images")

        # Side-Scan Sonar Object Detection Challenge dataset (Complete)
        add_directory(ROOT_DIR / "data" / "side-scan-sonar-object-detection-challenge", "data/side-scan-sonar-object-detection-challenge")

        # Extracted datasets
        add_directory(ROOT_DIR / "data" / "extracted", "data/extracted")

        # Hydrophone Acoustic Dataset: Manifests + 182 Diverse WAV Recordings across all 4 categories
        hp_dir = ROOT_DIR / "data" / "hydrophone_acoustic_dataset"
        if hp_dir.exists():
            for mf in hp_dir.glob("*.json"):
                add_file(mf, f"data/hydrophone_acoustic_dataset/{mf.name}")
            for wav in selected_hp_wavs:
                rel = wav.relative_to(ROOT_DIR)
                add_file(wav, str(rel).replace("\\", "/"))

        # Hydrophys 8-Class Dataset:
        # Include strata 1D pings, full labels, test images, metadata manifests, and sample train images
        # Exclude massive duplicate raw 'unified' (820MB) and 'optical' (82MB) folders
        for folder in ["strata_1d_pings", "labels", "images/test"]:
            p = ROOT_DIR / "data" / "hydrophys_8class_dataset" / "sonar" / folder if folder != "strata_1d_pings" else ROOT_DIR / "data" / "hydrophys_8class_dataset" / folder
            if p.exists():
                add_directory(p, f"data/hydrophys_8class_dataset/{folder}")

        # Add sampled training sonar images
        for p in selected_sonar_train:
            rel = p.relative_to(ROOT_DIR)
            add_file(p, str(rel).replace("\\", "/"))

        # Add manifest files from hydrophys_8class_dataset
        for mf in (ROOT_DIR / "data" / "hydrophys_8class_dataset").glob("*.json"):
            add_file(mf, f"data/hydrophys_8class_dataset/{mf.name}")
        for mf in (ROOT_DIR / "data" / "hydrophys_8class_dataset").glob("*.yaml"):
            add_file(mf, f"data/hydrophys_8class_dataset/{mf.name}")

        print("[6/6] Packaging Reports and Documentation...")
        add_directory(ROOT_DIR / "reports", "reports")

    elapsed = round(time.time() - start_time, 2)
    zip_size_bytes = TARGET_ZIP.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Archive generated successfully in {elapsed}s!")
    print(f"Archive Path     : {TARGET_ZIP}")
    print(f"Total Files      : {total_files}")
    print(f"Uncompressed Size: {total_uncompressed / (1024*1024):.2f} MB")
    print(f"Compressed Size  : {zip_size_mb:.2f} MB")
    print(f"Target Constraint: < 512.00 MB -> {'PASSED [OK]' if zip_size_mb < 512.0 else 'FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    build_zip()
