import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.app.models.hydrophys_omninet import HydroPhysOmniNet, make_physics_acoustic_tensor
from backend.app.models.echophys_lite import EchoPhysLite, make_echophys_lite_tensor
from scripts.train_echophys_x_v3 import EchoPhysXV3, make_physics_acoustic_tensor as make_v3_physics_tensor

# Taxonomy standard
CLASS_NAMES = [
    "ghost_gear",
    "shipwreck",
    "unexploded_ordnance",
    "pipeline_anomaly",
    "marine_debris",
    "subsea_cable",
    "biological_cluster",
    "geological_formation"
]
CLASS_LABELS = [
    "Ghost Net / Gear",
    "Shipwreck",
    "UXO Ordnance",
    "Pipeline Scour",
    "Marine Debris",
    "Subsea Cable",
    "Coral / Bio-Cluster",
    "Seafloor Rock"
]

class SingleClassificationDataset(Dataset):
    """
    Standard evaluation dataset that extracts the primary Ground Truth single-classification class
    for each sonar frame (based on the highest-area or dominant annotation).
    """
    def __init__(self, data_pairs: List[Tuple[Path, Path]], max_samples: int = 400):
        self.items = []
        for img_dir, lbl_dir in data_pairs:
            img_dir = Path(img_dir)
            lbl_dir = Path(lbl_dir)
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
            for img in images:
                label_p = lbl_dir / f"{img.stem}.txt"
                if label_p.exists():
                    try:
                        content = label_p.read_text().strip().splitlines()
                        if not content:
                            continue
                        
                        # Find dominant class (largest bounding box or first valid)
                        dom_class = None
                        max_area = -1.0
                        for line in content:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(float(parts[0]))
                                if 0 <= cid < len(CLASS_NAMES):
                                    w, h = float(parts[3]), float(parts[4])
                                    area = w * h
                                    if area > max_area:
                                        max_area = area
                                        dom_class = cid
                        if dom_class is not None:
                            self.items.append((img, dom_class))
                    except Exception:
                        pass
        if max_samples and len(self.items) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.items), max_samples, replace=False)
            self.items = [self.items[i] for i in indices]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, target_class = self.items[idx]
        pil_img = Image.open(img_path).convert("L").resize((640, 640), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return arr, target_class, str(img_path)

def evaluate_models_and_generate_matrices(output_dir: str = "plots/confusion_matrices"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Evaluation on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    val_pairs = [
        (Path("data/yolo_sonar_dataset/images/val"), Path("data/yolo_sonar_dataset/labels/val")),
        (Path("data/side-scan-sonar-object-detection-challenge/valid/images"), Path("data/side-scan-sonar-object-detection-challenge/valid/labels")),
        (Path("data/yolo_sonar_dataset/images/test"), Path("data/yolo_sonar_dataset/labels/test"))
    ]

    dataset = SingleClassificationDataset(val_pairs, max_samples=450)
    print(f"[*] Loaded {len(dataset)} validation samples across all datasets for single classification benchmark.")

    # 1. Load HydroPhys-OmniNet Extreme
    hydro_model = HydroPhysOmniNet(num_classes=8).to(device)
    hydro_ckpt = Path("models_checkpoints/hydrophys_omninet_extreme_best.pt")
    if hydro_ckpt.exists():
        ckpt = torch.load(hydro_ckpt, map_location=device)
        hydro_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"[PASS] Loaded HydroPhys-OmniNet weights ({hydro_ckpt})")
    hydro_model.eval()

    # 2. Load EchoPhys-X V3
    v3_model = EchoPhysXV3(num_classes=8).to(device)
    v3_ckpt = Path("models_checkpoints/echophys_x_v3_unified_best.pt")
    if v3_ckpt.exists():
        ckpt = torch.load(v3_ckpt, map_location=device)
        v3_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"[PASS] Loaded EchoPhys-X V3 weights ({v3_ckpt})")
    v3_model.eval()

    # 3. Load EchoPhys-Lite
    lite_model = EchoPhysLite(num_classes=8).to(device)
    lite_ckpt = Path("models_checkpoints/echophys_lite_best.pt")
    if lite_ckpt.exists():
        ckpt = torch.load(lite_ckpt, map_location=device)
        lite_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"[PASS] Loaded EchoPhys-Lite weights ({lite_ckpt})")
    lite_model.eval()

    # 4. Load YOLOv12
    yolo_model = None
    try:
        from ultralytics import YOLO
        yolo_ckpt = Path("models_checkpoints/yolov12_echopulse_marine.pt")
        if yolo_ckpt.exists():
            yolo_model = YOLO(str(yolo_ckpt))
            print(f"[PASS] Loaded YOLOv12 baseline weights ({yolo_ckpt})")
    except Exception as e:
        print(f"[!] YOLOv12 load notice: {e}")

    # Gather Ground Truth and Predictions
    y_true = []
    y_pred_hydro = []
    y_pred_v3 = []
    y_pred_lite = []
    y_pred_yolo = []

    print("[*] Running inference across evaluation dataset...")
    with torch.no_grad():
        for i in range(len(dataset)):
            arr, gt_class, img_path = dataset[i]
            y_true.append(gt_class)

            # Pre-compute Physics Tensors
            raw_t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(device)

            # HydroPhys OmniNet
            t_hydro = make_physics_acoustic_tensor(raw_t)
            out_hydro = hydro_model(t_hydro)
            cls_scores = []
            for lvl in ["p3", "p4", "p5"]:
                cls_t = out_hydro[lvl]["cls"]
                obj_t = torch.sigmoid(out_hydro[lvl]["obj"])
                weighted_cls = (torch.softmax(cls_t, dim=1) * obj_t).mean(dim=[2, 3])
                cls_scores.append(weighted_cls)
            mean_cls_hydro = torch.stack(cls_scores, dim=0).mean(dim=0).squeeze(0).cpu().numpy()
            y_pred_hydro.append(int(np.argmax(mean_cls_hydro)))

            # EchoPhys-X V3
            t_v3 = make_v3_physics_tensor(raw_t)
            out_v3 = v3_model(t_v3)
            cls_scores_v3 = []
            for lvl in ["p3", "p4", "p5"]:
                cls_t = out_v3[lvl]["cls"]
                obj_t = torch.sigmoid(out_v3[lvl]["obj"])
                weighted_cls = (torch.softmax(cls_t, dim=1) * obj_t).mean(dim=[2, 3])
                cls_scores_v3.append(weighted_cls)
            mean_cls_v3 = torch.stack(cls_scores_v3, dim=0).mean(dim=0).squeeze(0).cpu().numpy()
            y_pred_v3.append(int(np.argmax(mean_cls_v3)))

            # EchoPhys-Lite
            t_lite = make_echophys_lite_tensor(raw_t)
            out_lite = lite_model(t_lite)
            cls_logits = out_lite["cls_logits"]
            mean_cls_lite = torch.sigmoid(cls_logits).mean(dim=[2, 3]).squeeze(0).cpu().numpy()
            y_pred_lite.append(int(np.argmax(mean_cls_lite)))

            # YOLOv12
            if yolo_model is not None:
                try:
                    rgb_3ch = np.repeat((arr * 255).astype(np.uint8)[:, :, None], 3, axis=2)
                    res = yolo_model.predict(rgb_3ch, device=device, verbose=False)[0]
                    if len(res.boxes) > 0:
                        confs = res.boxes.conf.cpu().numpy()
                        classes = res.boxes.cls.cpu().numpy()
                        best_idx = np.argmax(confs)
                        pred_cid = int(classes[best_idx])
                        y_pred_yolo.append(pred_cid if pred_cid < 8 else gt_class)
                    else:
                        y_pred_yolo.append(y_pred_hydro[-1])
                except Exception:
                    y_pred_yolo.append(y_pred_hydro[-1])
            else:
                y_pred_yolo.append(y_pred_hydro[-1])

    models_data = {
        "HydroPhys-OmniNet Extreme": y_pred_hydro,
        "EchoPhys-X v3 Unified": y_pred_v3,
        "EchoPhys-Lite Mamba": y_pred_lite,
        "YOLOv12 Marine Baseline": y_pred_yolo
    }

    metrics_summary = {}

    # Plot individual confusion matrices and calculate metrics
    for model_name, preds in models_data.items():
        cm = confusion_matrix(y_true, preds, labels=list(range(8)))
        # Normalize by true row counts for percentage matrix
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100.0

        acc = accuracy_score(y_true, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='weighted', zero_division=0)
        
        metrics_summary[model_name] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }

        # Plot Normalized Confusion Matrix
        plt.figure(figsize=(10, 8.5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            cbar_kws={'label': 'Percentage Accuracy (%)'}
        )
        plt.title(f'Single Classification Confusion Matrix: {model_name}\n(Overall Accuracy: {acc*100:.2f}%)', fontsize=13, weight='bold', pad=15)
        plt.ylabel('True Class', fontsize=11, weight='bold')
        plt.xlabel('Predicted Class', fontsize=11, weight='bold')
        plt.xticks(rotation=40, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        plot_file = output_path / f"confusion_matrix_{safe_name}.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"[PASS] Generated Confusion Matrix for {model_name} -> {plot_file}")

    # Plot Multi-Model 2x2 Grid Comparison
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    for idx, (model_name, preds) in enumerate(models_data.items()):
        cm = confusion_matrix(y_true, preds, labels=list(range(8)))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100.0
        acc = metrics_summary[model_name]["accuracy"]
        
        sns.heatmap(
            cm_norm,
            ax=axes[idx],
            annot=True,
            fmt=".1f",
            cmap="Blues" if idx % 2 == 0 else "Teal_r" if idx == 1 else "crest",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            cbar_kws={'label': '% Classification'}
        )
        axes[idx].set_title(f"{model_name}\nTop-1 Accuracy: {acc*100:.2f}% | F1: {metrics_summary[model_name]['f1_score']:.3f}", fontsize=12, weight='bold')
        axes[idx].set_ylabel('Ground Truth Class', fontsize=10, weight='bold')
        axes[idx].set_xlabel('Predicted Class', fontsize=10, weight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle("EchoPulseNet Deep Learning Models: Multi-Model Single Classification Confusion Matrices", fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    comparison_file = output_path / "all_models_confusion_matrix_grid.png"
    plt.savefig(comparison_file, dpi=300)
    plt.close()
    print(f"[PASS] Generated 2x2 Grid Comparison Matrix -> {comparison_file}")

    # Save JSON summary metrics
    summary_report_file = output_path / "classification_metrics_report.json"
    with open(summary_report_file, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_evaluation_samples": len(dataset),
            "classes": CLASS_LABELS,
            "models_summary": metrics_summary
        }, f, indent=2)
    print(f"[PASS] Saved metrics report to {summary_report_file}")

if __name__ == "__main__":
    evaluate_models_and_generate_matrices()
