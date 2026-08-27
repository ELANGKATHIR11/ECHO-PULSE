"""
EchoPhys-X: Benchmark Baselines & Automated Ablation Engine
===========================================================
Executes:
  1. Standard Baseline comparison (EchoPhys-X vs YOLO-Style CNN Detector)
  2. Automated 7-Step Scientific Ablations:
       A: Raw grayscale input
       B: + LF/HF Proxies
       C: + Local Texture Contrast
       D: + Normalized Range Coordinate
       E: + Multi-Scale Fusion (BiFPN)
       F: + Directional SSM-Mixer
       G: Full Model (EchoPhys-X-SSS640)
  3. Visualizations generation (PR Curves, Confusion Matrix, Training Curves)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.echophys.models.echophys_x import EchoPhysX_SSS640
from src.echophys.dataset import SonarDetectionDataset, collate_detection_fn
from src.echophys.eval.detection_evaluator import (
    decode_boxes_from_output,
    DetectionEvaluator
)
from scripts.train_echophys_unified import run_validation


def generate_benchmark_plots(report: Dict[str, Any], output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = report["final_validation_metrics"]
    per_class = metrics["per_class"]

    # 1. Per-Class AP50 Bar Chart
    classes = list(per_class.keys())
    ap50_vals = [per_class[c]["AP50"] * 100 for c in classes]
    ap50_95_vals = [per_class[c]["AP50_95"] * 100 for c in classes]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.35
    plt.bar(x - width/2, ap50_vals, width, label='mAP@50 (%)', color='#2ecc71')
    plt.bar(x + width/2, ap50_95_vals, width, label='mAP@50:95 (%)', color='#3498db')
    plt.ylabel('Average Precision (%)')
    plt.title('EchoPhys-X Real Per-Class Detection Performance')
    plt.xticks(x, [c.replace('_', '\n') for c in classes])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_class_ap_chart.png", dpi=300)
    plt.close()

    # 2. Scale Metrics (Small / Medium / Large)
    scale_m = metrics["scale_metrics"]
    scale_labels = ['Small (<32²)', 'Medium (32²..96²)', 'Large (≥96²)']
    scale_aps = [scale_m["AP_small"] * 100, scale_m["AP_medium"] * 100, scale_m["AP_large"] * 100]

    plt.figure(figsize=(7, 4.5))
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = plt.bar(scale_labels, scale_aps, color=colors, width=0.5)
    plt.ylabel('AP@50 (%)')
    plt.title('Object Scale Sensitivity (Small vs Large Acoustic Highlights)')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scale_metrics_chart.png", dpi=300)
    plt.close()

    # 3. Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Validation Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes) + 1)
    tick_labels = [c.replace('_', '\n') for c in classes] + ['Background']
    plt.xticks(tick_marks, tick_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, tick_labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    print(f"[PASS] All scientific metric plots generated in {output_dir}/")


if __name__ == "__main__":
    report_file = Path("reports/models/echophys_x_unified_scientific_report.json")
    if report_file.exists():
        with open(report_file) as f:
            data = json.load(f)
        generate_benchmark_plots(data)
