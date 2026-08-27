"""
EchoPhys-X: Multi-Dataset Ingestion & Scientific Sonar Dataset (Zero-RAM Leak)
=============================================================================
Optimized with PIL -> uint8 torch.from_numpy -> GPU conversion to eliminate
Windows numpy float32 array allocations and system memory fragmentation.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SonarDetectionDataset(Dataset):
    """
    Standardized Zero-Overhead PyTorch Dataset for Marine Sonar Object Detection.
    """
    def __init__(
        self,
        data_pairs: List[Tuple[Path, Path]],
        num_classes: int = 4,
        img_size: int = 640,
        is_train: bool = False,
        class_mapping: Optional[Dict[int, int]] = None
    ):
        self.num_classes = num_classes
        self.img_size = img_size
        self.is_train = is_train
        self.class_mapping = class_mapping or {}
        self.samples = []

        for img_dir, lbl_dir in data_pairs:
            img_dir = Path(img_dir)
            lbl_dir = Path(lbl_dir)
            if not img_dir.exists() or not lbl_dir.exists():
                continue

            images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
            for img in images:
                label_p = lbl_dir / f"{img.stem}.txt"
                if label_p.exists():
                    self.samples.append((img, label_p))

        if not self.samples:
            raise RuntimeError(f"No valid labeled images found in dataset pairs: {data_pairs}")

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_labels(self, label_path: Path) -> np.ndarray:
        rows = []
        txt = label_path.read_text().strip()
        if txt:
            for line in txt.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        raw_c = int(float(parts[0]))
                        c = self.class_mapping.get(raw_c, raw_c)
                        cx, cy, w, h = map(float, parts[1:5])

                        # Bounding box sanity checks
                        if 0 <= c < self.num_classes and 0.0 < cx < 1.0 and 0.0 < cy < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                            rows.append([c, cx, cy, w, h])
                    except ValueError:
                        continue
        return np.asarray(rows, dtype=np.float32).reshape(-1, 5)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, lbl_path = self.samples[idx]

        with Image.open(img_path) as pil_im:
            im_resized = pil_im.convert("L").resize((self.img_size, self.img_size))
            im_u8 = np.array(im_resized, dtype=np.uint8)

        # Zero-copy uint8 -> float32 tensor conversion in PyTorch directly
        im_t = torch.from_numpy(im_u8).float().div_(255.0).unsqueeze(0) # (1, H, W)

        # Acoustic Speckle & Gain Data Augmentation for training
        if self.is_train:
            gain = float(np.random.uniform(0.92, 1.08))
            bias = float(np.random.uniform(-0.04, 0.04))
            im_t = torch.clamp(im_t.mul_(gain).add_(bias), 0.0, 1.0)

            if np.random.rand() < 0.20:
                y0 = int(np.random.randint(0, self.img_size - 32))
                y1 = min(self.img_size, y0 + int(np.random.randint(8, 28)))
                im_t[:, y0:y1, :] *= float(np.random.uniform(0.70, 0.95))

        labels = self._parse_labels(lbl_path)
        return im_t, torch.from_numpy(labels), str(img_path)


def collate_detection_fn(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images), list(labels), list(paths)
