"""
EchoPhys-X: Center-Region Small Object Target Assigner & Multi-Scale Loss (Optimized)
=====================================================================================
Optimized for zero memory allocations in autograd computation graph.
"""

import math
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = alpha_factor * (1.0 - p_t).pow(gamma)
    return (focal_weight * ce).mean()


def compute_ciou_loss(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    w_p = pred_ltrb[:, 0] + pred_ltrb[:, 2]
    h_p = pred_ltrb[:, 1] + pred_ltrb[:, 3]
    w_g = target_ltrb[:, 0] + target_ltrb[:, 2]
    h_g = target_ltrb[:, 1] + target_ltrb[:, 3]

    area_p = torch.clamp(w_p * h_p, min=1e-6)
    area_g = torch.clamp(w_g * h_g, min=1e-6)

    inter_w = torch.clamp(torch.min(pred_ltrb[:, 0], target_ltrb[:, 0]) + torch.min(pred_ltrb[:, 2], target_ltrb[:, 2]), min=0.0)
    inter_h = torch.clamp(torch.min(pred_ltrb[:, 1], target_ltrb[:, 1]) + torch.min(pred_ltrb[:, 3], target_ltrb[:, 3]), min=0.0)
    inter_area = inter_w * inter_h
    union_area = area_p + area_g - inter_area
    iou = torch.clamp(inter_area / (union_area + 1e-7), min=0.0, max=1.0)

    # Enclosing box
    enc_w = torch.max(pred_ltrb[:, 0], target_ltrb[:, 0]) + torch.max(pred_ltrb[:, 2], target_ltrb[:, 2])
    enc_h = torch.max(pred_ltrb[:, 1], target_ltrb[:, 1]) + torch.max(pred_ltrb[:, 3], target_ltrb[:, 3])
    c2 = enc_w ** 2 + enc_h ** 2 + 1e-7

    # Center distance
    rho2 = (pred_ltrb[:, 0] - target_ltrb[:, 0]) ** 2 + (pred_ltrb[:, 1] - target_ltrb[:, 1]) ** 2
    
    # Aspect ratio consistency
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w_g / (h_g + 1e-6)) - torch.atan(w_p / (h_p + 1e-6)), 2)
    with torch.no_grad():
        alpha_ciou = v / ((1.0 - iou) + v + 1e-7)

    ciou = iou - (rho2 / c2) - alpha_ciou * v
    return torch.mean(1.0 - ciou)


@torch.no_grad()
def assign_center_region_targets(
    labels_batch: List[torch.Tensor],
    level_shapes: List[Tuple[int, int]],
    num_classes: int,
    device: torch.device
):
    """
    Constructs ground truth detection target tensors on CPU/GPU without accumulating autograd graph.
    """
    strides = [8, 16, 32]
    batch_size = len(labels_batch)
    targets = []

    for (H, W), stride in zip(level_shapes, strides):
        obj = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.float32)
        cls_target = torch.zeros((batch_size, num_classes, H, W), device=device, dtype=torch.float32)
        box_target = torch.zeros((batch_size, 4, H, W), device=device, dtype=torch.float32)
        mask = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.bool)

        for b_idx, labs in enumerate(labels_batch):
            if labs is None or len(labs) == 0:
                continue

            for row in labs.tolist():
                c, cx, cy, w, h = row[:5]
                c_idx = int(c)
                if c_idx >= num_classes or w <= 0 or h <= 0:
                    continue

                area = w * h
                if stride == 8 and area > 0.12:
                    continue
                if stride == 16 and not (0.015 <= area <= 0.30):
                    continue
                if stride == 32 and area < 0.05:
                    continue

                gx = min(W - 1, max(0, int(cx * W)))
                gy = min(H - 1, max(0, int(cy * H)))

                obj[b_idx, 0, gy, gx] = 1.0
                cls_target[b_idx, c_idx, gy, gx] = 1.0
                mask[b_idx, 0, gy, gx] = True

                l = (gx + 0.5) - (cx - w / 2.0) * W
                t = (gy + 0.5) - (cy - h / 2.0) * H
                r = (cx + w / 2.0) * W - (gx + 0.5)
                btm = (cy + h / 2.0) * H - (gy + 0.5)

                box_target[b_idx, :, gy, gx] = torch.tensor(
                    [max(0.1, l), max(0.1, t), max(0.1, r), max(0.1, btm)],
                    device=device,
                    dtype=torch.float32
                )

                # Center region neighbor cell activation (multi-positive assignment)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        obj[b_idx, 0, ny, nx] = 1.0
                        cls_target[b_idx, c_idx, ny, nx] = 1.0
                        mask[b_idx, 0, ny, nx] = True
                        nl = (nx + 0.5) - (cx - w / 2.0) * W
                        nt = (ny + 0.5) - (cy - h / 2.0) * H
                        nr = (cx + w / 2.0) * W - (nx + 0.5)
                        nbtm = (cy + h / 2.0) * H - (ny + 0.5)
                        box_target[b_idx, :, ny, nx] = torch.tensor(
                            [max(0.1, nl), max(0.1, nt), max(0.1, nr), max(0.1, nbtm)],
                            device=device,
                            dtype=torch.float32
                        )

        targets.append((obj, cls_target, box_target, mask))

    return targets


class EchoPhysLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_box: float = 2.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box

    def forward(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        labels: List[torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        levels = [outputs["p3"], outputs["p4"], outputs["p5"]]
        shapes = [(lvl["obj"].shape[-2], lvl["obj"].shape[-1]) for lvl in levels]
        scale_weights = [1.5, 1.0, 0.8]

        targets = assign_center_region_targets(labels, shapes, self.num_classes, device)

        total_loss = torch.tensor(0.0, device=device)
        breakdown = {"obj": 0.0, "cls": 0.0, "box": 0.0}

        for lvl, (tgt_obj, tgt_cls, tgt_box, mask), w_s in zip(levels, targets, scale_weights):
            p_obj = lvl["obj"]
            p_cls = lvl["cls"]
            p_box = lvl["box"]

            l_obj = focal_bce_loss(p_obj, tgt_obj)
            l_cls = focal_bce_loss(p_cls, tgt_cls)

            if mask.any():
                pred_matched_box = p_box.permute(0, 2, 3, 1)[mask.squeeze(1)]
                tgt_matched_box = tgt_box.permute(0, 2, 3, 1)[mask.squeeze(1)]
                l_box = compute_ciou_loss(pred_matched_box, tgt_matched_box)
            else:
                l_box = p_box.sum() * 0.0

            lvl_loss = self.lambda_obj * l_obj + self.lambda_cls * l_cls + (self.lambda_box * w_s) * l_box
            total_loss = total_loss + lvl_loss

            breakdown["obj"] += float(l_obj.detach())
            breakdown["cls"] += float(l_cls.detach())
            breakdown["box"] += float(l_box.detach())

        return total_loss, breakdown
