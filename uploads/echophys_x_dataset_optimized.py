from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# EchoPhys-X Dataset-Optimized V2
# Target dataset:
#   640x640 grayscale SSS imagery
#   4 YOLO classes (IDs 0..3)
#   402 train / 110 validation images
#   677 / 172 objects
#   single-frequency grayscale only
#
# Design choices specifically matched to this dataset:
#   - 5-channel SSS representation (raw + LF proxy + HF proxy + local contrast + range)
#   - P3/P4/P5 multi-scale detector (stride 8/16/32)
#   - explicit small-object head at 80x80 for 640px inputs
#   - anchor-free objectness + class + LTRB regression
#   - no quadratic attention
#   - lightweight depthwise-separable blocks
#   - geometry/range proxy retained, but NO claim of measured LF/HF physics
#   - vertical flipping disabled because range direction should not be inverted
#   - mild intensity/noise augmentation rather than aggressive image transforms
# ============================================================

IMG_SIZE = 640
NUM_CLASSES = 4
IN_CHANNELS = 5
MEAN = 0.5
STD = 0.25


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def blur_np(im: np.ndarray, radius: float = 2.0) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(im * 255.0, 0, 255)))
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0


def make_sss_channels(im: np.ndarray) -> np.ndarray:
    """Create deterministic SSS-adapted channels from one grayscale frame.

    Channels:
      0 raw calibrated intensity
      1 low-frequency/base-response proxy
      2 high-frequency residual proxy
      3 local contrast / texture proxy
      4 normalized range-coordinate proxy
    """
    lf = blur_np(im, 2.2)
    hf = np.clip(im - lf + 0.5, 0.0, 1.0)
    local = np.abs(im - blur_np(im, 5.0))
    local = np.clip(local * 3.0, 0.0, 1.0)
    range_coord = np.repeat(np.linspace(0.0, 1.0, im.shape[1], dtype=np.float32)[None, :], im.shape[0], axis=0)
    x = np.stack([im, lf, hf, local, range_coord], axis=0)
    return x.astype(np.float32)


class SSSDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, train: bool = False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.train = train
        images = sorted(image_dir.glob("*.jpg"))
        self.items = []
        for img in images:
            label = label_dir / f"{img.stem}.txt"
            if label.exists():
                self.items.append((img, label))
        if not self.items:
            raise RuntimeError(f"No labeled images found in {image_dir}")

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _read_labels(path: Path) -> np.ndarray:
        rows = []
        txt = path.read_text().strip()
        if txt:
            for line in txt.splitlines():
                z = line.split()
                if len(z) >= 5:
                    c = int(float(z[0]))
                    cx, cy, w, h = map(float, z[1:5])
                    if 0 <= c < NUM_CLASSES and w > 0 and h > 0:
                        rows.append([c, cx, cy, w, h])
        return np.asarray(rows, np.float32).reshape(-1, 5)

    def __getitem__(self, idx: int):
        img_path, label_path = self.items[idx]
        im = np.asarray(Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE)), np.float32) / 255.0

        if self.train:
            # SSS-safe augmentation: do not flip the range axis or rotate heavily.
            gain = np.random.uniform(0.92, 1.08)
            bias = np.random.uniform(-0.04, 0.04)
            im = np.clip(im * gain + bias, 0, 1)
            if np.random.rand() < 0.30:
                im = np.clip(im + np.random.normal(0, 0.018, im.shape).astype(np.float32), 0, 1)
            if np.random.rand() < 0.20:
                # Local attenuation / ping dropout simulation.
                y0 = np.random.randint(0, IMG_SIZE - 32)
                y1 = min(IMG_SIZE, y0 + np.random.randint(8, 32))
                im[y0:y1] *= np.random.uniform(0.80, 0.97)

        x = make_sss_channels(im)
        labels = self._read_labels(label_path)
        return torch.from_numpy(x), torch.from_numpy(labels), str(img_path)


def collate(batch):
    xs, ys, paths = zip(*batch)
    return torch.stack(xs), list(ys), list(paths)


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DSConv(nn.Module):
    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, 3, s, groups=cin)
        self.pw = ConvBNAct(cin, cout, 1, 1)
    def forward(self, x):
        return self.pw(self.dw(x))


class ResidualDS(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = DSConv(c, c)
        self.b = DSConv(c, c)
    def forward(self, x):
        return x + self.b(self.a(x))


class LiteDirectionalMixer(nn.Module):
    """Linear-cost directional context mixer; not marketed as official Mamba."""
    def __init__(self, c):
        super().__init__()
        self.row = nn.Conv2d(c, c, (1, 7), padding=(0, 3), groups=c, bias=False)
        self.col = nn.Conv2d(c, c, (7, 1), padding=(3, 0), groups=c, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.out = nn.Conv2d(c, c, 1, bias=False)
    def forward(self, x):
        r = self.row(x)
        c = self.col(x)
        g = self.gate(x)
        return x + self.out(g * r + (1 - g) * c)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(ConvBNAct(IN_CHANNELS, 32, 3, 2), ResidualDS(32))   # 320
        self.s2 = nn.Sequential(ConvBNAct(32, 64, 3, 2), ResidualDS(64), ResidualDS(64))  # 160
        self.s3 = nn.Sequential(ConvBNAct(64, 96, 3, 2), ResidualDS(96), LiteDirectionalMixer(96)) # 80
        self.s4 = nn.Sequential(ConvBNAct(96, 160, 3, 2), ResidualDS(160), LiteDirectionalMixer(160)) # 40
        self.s5 = nn.Sequential(ConvBNAct(160, 224, 3, 2), ResidualDS(224), LiteDirectionalMixer(224)) # 20
    def forward(self, x):
        x = self.s1(x); p2 = self.s2(x); p3 = self.s3(p2); p4 = self.s4(p3); p5 = self.s5(p4)
        return p3, p4, p5


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.p5 = ConvBNAct(224, 128, 1)
        self.p4 = ConvBNAct(160, 128, 1)
        self.p3 = ConvBNAct(96, 128, 1)
        self.ref4 = DSConv(256, 128)
        self.ref3 = DSConv(256, 128)
    def forward(self, p3, p4, p5):
        q5 = self.p5(p5)
        q4 = self.ref4(torch.cat([self.p4(p4), F.interpolate(q5, scale_factor=2, mode="nearest")], dim=1))
        q3 = self.ref3(torch.cat([self.p3(p3), F.interpolate(q4, scale_factor=2, mode="nearest")], dim=1))
        return q3, q4, q5


class Head(nn.Module):
    def __init__(self, c=128, k=NUM_CLASSES):
        super().__init__()
        self.stem = DSConv(c, c)
        self.obj = nn.Conv2d(c, 1, 1)
        self.cls = nn.Conv2d(c, k, 1)
        self.box = nn.Conv2d(c, 4, 1)
    def forward(self, x):
        x = self.stem(x)
        return self.obj(x), self.cls(x), F.softplus(self.box(x))


class EchoPhysXDatasetOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.fpn = FPN()
        self.h3 = Head(); self.h4 = Head(); self.h5 = Head()
    def forward(self, x):
        p3,p4,p5=self.backbone(x)
        f3,f4,f5=self.fpn(p3,p4,p5)
        return {"p3":self.h3(f3),"p4":self.h4(f4),"p5":self.h5(f5)}


def build_targets(labels: List[torch.Tensor], level_shapes: List[Tuple[int,int]], device):
    targets=[]
    strides=[8,16,32]
    for (H,W), stride in zip(level_shapes,strides):
        obj=[]; cls=[]; box=[]; mask=[]
        for labs in labels:
            o=torch.zeros(1,H,W,device=device)
            c=torch.zeros(NUM_CLASSES,H,W,device=device)
            b=torch.zeros(4,H,W,device=device)
            m=torch.zeros(1,H,W,dtype=torch.bool,device=device)
            if len(labs):
                for row in labs.tolist():
                    cc,cx,cy,w,h=row
                    # Assign by object scale; P3 receives small objects.
                    area=w*h
                    if stride==8 and area>0.08: continue
                    if stride==16 and not (0.01<=area<=0.20): continue
                    if stride==32 and area<0.04: continue
                    gx=min(W-1,max(0,int(cx*W))); gy=min(H-1,max(0,int(cy*H)))
                    o[0,gy,gx]=1.0; c[int(cc),gy,gx]=1.0; m[0,gy,gx]=True
                    fx=cx*W-gx; fy=cy*H-gy
                    b[:,gy,gx]=torch.tensor([fx-w*W/2,fy-h*H/2,w*W-fx,h*H-fy],device=device)
            obj.append(o); cls.append(c); box.append(b); mask.append(m)
        targets.append((torch.stack(obj),torch.stack(cls),torch.stack(box),torch.stack(mask)))
    return targets


def focal_bce(logits, target, gamma=2.0, alpha=0.25):
    p=torch.sigmoid(logits); ce=F.binary_cross_entropy_with_logits(logits,target,reduction="none")
    pt=p*target+(1-p)*(1-target); at=alpha*target+(1-alpha)*(1-target)
    return (at*(1-pt).pow(gamma)*ce).mean()


def loss_fn(outputs, labels, device):
    levels=[outputs["p3"],outputs["p4"],outputs["p5"]]
    shapes=[(x[0].shape[-2],x[0].shape[-1]) for x in levels]
    tgts=build_targets(labels,shapes,device)
    total=0.0
    parts={"obj":0.0,"cls":0.0,"box":0.0}
    for out,(o,c,b,m) in zip(levels,tgts):
        po,pc,pb=out
        lo=focal_bce(po,o)
        lc=focal_bce(pc,c)
        if m.any():
            lb=F.smooth_l1_loss(pb[m.expand_as(pb)],b[m.expand_as(b)])
        else: lb=pb.sum()*0
        total=total+lo+lc+2.0*lb
        parts["obj"]+=float(lo.detach()); parts["cls"]+=float(lc.detach()); parts["box"]+=float(lb.detach())
    return total,parts


def sanity(root: Path, batch=2, device=None):
    device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=SSSDataset(root/"train/images",root/"train/labels",False)
    xb, labs, _=collate([ds[0],ds[1]])
    model=EchoPhysXDatasetOptimized().to(device)
    xb=xb.to(device)
    model.eval()
    with torch.no_grad():
        t0=time.perf_counter(); out=model(xb)
        if device.type=="cuda": torch.cuda.synchronize()
        ms=(time.perf_counter()-t0)*1000/batch
    loss,_=loss_fn(out,labs,device)
    return {
      "parameters":sum(p.numel() for p in model.parameters()),
      "input":list(xb.shape),
      "outputs":{k:[list(v[0].shape),list(v[1].shape),list(v[2].shape)] for k,v in out.items()},
      "finite":bool(torch.isfinite(loss)),"sanity_loss":float(loss.detach()),"latency_ms_batch_avg":ms,
      "device":str(device),"train_images":len(ds)
    }


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args(); seed_everything(42)
    print(json.dumps(sanity(args.root),indent=2))
