#!/usr/bin/env python3
"""COVA-MTL v6.3 on StateFarm Distracted Driver Detection (five seeds).

StateFarm is a single-frame, single-task, 10-class dataset.  The four selectable
semantic visual branches are exactly the pseudo-regions used by the supplied
StateFarm baseline: full, upper, lower-left, and lower-right.  They are views
derived from one decoded image, not four independent physical sensors.

Pipeline per model seed
-----------------------
teacher -> cache 15 subsets -> ACVE -> calibration -> sparse SCML student
        -> student recalibration -> dense/dynamic/failure evaluation

The 64/16/20 driver-group split is created once with --split-seed and shared by
all model seeds.  If driver_imgs_list.csv is unavailable, the code explicitly
falls back to a stratified image split and records that weaker protocol.

Expected layout
---------------
STATEFARM_ROOT/
    driver_imgs_list.csv
    imgs/train/c0/*.jpg ... c9/*.jpg

Also supported: train/c0...c9 or directly c0...c9.

Example
-------
python cross_dataset/statefarm.py \
  --data-root /path/to/StateFarm \
  --run-dir /path/to/RUNS/STATEFARM_COVA_v6_3_MultiSeed5 \
  --stage all

Resume is enabled by default.  Only last.pt and best*.pt are retained.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

try:
    from torch.amp import GradScaler, autocast
    AMP_STYLE = "torch.amp"
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    AMP_STYLE = "torch.cuda.amp"


MODALITY_NAMES = ("full", "upper", "lower_left", "lower_right")
TASK_NAMES = ("DIR",)
TASK_CLASSES = (10,)
STATEFARM_CLASSES = tuple(f"c{i}" for i in range(10))
CLASS_TO_IDX = {name: index for index, name in enumerate(STATEFARM_CLASSES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
CLASS_DESCRIPTIONS = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "talking on the phone - right",
    "c3": "texting - left",
    "c4": "talking on the phone - left",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
M = len(MODALITY_NAMES)
K = len(TASK_NAMES)
NUM_SUBSETS = (1 << M) - 1


# -----------------------------------------------------------------------------
# Reproducibility, I/O, AMP
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def amp_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if AMP_STYLE == "torch.amp":
        return autocast(device_type="cuda", enabled=True)
    return autocast(enabled=True)


def make_scaler(device: torch.device):
    if AMP_STYLE == "torch.amp":
        return GradScaler("cuda", enabled=device.type == "cuda")
    return GradScaler(enabled=device.type == "cuda")


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer=None, scaler=None,
                    best: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "model": model.state_dict(), "best": best or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scaler=None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def completed_stage(stage_dir: Path, best_path: Path, resume: bool) -> bool:
    marker = stage_dir / "complete.json"
    if resume and marker.exists() and best_path.exists():
        print(f"stage already complete: {stage_dir}")
        return True
    return False


def finalize_stage(stage_dir: Path, best_path: Path, payload: dict) -> None:
    if not best_path.exists():
        raise FileNotFoundError(f"stage ended without a best checkpoint: {best_path}")
    last_path = stage_dir / "last.pt"
    if last_path.exists():
        last_path.unlink()
    write_json(stage_dir / "complete.json", {
        **payload,
        "best_checkpoint": best_path,
        "checkpoint_policy": (
            "last.pt is retained only while a stage is incomplete; after success, "
            "only the compact best checkpoint remains"
        ),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# -----------------------------------------------------------------------------
# StateFarm indexing, driver-disjoint split, and semantic region views
# -----------------------------------------------------------------------------
def candidate_image_roots(root: Path) -> List[Path]:
    return [root / "imgs" / "train", root / "train", root]


def resolve_image_path(root: Path, class_name: str, image_name: str) -> Path:
    for base in candidate_image_roots(root):
        path = base / class_name / image_name
        if path.exists():
            return path.resolve()
    return (root / "imgs" / "train" / class_name / image_name).resolve()


def collect_image_df(data_root: Path) -> Tuple[pd.DataFrame, str]:
    csv_path = data_root / "driver_imgs_list.csv"
    rows: List[dict] = []
    if csv_path.exists():
        metadata = pd.read_csv(csv_path)
        required = {"subject", "classname", "img"}
        if not required.issubset(metadata.columns):
            raise ValueError(
                f"{csv_path} must contain {sorted(required)}, got {list(metadata.columns)}"
            )
        for row in metadata.itertuples(index=False):
            class_name = str(row.classname)
            if class_name not in CLASS_TO_IDX:
                continue
            image_path = resolve_image_path(data_root, class_name, str(row.img))
            if image_path.exists():
                rows.append({
                    "image_path": str(image_path),
                    "class_name": class_name,
                    "class_description": CLASS_DESCRIPTIONS[class_name],
                    "label": CLASS_TO_IDX[class_name],
                    "subject": str(row.subject),
                    "image_name": str(row.img),
                })
        split_mode = "driver_group"
    else:
        print(
            f"[WARN] {csv_path} is absent. Falling back to a stratified image split; "
            "this is not driver-disjoint."
        )
        base = next(
            (candidate for candidate in candidate_image_roots(data_root)
             if all((candidate / name).exists() for name in STATEFARM_CLASSES)),
            None,
        )
        if base is not None:
            for class_name in STATEFARM_CLASSES:
                for image_path in sorted((base / class_name).rglob("*")):
                    if image_path.is_file() and image_path.suffix.lower() in VALID_EXTS:
                        rows.append({
                            "image_path": str(image_path.resolve()),
                            "class_name": class_name,
                            "class_description": CLASS_DESCRIPTIONS[class_name],
                            "label": CLASS_TO_IDX[class_name],
                            "subject": "unknown",
                            "image_name": image_path.name,
                        })
        split_mode = "stratified_image"
    if not rows:
        raise RuntimeError(
            f"No StateFarm images found under {data_root}. Expected driver_imgs_list.csv "
            "with imgs/train/c0..c9, train/c0..c9, or c0..c9."
        )
    return pd.DataFrame(rows).reset_index(drop=True), split_mode


def make_split_frames(frame: pd.DataFrame, split_mode: str, seed: int):
    if split_mode == "driver_group":
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        trainval_index, test_index = next(
            outer.split(frame, frame["label"], groups=frame["subject"])
        )
        trainval = frame.iloc[trainval_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_index, val_index = next(
            inner.split(trainval, trainval["label"], groups=trainval["subject"])
        )
        train = trainval.iloc[train_index].reset_index(drop=True)
        val = trainval.iloc[val_index].reset_index(drop=True)
    else:
        trainval, test = train_test_split(
            frame, test_size=0.20, random_state=seed, shuffle=True,
            stratify=frame["label"],
        )
        train, val = train_test_split(
            trainval, test_size=0.20, random_state=seed, shuffle=True,
            stratify=trainval["label"],
        )
        train, val, test = (
            train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True),
        )
    for name, part in (("train", train), ("val", val), ("test", test)):
        missing = sorted(set(range(TASK_CLASSES[0])) - set(part["label"].unique()))
        if missing:
            raise RuntimeError(
                f"split_seed={seed} produced {name} without classes {missing}; "
                "choose another --split-seed."
            )
    return train, val, test


def ensure_splits(args) -> None:
    args.split_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    meta_path = args.split_dir / "split_meta.json"
    if all(path.exists() for path in paths.values()) and not args.force_splits:
        print(f"using shared split: {args.split_dir}")
        return
    frame, split_mode = collect_image_df(args.data_root)
    train, val, test = make_split_frames(frame, split_mode, args.split_seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.reset_index(drop=True).to_csv(paths[name], index=False)
    write_json(meta_path, {
        "split_seed": args.split_seed,
        "split_mode": split_mode,
        "protocol": (
            "driver-disjoint GroupShuffleSplit: approximately train 64%, val 16%, test 20%"
            if split_mode == "driver_group"
            else "stratified image split fallback: train 64%, val 16%, test 20%"
        ),
        "data_root": args.data_root,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "subjects": {
            name: sorted(part["subject"].astype(str).unique().tolist())
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "class_to_idx": CLASS_TO_IDX,
        "class_descriptions": CLASS_DESCRIPTIONS,
    })
    print(f"created shared split: {args.split_dir} | mode={split_mode}")
    args.force_splits = False


def random_crop(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = random.uniform(0.86, 1.0)
    ratio = random.uniform(0.90, 1.10)
    area = h * w * scale
    crop_w = min(w, max(1, int(math.sqrt(area * ratio))))
    crop_h = min(h, max(1, int(math.sqrt(area / ratio))))
    x = random.randint(0, max(w - crop_w, 0))
    y = random.randint(0, max(h - crop_h, 0))
    return cv2.resize(image[y:y + crop_h, x:x + crop_w], (size, size), interpolation=cv2.INTER_AREA)


def color_jitter(image: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32)
    output *= random.uniform(0.85, 1.15)
    output = (output - 127.5) * random.uniform(0.85, 1.15) + 127.5
    return np.clip(output, 0, 255).astype(np.uint8)


def bgr_to_tensor(image: np.ndarray, size: int) -> torch.Tensor:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


class StateFarmCOVADataset(Dataset):
    def __init__(self, frame: pd.DataFrame, args, train: bool):
        self.frame, self.args, self.train = frame.reset_index(drop=True), args, train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        source = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"failed to read {row.image_path}")
        full = random_crop(source, self.args.image_size) if self.train else cv2.resize(
            source, (self.args.image_size, self.args.image_size), interpolation=cv2.INTER_AREA
        )
        if self.train and random.random() < 0.8:
            full = color_jitter(full)
        height, width = full.shape[:2]
        upper = full[:max(1, int(round(0.64 * height))), :]
        lower_left = full[
            min(height - 1, int(round(0.38 * height))):,
            :max(1, int(round(0.62 * width))),
        ]
        lower_right = full[
            min(height - 1, int(round(0.38 * height))):,
            min(width - 1, int(round(0.38 * width))):,
        ]
        views = torch.stack([
            bgr_to_tensor(full, self.args.image_size),
            bgr_to_tensor(upper, self.args.image_size),
            bgr_to_tensor(lower_left, self.args.image_size),
            bgr_to_tensor(lower_right, self.args.image_size),
        ])
        return views, torch.tensor(int(row.label), dtype=torch.long), str(row.image_path)


def build_loaders(args, cache_order: bool = False) -> Dict[str, DataLoader]:
    ensure_splits(args)
    frames = {name: pd.read_csv(args.split_dir / f"{name}.csv") for name in ("train", "val", "test")}
    datasets = {name: StateFarmCOVADataset(frame, args, train=(name == "train" and not cache_order))
                for name, frame in frames.items()}
    loaders = {}
    for offset, name in enumerate(("train", "val", "test")):
        generator = torch.Generator().manual_seed(args.seed * 100 + offset)
        shuffle = name == "train" and not cache_order
        loaders[name] = DataLoader(
            datasets[name], batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker,
            generator=generator if shuffle else None, drop_last=False,
        )
    print("dataset sizes:", {name: len(ds) for name, ds in datasets.items()})
    return loaders


def move_batch(batch, device: torch.device):
    return batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2]


# -----------------------------------------------------------------------------
# SCML: shared visual encoder, subset calibration, and mask-aware fusion
# -----------------------------------------------------------------------------
class IdentityAttention(nn.Module):
    def forward(self, x):
        return x


class DirectionalSpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.v = nn.Conv2d(2, 1, (kernel, 1), padding=(pad, 0), bias=False)
        self.h = nn.Conv2d(2, 1, (1, kernel), padding=(0, pad), bias=False)

    def forward(self, x):
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        return x * torch.sigmoid(self.v(pooled) + self.h(pooled))


class SpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        return x * torch.sigmoid(self.conv(pooled))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden, bias=False), nn.ReLU(),
                                 nn.Linear(hidden, channels, bias=False))

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(x, 1).flatten(1)
        return x * torch.sigmoid(self.mlp(avg) + self.mlp(maximum))[:, :, None, None]


class PSMSBN2d(nn.Module):
    def __init__(self, reference: nn.BatchNorm2d, domains: int = M, mix: float = 0.2):
        super().__init__()
        self.domains, self.channels = domains, reference.num_features
        self.eps, self.momentum = reference.eps, reference.momentum
        self.weight = nn.Parameter(reference.weight.detach()[None].repeat(domains, 1))
        self.bias = nn.Parameter(reference.bias.detach()[None].repeat(domains, 1))
        self.register_buffer("running_mean_dom", reference.running_mean.detach()[None].repeat(domains, 1))
        self.register_buffer("running_var_dom", reference.running_var.detach()[None].repeat(domains, 1))
        self.register_buffer("running_mean_global", reference.running_mean.detach().clone())
        self.register_buffer("running_var_global", reference.running_var.detach().clone())
        self.register_buffer("batches_dom", torch.zeros(domains, dtype=torch.long))
        self.register_buffer("batches_global", torch.zeros((), dtype=torch.long))
        mix = float(np.clip(mix, 1e-4, 1 - 1e-4))
        self.mix_logit = nn.Parameter(torch.full((self.channels,), math.log(mix / (1 - mix))))

    def _factor(self, tracked):
        return 1.0 / float(tracked.item()) if self.momentum is None else float(self.momentum)

    def forward(self, x, domain: int):
        if not 0 <= domain < self.domains:
            raise ValueError(f"invalid modality id {domain}")
        if self.training:
            xf = x.float()
            local_mean, local_var = xf.mean((0, 2, 3)), xf.var((0, 2, 3), unbiased=False)
            global_mean = self.running_mean_global.detach().clone().float()
            global_var = self.running_var_global.detach().clone().float()
            with torch.no_grad():
                self.batches_dom[domain] += 1
                self.batches_global += 1
                self.running_mean_dom[domain].lerp_(local_mean.detach(), self._factor(self.batches_dom[domain]))
                self.running_var_dom[domain].lerp_(local_var.detach(), self._factor(self.batches_dom[domain]))
                self.running_mean_global.lerp_(local_mean.detach(), self._factor(self.batches_global))
                self.running_var_global.lerp_(local_var.detach(), self._factor(self.batches_global))
        else:
            local_mean = self.running_mean_dom[domain].float()
            local_var = self.running_var_dom[domain].float()
            global_mean = self.running_mean_global.float()
            global_var = self.running_var_global.float()
        mix = torch.sigmoid(self.mix_logit).float()
        mean = (1 - mix) * local_mean + mix * global_mean
        var = (1 - mix) * local_var + mix * global_var
        output = (x.float() - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        output = output.to(x.dtype)
        return output * self.weight[domain][None, :, None, None].to(x.dtype) + self.bias[domain][None, :, None, None].to(x.dtype)


class DomainIdentity(nn.Module):
    def forward(self, x, domain=0):
        return x


class SharedResNet18PSMSBN(nn.Module):
    def __init__(self, pretrained=True, train_base=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1, self.relu, self.maxpool = base.conv1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.stem_bn = PSMSBN2d(base.bn1)
        self.attention = nn.ModuleList((IdentityAttention(), DirectionalSpatialAttention(),
                                        SpatialAttention(), ChannelAttention(512)))

        def convert(layer):
            bn1 = nn.ModuleList([PSMSBN2d(block.bn1) for block in layer])
            bn2 = nn.ModuleList([PSMSBN2d(block.bn2) for block in layer])
            down = nn.ModuleList([PSMSBN2d(block.downsample[1]) if block.downsample is not None
                                  else DomainIdentity() for block in layer])
            return bn1, bn2, down
        self.bns = nn.ModuleList()
        converted = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            converted.append(convert(layer))
        self.bn_sets = converted
        # Register converted modules (plain tuple/list is not enough for state_dict).
        for stage, (bn1, bn2, down) in enumerate(converted):
            self.add_module(f"s{stage}_bn1", bn1)
            self.add_module(f"s{stage}_bn2", bn2)
            self.add_module(f"s{stage}_down", down)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if not train_base:
            for module in (self.conv1, self.layer1, self.layer2, self.layer3, self.layer4):
                module.requires_grad_(False)

    @staticmethod
    def _block(block, x, bn1, bn2, down, domain):
        residual = x if block.downsample is None else down(block.downsample[0](x), domain)
        output = block.relu(bn1(block.conv1(x), domain))
        output = bn2(block.conv2(output), domain)
        return block.relu(output + residual)

    def forward(self, x, domain: int):
        x = self.maxpool(self.relu(self.stem_bn(self.conv1(x), domain)))
        for stage, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4)):
            bn1, bn2, down = self.bn_sets[stage]
            for index, block in enumerate(layer):
                x = self._block(block, x, bn1[index], bn2[index], down[index], domain)
            x = self.attention[stage](x)
        return self.pool(x).flatten(1)


class LightweightPreview(nn.Module):
    def __init__(self, embed: int, size: int):
        super().__init__()
        self.size = size
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.ModuleList([nn.Linear(64, embed) for _ in range(M)])
        self.type_embedding = nn.Parameter(torch.randn(M, embed) * 0.02)

    def forward(self, views):
        batch = views.size(0)
        low = F.interpolate(views.flatten(0, 1), size=(self.size, self.size), mode="bilinear", align_corners=False)
        features = self.cnn(low).flatten(1).reshape(batch, M, -1)
        tokens = torch.stack([self.project[m](features[:, m]) for m in range(M)], 1)
        return tokens + self.type_embedding[None]


class SubsetConditionedCalibration(nn.Module):
    def __init__(self, embed: int, hidden: int, strength: float):
        super().__init__()
        self.strength = strength
        self.norm = nn.LayerNorm(embed)
        self.context = nn.Sequential(nn.Linear(embed + M, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.affine = nn.ModuleList([nn.Linear(hidden, 2 * embed) for _ in range(M)])
        for layer in self.affine:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens, scout, mask):
        if (~mask.bool()).all(1).any():
            raise ValueError("SCML requires at least one active branch")
        weight = mask[:, :, None].to(scout.dtype)
        summary = (scout * weight).sum(1) / weight.sum(1).clamp_min(1)
        context = self.context(torch.cat((summary, mask.to(summary.dtype)), 1))
        output = []
        for m in range(M):
            gamma, beta = self.affine[m](context).chunk(2, 1)
            gamma, beta = torch.tanh(gamma), torch.tanh(beta)
            calibrated = tokens[:, m] + self.strength * (
                gamma * self.norm(tokens[:, m]) + beta
            )
            output.append(calibrated * mask[:, m, None].to(calibrated.dtype))
        return torch.stack(output, 1)


class MaskAwareFusion(nn.Module):
    def __init__(self, embed: int, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.project = nn.Linear(embed, d_model)
        self.modality_embedding = nn.Parameter(torch.randn(M, d_model) * 0.02)
        self.task_tokens = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tokens, mask):
        if (~mask.bool()).all(1).any():
            raise ValueError("mask-aware fusion requires at least one active branch")
        key = self.project(tokens) + self.modality_embedding[None]
        query = self.task_tokens[None].expand(tokens.size(0), -1, -1)
        attended, _ = self.cross(query, key, key, key_padding_mask=~mask, need_weights=False)
        state = self.norm1(query + attended)
        return self.norm2(state + self.ffn(state))


class COVAStateFarm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        self.scout = LightweightPreview(args.embed, args.preview_size)
        self.backbone = SharedResNet18PSMSBN(args.pretrained, args.train_base)
        self.visual_project = nn.Sequential(nn.Linear(512, args.embed), nn.GELU(), nn.LayerNorm(args.embed))
        self.scmc = SubsetConditionedCalibration(args.embed, args.scmc_hidden, args.scmc_strength)
        self.fusion = MaskAwareFusion(args.embed, args.d_model, args.attention_heads, args.dropout)
        self.heads = nn.ModuleList([nn.Linear(args.d_model, classes) for classes in TASK_CLASSES])
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def encode_modality(self, views, modality: int):
        image = views[:, modality]
        image = (image - self.image_mean.to(image.dtype)) / self.image_std.to(image.dtype)
        return self.visual_project(self.backbone(image, modality))

    def encode_all(self, views):
        scout = self.scout(views)
        order = torch.randperm(M).tolist() if self.training else list(range(M))
        encoded = [None] * M
        for modality in order:
            encoded[modality] = self.encode_modality(views, modality)
        return scout, torch.stack(encoded, 1)

    def encode_selected(self, mask, views):
        batch = views.size(0)
        streams = []
        for modality in range(M):
            rows = torch.nonzero(mask[:, modality], as_tuple=False).flatten()
            if rows.numel() == 0:
                stream = views.new_zeros(batch, self.embed)
            else:
                selected_views = views.index_select(0, rows)
                values = self.encode_modality(selected_views, modality)
                stream = values.new_zeros(batch, self.embed).index_copy(0, rows, values)
            streams.append(stream)
        return torch.stack(streams, 1)

    def predict(self, scout, tokens, mask):
        tokens = self.scmc(tokens, scout, mask)
        task_state = self.fusion(tokens, mask)
        logits = tuple(head(task_state[:, i]) for i, head in enumerate(self.heads))
        return logits, task_state.mean(1), task_state

    def forward_dense(self, views):
        scout, tokens = self.encode_all(views)
        mask = torch.ones(views.size(0), M, dtype=torch.bool, device=views.device)
        return self.predict(scout, tokens, mask)[0]


# -----------------------------------------------------------------------------
# ACVE and SSR
# -----------------------------------------------------------------------------
def mask_table(device=None):
    return torch.tensor([[bool(s & (1 << m)) for m in range(M)] for s in range(1 << M)],
                        dtype=torch.bool, device=device)


def masks_to_subset_ids(mask):
    weights = torch.tensor([1 << m for m in range(M)], device=mask.device, dtype=torch.long)
    return (mask.long() * weights).sum(1)


def sample_balanced_masks(batch, device, include_full=True):
    output = torch.zeros(batch, M, dtype=torch.bool, device=device)
    maximum = M if include_full else M - 1
    cards = torch.randint(1, maximum + 1, (batch,), device=device)
    order = torch.rand(batch, M, device=device).argsort(1)
    for i in range(batch):
        output[i, order[i, :int(cards[i])]] = True
    return output


class ACVE(nn.Module):
    def __init__(self, scout_dim: int, hidden: int):
        super().__init__()
        token_dim = max(hidden // 2, 32)
        self.token = nn.Sequential(nn.LayerNorm(scout_dim), nn.Linear(scout_dim, token_dim), nn.GELU())
        self.modality_embedding = nn.Parameter(torch.randn(M, token_dim) * 0.02)
        self.availability_embedding = nn.Parameter(torch.randn(2, token_dim) * 0.02)
        candidates = mask_table()[1:]
        self.register_buffer("candidate_masks", candidates, persistent=True)
        self.register_buffer("candidate_cardinality", candidates.sum(1), persistent=True)
        self.register_buffer("calibration_q", torch.ones(M, M), persistent=False)
        feature_dim = M * token_dim + 3 * token_dim + 2 * M + 2
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 2 * K))

    def forward(self, scout, available):
        available = available.bool()
        if (~available).all(1).any():
            raise ValueError("every sample requires at least one available branch")
        encoded = self.token(scout) + self.modality_embedding[None] + self.availability_embedding[available.long()]
        encoded = encoded * available[:, :, None].to(encoded.dtype)
        candidate = self.candidate_masks.to(encoded.dtype)
        av = available.to(encoded.dtype)
        selected_mask = candidate[None] * av[:, None]
        remaining_mask = (1 - candidate)[None] * av[:, None]
        selected = encoded[:, None] * selected_mask[:, :, :, None]
        remaining = encoded[:, None] * remaining_mask[:, :, :, None]
        selected_mean = selected.sum(2) / selected_mask.sum(2, keepdim=True).clamp_min(1)
        remaining_mean = remaining.sum(2) / remaining_mask.sum(2, keepdim=True).clamp_min(1)
        av_mean = (encoded.sum(1) / av.sum(1, keepdim=True).clamp_min(1))[:, None].expand(-1, NUM_SUBSETS, -1)
        cardinality = candidate.sum(1, keepdim=True)
        features = torch.cat((
            selected.flatten(2), selected_mean, remaining_mean, av_mean,
            candidate[None].expand(encoded.size(0), -1, -1),
            av[:, None].expand(-1, NUM_SUBSETS, -1),
            (cardinality / M)[None].expand(encoded.size(0), -1, -1),
            (av.sum(1, keepdim=True) / M)[:, None].expand(-1, NUM_SUBSETS, -1),
        ), -1)
        mean, raw_scale = self.net(features).chunk(2, -1)
        scale = F.softplus(raw_scale.float()).to(mean.dtype).clamp_min(1e-4)
        return mean, scale


def predicted_regret_statistics(task_mean, task_scale):
    mean = task_mean.float().mean(-1)
    scale = task_scale.float().square().sum(-1).sqrt() / task_scale.size(-1)
    return mean, scale.clamp_min(1e-4)


def availability_relative_targets(subset_loss, available, candidates):
    rows = torch.arange(subset_loss.size(0), device=subset_loss.device)
    reference_id = masks_to_subset_ids(available)
    reference = subset_loss[rows, reference_id]
    regret = subset_loss[:, 1:] - reference[:, None]
    valid = ~(candidates[None] & (~available)[:, None]).any(-1)
    return regret, valid


def calibrated_relative(router, task_mean, task_scale, available):
    mean, scale = predicted_regret_statistics(task_mean, task_scale)
    rows = torch.arange(mean.size(0), device=mean.device)
    reference = masks_to_subset_ids(available) - 1
    reference_mean, reference_scale = mean[rows, reference], scale[rows, reference]
    relative_mean = mean - reference_mean[:, None]
    relative_scale = (scale.square() + reference_scale[:, None].square()).sqrt().clamp_min(1e-4)
    q = router.calibration_q[available.sum(1).long()[:, None] - 1,
                             router.candidate_cardinality.long()[None] - 1]
    upper = relative_mean + q * relative_scale
    relative_mean, upper = relative_mean.clone(), upper.clone()
    relative_mean[rows, reference], upper[rows, reference] = 0.0, 0.0
    return relative_mean, relative_scale, upper, mean, scale


@torch.no_grad()
def select_subset(router, scout, available=None):
    batch = scout.size(0)
    if available is None:
        available = torch.ones(batch, M, dtype=torch.bool, device=scout.device)
    available = available.bool()
    routed_scout = scout * available[:, :, None].to(scout.dtype)
    task_mean, task_scale = router(routed_scout, available)
    valid = ~(router.candidate_masks[None] & (~available)[:, None]).any(-1)
    relative_mean, relative_scale, upper, mean, scale = calibrated_relative(
        router, task_mean, task_scale, available
    )
    finite = torch.isfinite(relative_mean) & torch.isfinite(upper)
    admissible = valid & finite & (relative_mean <= 0)
    cards = router.candidate_cardinality[None].expand(batch, -1)
    minimum = cards.masked_fill(~admissible, M + 1).min(1).values
    eligible = admissible & (cards == minimum[:, None])
    best = upper.masked_fill(~eligible, torch.inf).argmin(1)
    fallback = masks_to_subset_ids(available) - 1
    has = admissible.any(1)
    best = torch.where(has, best, fallback)
    chosen = torch.arange(NUM_SUBSETS, device=scout.device)[None] == best[:, None]
    alternatives = admissible & ~chosen
    alt_card = cards.masked_fill(~alternatives, M + 1).min(1).values
    alt_eligible = alternatives & (cards == alt_card[:, None])
    alternative = upper.masked_fill(~alt_eligible, torch.inf).argmin(1)
    has_alt = alternatives.any(1)
    alternative = torch.where(has_alt, alternative, best)
    rows = torch.arange(batch, device=scout.device)
    return {
        "mask": router.candidate_masks[best], "subset_id": best + 1,
        "alternative_mask": router.candidate_masks[alternative],
        "alternative_subset_id": alternative + 1, "has_alternative": has_alt,
        "available": available, "fallback": ~has,
        "selected_full_available": best.eq(fallback),
        "confidence_supported": upper[rows, best] <= 0,
        "predicted_relative_mean_regret": relative_mean[rows, best],
        "predicted_relative_upper_regret": upper[rows, best],
        "predicted_task_regret": task_mean.float()[rows, best],
        "predicted_task_scale": task_scale.float()[rows, best],
    }


@torch.no_grad()
def route_raw(
    model, router, views, available=None,
    unexpected_post_route_failure: bool = False,
    failure_offset: int = 0,
):
    scout = model.scout(views)
    route = select_subset(router, scout, available)
    tokens = model.encode_selected(route["mask"], views)
    execution_mask = route["mask"].clone()
    failure_eligible = execution_mask.sum(1) > 1
    failed_branch = torch.full(
        (execution_mask.size(0),), -1, dtype=torch.long,
        device=execution_mask.device,
    )
    if unexpected_post_route_failure:
        for row in torch.nonzero(failure_eligible, as_tuple=False).flatten().tolist():
            active = torch.nonzero(execution_mask[row], as_tuple=False).flatten()
            chosen = active[(failure_offset + row) % active.numel()]
            execution_mask[row, chosen] = False
            failed_branch[row] = chosen
    logits, state, task_state = model.predict(scout, tokens, execution_mask)
    route.update({"logits": logits, "state": state, "task_state": task_state,
                  "tokens": tokens, "scout": scout,
                  "execution_mask": execution_mask,
                  "post_route_failure_eligible": failure_eligible,
                  "failed_branch": failed_branch})
    return route


# -----------------------------------------------------------------------------
# Losses and metrics
# -----------------------------------------------------------------------------
def ce_loss(logits, label):
    return F.cross_entropy(logits[0], label)


def normalized_ce(logits, label):
    return (
        F.cross_entropy(logits[0], label, reduction="none")
        / math.log(TASK_CLASSES[0])
    )[:, None]


def cross_configuration_kd(student, teacher, temperature, threshold):
    confidence = torch.softmax(teacher[0].detach().float(), 1).amax(1)
    valid = confidence >= threshold
    if not valid.any():
        return student[0].sum() * 0.0
    target = torch.softmax(teacher[0].detach()[valid] / temperature, 1)
    return F.kl_div(torch.log_softmax(student[0][valid] / temperature, 1), target,
                    reduction="batchmean") * temperature ** 2


def new_store():
    return {"labels": [], "predictions": []}


def update_store(store, logits, label):
    store["labels"].append(label.detach().cpu())
    store["predictions"].append(logits[0].argmax(1).detach().cpu())


def finish_metrics(store):
    labels = torch.cat(store["labels"]).numpy()
    pred = torch.cat(store["predictions"]).numpy()
    class_ids = list(range(TASK_CLASSES[0]))
    cm = confusion_matrix(labels, pred, labels=class_ids)
    per_class_acc = {}
    per_class_f1_values = f1_score(
        labels, pred, labels=class_ids, average=None, zero_division=0
    )
    per_class_f1 = {}
    for index, name in IDX_TO_CLASS.items():
        per_class_acc[name] = float(cm[index, index] / max(cm[index].sum(), 1))
        per_class_f1[name] = float(per_class_f1_values[index])
    macro_f1 = float(f1_score(labels, pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(labels, pred))
    balanced = float(balanced_accuracy_score(labels, pred))
    return {
        "acc": {"DIR": accuracy},
        "f1": {"DIR": macro_f1},
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "macc": balanced,
        "mf1": macro_f1,
        "per_class_acc": per_class_acc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
    }


def model_from_args(args, device):
    return COVAStateFarm(args).to(device)


def freeze_bn_statistics(module):
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, PSMSBN2d)):
            child.eval()


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, args, dynamic=False, router=None,
             forced_available: Optional[Sequence[bool]] = None,
             unexpected_post_route_failure: bool = False):
    model.eval()
    if router is not None:
        router.eval()
    store = new_store()
    selected_sum = torch.zeros(M)
    subset_count = torch.zeros(1 << M, dtype=torch.long)
    selected_total = cost_total = available_cost_total = 0.0
    sample_count = fallback_total = full_total = confidence_total = 0
    relative_mean_total = relative_upper_total = 0.0
    failure_eligible_total = 0
    elapsed = 0.0
    measured = 0
    costs = torch.tensor(args.modality_costs, dtype=torch.float32)
    for batch_index, batch in enumerate(tqdm(loader, desc="evaluation", leave=False)):
        views, label, _ = move_batch(batch, device)
        available = None
        if forced_available is not None:
            available = torch.tensor(forced_available, dtype=torch.bool, device=device)[None].expand(views.size(0), -1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with amp_context(device):
            if dynamic:
                result = route_raw(
                    model, router, views, available,
                    unexpected_post_route_failure=unexpected_post_route_failure,
                    failure_offset=batch_index * args.batch_size,
                )
                logits = result["logits"]
            elif available is None:
                logits = model.forward_dense(views)
            else:
                scout = model.scout(views)
                tokens = model.encode_selected(available, views)
                logits = model.predict(scout, tokens, available)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        if batch_index >= args.latency_warmup_batches:
            elapsed += time.perf_counter() - start
            measured += views.size(0)
        update_store(store, logits, label)
        if dynamic:
            mask = result["mask"].detach().cpu()
            ids = masks_to_subset_ids(mask)
            selected_sum += mask.float().sum(0)
            subset_count += torch.bincount(ids, minlength=1 << M)
            selected_total += float(mask.sum())
            cost_total += float((mask.float() * costs[None]).sum())
            current_available = result["available"].detach().cpu()
            available_cost_total += float((current_available.float() * costs[None]).sum())
            sample_count += mask.size(0)
            fallback_total += int(result["fallback"].sum())
            full_total += int(result["selected_full_available"].sum())
            confidence_total += int(result["confidence_supported"].sum())
            relative_mean_total += float(result["predicted_relative_mean_regret"].sum())
            relative_upper_total += float(result["predicted_relative_upper_regret"].sum())
            failure_eligible_total += int(result["post_route_failure_eligible"].sum())
    metrics = finish_metrics(store)
    metrics["model_latency_ms_per_sample"] = 1000.0 * elapsed / max(measured, 1)
    if dynamic:
        metrics.update({
            "avg_modalities": selected_total / max(sample_count, 1),
            "avg_cost": cost_total / max(sample_count, 1),
            "full_cost": float(costs.sum()),
            "relative_branch_cost": cost_total / max(sample_count * float(costs.sum()), 1e-8),
            "available_baseline_cost": available_cost_total / max(sample_count, 1),
            "relative_to_available_cost": cost_total / max(available_cost_total, 1e-8),
            "numerical_fallback_rate": fallback_total / max(sample_count, 1),
            "selected_full_available_rate": full_total / max(sample_count, 1),
            "confidence_supported_selection_rate": confidence_total / max(sample_count, 1),
            "mean_selected_relative_regret": relative_mean_total / max(sample_count, 1),
            "mean_selected_relative_upper_regret": relative_upper_total / max(sample_count, 1),
            "unexpected_post_route_failure_applied": bool(unexpected_post_route_failure),
            "unexpected_post_route_failure_eligible_rate": (
                failure_eligible_total / max(sample_count, 1)
            ),
            "selection_rate": {name: float(selected_sum[i] / max(sample_count, 1))
                               for i, name in enumerate(MODALITY_NAMES)},
            "selection_pattern_rate": {
                "+".join(MODALITY_NAMES[m] for m in range(M) if subset & (1 << m)):
                    float(subset_count[subset] / max(sample_count, 1))
                for subset in range(1, 1 << M) if subset_count[subset] > 0
            },
        })
    return metrics


@torch.no_grad()
def evaluate_all_subsets(model, loader, device):
    model.eval()
    masks = mask_table(device)
    stores = {subset: new_store() for subset in range(1, 1 << M)}
    for batch in tqdm(loader, desc=f"{NUM_SUBSETS}-subset evaluation", leave=False):
        views, label, _ = move_batch(batch, device)
        with amp_context(device):
            scout, tokens = model.encode_all(views)
            for subset in range(1, 1 << M):
                mask = masks[subset][None].expand(views.size(0), -1)
                logits = model.predict(scout, tokens, mask)[0]
                update_store(stores[subset], logits, label)
    by_subset, by_cardinality = {}, {}
    for subset, store in stores.items():
        names = [MODALITY_NAMES[m] for m in range(M) if subset & (1 << m)]
        by_subset[str(subset)] = {"modalities": names, "cardinality": len(names),
                                  **finish_metrics(store)}
    for cardinality in range(1, M + 1):
        group = [value for value in by_subset.values() if value["cardinality"] == cardinality]
        scores = np.asarray([value["macc"] for value in group])
        by_cardinality[str(cardinality)] = {
            "subset_count": len(group), "mean_macc": float(scores.mean()),
            "best_macc": float(scores.max()), "worst_macc": float(scores.min()),
        }
    return {"by_cardinality": by_cardinality, "by_subset": by_subset}


# -----------------------------------------------------------------------------
# Stage 1: subset-capable dense teacher
# -----------------------------------------------------------------------------
def train_teacher(args, loaders, device):
    stage_dir = args.run_dir / "teacher"
    last, best = stage_dir / "last.pt", stage_dir / "bestmacc.pt"
    if completed_stage(stage_dir, best, args.resume):
        return best
    model = model_from_args(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device)
    start, best_score = 1, -1.0
    if args.resume and last.exists():
        checkpoint = load_checkpoint(last, model, optimizer, scaler)
        start = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best", {}).get("macc", -1))
    for epoch in range(start, args.teacher_epochs + 1):
        model.train()
        running = count = 0.0
        for batch in tqdm(loaders["train"], desc=f"teacher {epoch}", leave=False):
            views, label, _ = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                scout, tokens = model.encode_all(views)
                full_mask = torch.ones(views.size(0), M, dtype=torch.bool, device=device)
                full_logits = model.predict(scout, tokens, full_mask)[0]
                full_loss = ce_loss(full_logits, label)
                subset_loss = full_loss.new_zeros(())
                kd_loss = full_loss.new_zeros(())
                for _ in range(args.subset_samples_per_batch):
                    subset_mask = sample_balanced_masks(views.size(0), device, include_full=True)
                    subset_logits = model.predict(scout, tokens, subset_mask)[0]
                    subset_loss += ce_loss(subset_logits, label)
                    kd_loss += cross_configuration_kd(subset_logits, full_logits,
                                                      args.kd_temperature, args.kd_confidence)
                subset_loss /= args.subset_samples_per_batch
                kd_loss /= args.subset_samples_per_batch
                heavy_target = tokens.detach()
                scout_loss = (1 - F.cosine_similarity(scout, heavy_target, dim=-1)).mean()
                scout_loss += 0.1 * F.smooth_l1_loss(scout, heavy_target)
                loss = full_loss + args.subset_loss_weight * subset_loss + \
                       args.kd_weight * kd_loss + args.scout_alignment_weight * scout_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            count += 1
        val = evaluate(model, loaders["val"], device, args)
        print(f"[teacher {epoch}] loss={running/max(count,1):.4f} val_mAcc={val['macc']:.4f}")
        if val["macc"] > best_score:
            best_score = val["macc"]
            save_checkpoint(best, epoch, model, best={"macc": best_score})
        save_checkpoint(last, epoch, model, optimizer, scaler, {"macc": best_score})
    finalize_stage(stage_dir, best, {"best_validation_macc": best_score})
    return best


# -----------------------------------------------------------------------------
# Stage 2: exhaustive counterfactual cache and ACVE
# -----------------------------------------------------------------------------
@torch.no_grad()
def build_cache(args, split, loader, model, device, cache_dir, source_checkpoint):
    model.eval()
    masks = mask_table(device)
    collected = {"scout_tokens": [], "subset_loss": [], "predictions": [], "labels": [], "paths": []}
    for batch in tqdm(loader, desc=f"counterfactual cache {split}", leave=False):
        views, label, paths = move_batch(batch, device)
        with amp_context(device):
            scout, tokens = model.encode_all(views)
            losses = torch.full((views.size(0), 1 << M, K), torch.nan, device=device)
            predictions = torch.full((views.size(0), 1 << M, K), -1, dtype=torch.long, device=device)
            for subset in range(1, 1 << M):
                mask = masks[subset][None].expand(views.size(0), -1)
                logits = model.predict(scout, tokens, mask)[0]
                losses[:, subset] = normalized_ce(logits, label)
                predictions[:, subset, 0] = logits[0].argmax(1)
        collected["scout_tokens"].append(scout.half().cpu())
        collected["subset_loss"].append(losses.float().cpu())
        collected["predictions"].append(predictions.cpu())
        collected["labels"].append(label[:, None].cpu())
        collected["paths"].extend(list(paths))
    output = {key: torch.cat(value, 0) for key, value in collected.items() if key != "paths"}
    output.update({
        "paths": collected["paths"], "mask_table": masks.cpu(),
        "modality_names": MODALITY_NAMES, "task_names": TASK_NAMES,
        "source_checkpoint": str(source_checkpoint),
        "target_semantics": "normalized DIR CE for every 15 non-empty view subset",
    })
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(output, temporary)
    temporary.replace(path)
    print(f"saved {path} ({len(collected['paths'])} samples)")
    return path


def analyze_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    losses = cache["subset_loss"][:, 1:, 0]
    local = losses.argmin(1)
    fixed = int(losses.mean(0).argmin())
    unique, counts = torch.unique(local, return_counts=True)
    report = {
        "unique_samplewise_best_subsets": int(unique.numel()),
        "dominant_best_subset_rate": float(counts.max() / len(local)),
        "globally_best_fixed_subset_id": fixed + 1,
        "globally_best_fixed_subset": [MODALITY_NAMES[m] for m in range(M) if (fixed + 1) & (1 << m)],
        "samplewise_minus_fixed_normalized_loss": float(losses[:, fixed].mean() - losses[torch.arange(len(losses)), local].mean()),
    }
    write_json(path.with_name(path.stem + "_report.json"), report)
    return report


def run_cache(args, loaders, device):
    model = model_from_args(args, device)
    load_checkpoint(args.teacher_ckpt, model)
    for split in ("train", "val", "test"):
        path = build_cache(args, split, loaders[split], model, device,
                           args.oracle_cache_dir, args.teacher_ckpt)
        if split == "val":
            print(json.dumps(analyze_cache(path), indent=2, ensure_ascii=False))


class RegretDataset(Dataset):
    def __init__(self, path):
        self.cache = torch.load(path, map_location="cpu", weights_only=False)

    def __len__(self):
        return self.cache["scout_tokens"].size(0)

    def __getitem__(self, index):
        return self.cache["scout_tokens"][index].float(), self.cache["subset_loss"][index].float()


def random_available(batch, device, probability):
    available = torch.rand(batch, M, device=device) >= probability
    empty = ~available.any(1)
    if empty.any():
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        keep = torch.randint(0, M, (rows.numel(),), device=device)
        available[rows, keep] = True
    return available


def deterministic_availability(device, all_patterns=True):
    table = mask_table(device)[1:]
    ordered = torch.cat((table[-1:], table[:-1]), 0)
    return ordered if all_patterns else ordered[ordered.sum(1) >= M - 1]


def router_batch_loss(router, batch, args, device, train=True, forced_available=None):
    scout, subset_loss = [item.to(device) for item in batch]
    if forced_available is None:
        available = random_available(scout.size(0), device,
                                     args.router_availability_dropout if train else 0)
    else:
        available = forced_available.to(device).bool()
        if available.ndim == 1:
            available = available[None].expand(scout.size(0), -1)
    task_mean, task_scale = router(scout * available[:, :, None].to(scout.dtype), available)
    target, valid = availability_relative_targets(subset_loss, available, router.candidate_masks)
    valid_tasks = valid[:, :, None].expand_as(task_mean)
    error = task_mean.float() - target.float()
    scale = task_scale.float().clamp_min(1e-4)
    nll = (torch.log(scale) + 0.5 * (error / scale).square())[valid_tasks].mean()
    smooth = F.smooth_l1_loss(task_mean.float()[valid_tasks], target.float()[valid_tasks])
    predicted_mean, predicted_scale = predicted_regret_statistics(task_mean, task_scale)
    target_mean = target.float().mean(-1)
    suff_target = (target_mean <= 0).float()
    sufficiency = F.binary_cross_entropy_with_logits(
        (-predicted_mean / predicted_scale.detach())[valid], suff_target[valid]
    )
    temperature = max(args.router_temperature, 1e-4)
    predicted_logits = (-predicted_mean / temperature).masked_fill(~valid, -torch.inf)
    target_logits = (-target_mean / temperature).masked_fill(~valid, -torch.inf)
    target_probability = torch.softmax(target_logits, 1)
    predicted_log_probability = torch.log_softmax(predicted_logits, 1).masked_fill(~valid, 0)
    listwise = -(target_probability * predicted_log_probability).sum(1).mean()
    loss = nll + args.router_smooth_weight * smooth + \
           args.router_sufficiency_weight * sufficiency + args.router_listwise_weight * listwise
    with torch.no_grad():
        pred_sufficient = valid & (predicted_mean <= 0)
        suff_acc = (pred_sufficient[valid] == suff_target[valid].bool()).float().mean()
        cards = router.candidate_cardinality[None].expand_as(valid)
        oracle_card = cards.masked_fill(~(valid & suff_target.bool()), M + 1).min(1).values
        pred_card = cards.masked_fill(~pred_sufficient, M + 1).min(1).values
        pred_card = torch.where(pred_card == M + 1, available.sum(1), pred_card)
        card_error = (pred_card.float() - oracle_card.float()).abs().mean()
    return loss, nll.detach(), smooth.detach(), sufficiency.detach(), listwise.detach(), suff_acc, card_error


@torch.no_grad()
def evaluate_router(router, loader, args, device):
    router.eval()
    total = np.zeros(7)
    count = 0
    for batch in loader:
        for pattern in deterministic_availability(device, all_patterns=False):
            values = router_batch_loss(router, batch, args, device, False, pattern)
            total += np.asarray([float(value) for value in values])
            count += 1
    keys = ("loss", "nll", "smooth", "sufficiency", "listwise", "sufficiency_accuracy", "cardinality_error")
    return dict(zip(keys, (total / max(count, 1)).tolist()))


def train_router(args, device):
    train_ds = RegretDataset(args.oracle_cache_dir / "train.pt")
    val_ds = RegretDataset(args.oracle_cache_dir / "val.pt")
    train_loader = DataLoader(train_ds, batch_size=args.router_batch_size, shuffle=True,
                              num_workers=min(args.num_workers, 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.router_batch_size, shuffle=False,
                            num_workers=min(args.num_workers, 4), pin_memory=True)
    router = ACVE(args.embed, args.router_hidden).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.router_lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device)
    last, best = args.run_dir / "router" / "last.pt", args.run_dir / "router" / "best.pt"
    if completed_stage(best.parent, best, args.resume):
        return best
    start, best_loss = 1, float("inf")
    if args.resume and last.exists():
        checkpoint = load_checkpoint(last, router, optimizer, scaler)
        start = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best", {}).get("loss", float("inf")))
    for epoch in range(start, args.router_epochs + 1):
        router.train()
        running = count = 0.0
        for batch in tqdm(train_loader, desc=f"ACVE {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                values = router_batch_loss(router, batch, args, device, True)
                loss = values[0]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(router.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            count += 1
        val = evaluate_router(router, val_loader, args, device)
        print(f"[ACVE {epoch}] train={running/max(count,1):.4f} val={val['loss']:.4f} "
              f"suff={val['sufficiency_accuracy']:.4f} card_err={val['cardinality_error']:.4f}")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            save_checkpoint(best, epoch, router, best={"loss": best_loss})
        save_checkpoint(last, epoch, router, optimizer, scaler, {"loss": best_loss})
    finalize_stage(best.parent, best, {"best_validation_loss": best_loss})
    return best


# -----------------------------------------------------------------------------
# Availability/subset-cardinality conditional calibration
# -----------------------------------------------------------------------------
def one_sided_quantile(values, coverage):
    values = values.float().flatten()
    if values.numel() == 0:
        raise ValueError("empty calibration group")
    rank = min(values.numel(), math.ceil((values.numel() + 1) * coverage))
    return max(0.0, float(values.kthvalue(rank).values))


@torch.no_grad()
def calibrate_from_cache(args, device, cache_path, output_path, source_checkpoint):
    dataset = RegretDataset(cache_path)
    loader = DataLoader(dataset, batch_size=args.router_batch_size, shuffle=False,
                        num_workers=min(args.num_workers, 4), pin_memory=True)
    router = ACVE(args.embed, args.router_hidden).to(device)
    load_checkpoint(args.router_ckpt, router)
    router.eval()
    residuals = {(a, s): [] for a in range(1, M + 1) for s in range(1, a + 1)}
    patterns = deterministic_availability(device, all_patterns=True)
    for batch in tqdm(loader, desc="ACVE calibration", leave=False):
        scout, subset_loss = [item.to(device) for item in batch]
        for pattern in patterns:
            available = pattern[None].expand(scout.size(0), -1)
            with amp_context(device):
                task_mean, task_scale = router(scout * available[:, :, None].to(scout.dtype), available)
            target, valid = availability_relative_targets(subset_loss, available, router.candidate_masks)
            mean, scale = predicted_regret_statistics(task_mean, task_scale)
            target_mean = target.float().mean(-1)
            rows = torch.arange(scout.size(0), device=device)
            reference = masks_to_subset_ids(available) - 1
            predicted_relative = mean - mean[rows, reference][:, None]
            target_relative = target_mean - target_mean[rows, reference][:, None]
            relative_scale = (scale.square() + scale[rows, reference][:, None].square()).sqrt().clamp_min(1e-4)
            standardized = (target_relative - predicted_relative) / relative_scale
            a_card = int(pattern.sum())
            for s_card in range(1, a_card + 1):
                use = valid & (router.candidate_cardinality[None] == s_card)
                if use.any():
                    residuals[(a_card, s_card)].append(standardized[use].cpu())
    q_matrix = torch.ones(M, M)
    realized, counts = {}, {}
    for a_card in range(1, M + 1):
        for s_card in range(1, a_card + 1):
            values = torch.cat(residuals[(a_card, s_card)])
            q = one_sided_quantile(values, args.calibration_coverage)
            q_matrix[a_card - 1, s_card - 1] = q
            key = f"A{a_card}_S{s_card}"
            realized[key] = float((values <= q).float().mean())
            counts[key] = int(values.numel())
    report = {
        "method": "one-sided availability/subset-cardinality calibration of paired relative regret",
        "coverage_target": args.calibration_coverage,
        "q_by_available_and_selected_cardinality": q_matrix.tolist(),
        "realized_coverage": realized, "counts": counts,
        "router_checkpoint": args.router_ckpt, "source_model_checkpoint": source_checkpoint,
        "source_cache": cache_path, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_path, report)
    print(f"calibration saved: {output_path}")
    return report


def load_router(args, device, prefer_student=True):
    router = ACVE(args.embed, args.router_hidden).to(device)
    load_checkpoint(args.router_ckpt, router)
    calibration = args.student_calibration_path if prefer_student and args.student_calibration_path.exists() \
        else args.teacher_calibration_path
    if calibration.exists():
        with open(calibration, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        q = torch.tensor(payload["q_by_available_and_selected_cardinality"], device=device)
        if q.shape != (M, M):
            raise ValueError(f"invalid calibration matrix in {calibration}")
        router.calibration_q.copy_(q)
    elif args.require_calibration:
        raise FileNotFoundError(f"run --stage calibrate first: {calibration}")
    router.loaded_calibration = calibration if calibration.exists() else None
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad = False
    return router


def run_teacher_calibration(args, device):
    return calibrate_from_cache(args, device, args.oracle_cache_dir / "val.pt",
                                args.teacher_calibration_path, args.teacher_ckpt)


# -----------------------------------------------------------------------------
# Stage 3: sparse SCML student
# -----------------------------------------------------------------------------
def randomly_drop_one(mask, probability):
    output = mask.clone()
    if probability <= 0:
        return output
    for row in range(output.size(0)):
        candidates = torch.nonzero(output[row], as_tuple=False).flatten()
        if candidates.numel() > 1 and random.random() < probability:
            chosen = candidates[torch.randint(candidates.numel(), (1,), device=mask.device)]
            output[row, chosen] = False
    return output


def explore_route(primary, alternative, has_alternative, probability):
    if probability <= 0:
        return primary
    choose = (torch.rand(primary.size(0), device=primary.device) < probability) & has_alternative
    return torch.where(choose[:, None], alternative, primary)


def train_sparse(args, loaders, device):
    last, best = args.run_dir / "sparse" / "last.pt", args.run_dir / "sparse" / "bestmacc.pt"
    if completed_stage(best.parent, best, args.resume):
        return best
    teacher = model_from_args(args, device)
    load_checkpoint(args.teacher_ckpt, teacher)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student = copy.deepcopy(teacher).to(device)
    for parameter in student.parameters():
        parameter.requires_grad = True
    # Scout semantics are already aligned by the teacher.  Freeze them so the
    # fixed ACVE input distribution does not drift during sparse fine-tuning.
    for parameter in student.scout.parameters():
        parameter.requires_grad = False
    router = load_router(args, device, prefer_student=False)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    print("sparse student trainable parameters:", sum(p.numel() for p in trainable))
    optimizer = torch.optim.AdamW(trainable, lr=args.sparse_lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device)
    start, best_score = 1, -1.0
    if args.resume and last.exists():
        checkpoint = load_checkpoint(last, student, optimizer, scaler)
        start = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best", {}).get("macc", -1))
    for epoch in range(start, args.sparse_epochs + 1):
        student.train()
        student.scout.eval()
        # Routed branch groups can be small and composition-dependent.  Keep
        # the subset-capable teacher's normalization statistics fixed while
        # adapting SCML weights.
        freeze_bn_statistics(student)
        running = count = 0.0
        for batch in tqdm(loaders["train"], desc=f"sparse {epoch}", leave=False):
            views, label, _ = move_batch(batch, device)
            available = random_available(views.size(0), device, args.sparse_availability_dropout)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), amp_context(device):
                scout, teacher_tokens = teacher.encode_all(views)
                teacher_logits = teacher.predict(scout, teacher_tokens, available)[0]
                route = select_subset(router, scout, available)
                selected_mask = explore_route(route["mask"], route["alternative_mask"],
                                              route["has_alternative"],
                                              args.route_exploration_probability)
            execution_mask = randomly_drop_one(selected_mask, args.failure_dropout)
            with amp_context(device):
                # dtype-safe: mask and indices only; no Half -> Float indexed assignment.
                student_tokens = student.encode_selected(selected_mask, views)
                student_logits = student.predict(scout.detach(), student_tokens, execution_mask)[0]
                classification = ce_loss(student_logits, label)
                kd = cross_configuration_kd(student_logits, teacher_logits,
                                             args.kd_temperature, args.kd_confidence)
                loss = classification + args.kd_weight * kd
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            count += 1
        val = evaluate(student, loaders["val"], device, args, dynamic=True, router=router)
        print(f"[sparse {epoch}] loss={running/max(count,1):.4f} val_mAcc={val['macc']:.4f} "
              f"avg_branches={val['avg_modalities']:.3f}")
        if val["macc"] > best_score:
            best_score = val["macc"]
            save_checkpoint(best, epoch, student, best={"macc": best_score})
        save_checkpoint(last, epoch, student, optimizer, scaler, {"macc": best_score})
    finalize_stage(best.parent, best, {"best_validation_macc": best_score})
    return best


def run_student_recalibration(args, loaders, device):
    student = model_from_args(args, device)
    load_checkpoint(args.sparse_ckpt, student)
    cache = build_cache(args, "val", loaders["val"], student, device,
                        args.student_cache_dir, args.sparse_ckpt)
    return calibrate_from_cache(args, device, cache, args.student_calibration_path, args.sparse_ckpt)


# -----------------------------------------------------------------------------
# Final evaluation
# -----------------------------------------------------------------------------
def run_final_evaluation(args, loaders, device):
    model = model_from_args(args, device)
    load_checkpoint(args.sparse_ckpt, model)
    router = load_router(args, device, prefer_student=True)
    dense = evaluate(model, loaders["test"], device, args)
    dynamic = evaluate(model, loaders["test"], device, args, dynamic=True, router=router)
    validation_dynamic = evaluate(model, loaders["val"], device, args, dynamic=True, router=router)
    validation_subsets = evaluate_all_subsets(model, loaders["val"], device) if args.eval_all_subsets else {}
    test_subsets = evaluate_all_subsets(model, loaders["test"], device) if args.eval_all_subsets else {}
    matched_fixed = {}
    if validation_subsets:
        cardinality = int(np.clip(round(validation_dynamic["avg_modalities"]), 1, M))
        group = [(sid, value) for sid, value in validation_subsets["by_subset"].items()
                 if value["cardinality"] == cardinality]
        sid, val_best = max(group, key=lambda item: item[1]["macc"])
        test_best = test_subsets["by_subset"][sid]
        matched_fixed = {
            "selection_split": "validation", "selection_metric": "macc",
            "rounded_validation_dynamic_cardinality": cardinality,
            "subset_id": int(sid), "modalities": test_best["modalities"],
            "validation_macc": val_best["macc"], "test_macc": test_best["macc"],
            "test_mf1": test_best["mf1"],
            "dynamic_minus_fixed_macc_pp": 100 * (dynamic["macc"] - test_best["macc"]),
        }
    failures = {}
    unexpected_failure = {}
    if args.eval_failures:
        for missing, name in enumerate(MODALITY_NAMES):
            available = [True] * M
            available[missing] = False
            failures[f"missing_{name}"] = {
                "static_available_streams": evaluate(model, loaders["test"], device, args,
                                                     forced_available=available),
                "dynamic": evaluate(model, loaders["test"], device, args, dynamic=True,
                                    router=router, forced_available=available),
            }
        unexpected_failure = evaluate(
            model, loaders["test"], device, args, dynamic=True, router=router,
            unexpected_post_route_failure=True,
        )
    split_protocol = {}
    split_meta = args.split_dir / "split_meta.json"
    if split_meta.exists():
        with open(split_meta, "r", encoding="utf-8") as handle:
            split_protocol = json.load(handle)
    result = {
        "method": "COVA-MTL v6.3 StateFarm static external validation",
        "scope_note": (
            "StateFarm has one 10-class driver-activity task. The four selectable "
            "branches are full, upper, lower-left, and lower-right semantic views "
            "from the supplied StateFarm protocol; they are not independent sensors."
        ),
        "branch_semantics": {
            "full": "entire StateFarm image",
            "upper": "top 64% of the image",
            "lower_left": "lower 62% height and left 62% width",
            "lower_right": "lower 62% height and right 62% width",
        },
        "seed": args.seed, "split_seed": args.split_seed, "run_dir": args.run_dir,
        "split_protocol": split_protocol,
        "checkpoint": args.sparse_ckpt, "router_checkpoint": args.router_ckpt,
        "calibration": getattr(router, "loaded_calibration", None),
        "selection": (
            "minimum-cardinality availability-valid subset with predicted relative mean "
            "DIR loss no greater than the all-currently-available self-reference; calibrated "
            "upper risk breaks same-cardinality ties"
        ),
        "selection_has_no_cost_or_budget_term": True,
        "modality_costs_reporting_only": dict(zip(MODALITY_NAMES, args.modality_costs)),
        "dense": dense, "dynamic": dynamic,
        "matched_cardinality_best_fixed_subset": matched_fixed,
        "exhaustive_nonempty_subsets": test_subsets,
        "single_branch_failure": failures,
        "unexpected_post_route_failure": unexpected_failure,
        "macc_drop_pp": 100 * (dense["macc"] - dynamic["macc"]),
        "accuracy_drop_pp": 100 * (dense["accuracy"] - dynamic["accuracy"]),
        "macro_f1_drop_pp": 100 * (dense["macro_f1"] - dynamic["macro_f1"]),
        "branch_cost_reduction_percent": 100 * (1.0 - dynamic["relative_branch_cost"]),
        "latency_reduction_percent": 100 * (
            dense["model_latency_ms_per_sample"]
            - dynamic["model_latency_ms_per_sample"]
        ) / max(dense["model_latency_ms_per_sample"], 1e-8),
        "parameter_count": {
            "student": sum(p.numel() for p in model.parameters()),
            "ACVE": sum(p.numel() for p in router.parameters()),
            "SCMC": sum(p.numel() for p in model.scmc.parameters()),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(args.run_dir / "final_test.json", result)
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
    return result


# -----------------------------------------------------------------------------
# Smoke test, CLI, five-seed orchestration
# -----------------------------------------------------------------------------
def smoke_test(args, device):
    args = copy.deepcopy(args)
    args.pretrained = False
    args.image_size = 64
    model = model_from_args(args, device).eval()
    router = ACVE(args.embed, args.router_hidden).to(device).eval()
    views = torch.rand(2, M, 3, 64, 64, device=device)
    with torch.no_grad():
        scout, tokens = model.encode_all(views)
        full = torch.ones(2, M, dtype=torch.bool, device=device)
        task_mean, task_scale = router(scout, full)
        logits = model.predict(scout, tokens, full)[0]
        route = route_raw(model, router, views)
        missing = full.clone(); missing[:, 1] = False
        failure = select_subset(router, scout, missing)
    assert scout.shape == (2, M, args.embed)
    assert tokens.shape == (2, M, args.embed)
    assert task_mean.shape == (2, NUM_SUBSETS, K)
    assert task_scale.shape == task_mean.shape and (task_scale > 0).all()
    assert logits[0].shape == (2, 10)
    assert route["mask"].shape == (2, M)
    assert not failure["mask"][:, 1].any()
    print("smoke test passed")


def parse_int_list(text):
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be unique comma-separated integers")
    return values


def parse_float_list(text, expected):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated costs")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description="COVA-MTL v6.3 on static StateFarm (five seeds)")
    parser.add_argument("--stage", choices=("teacher", "cache", "router", "calibrate", "sparse",
                                             "recalibrate", "eval", "all", "aggregate", "smoke"),
                        default="all")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("STATEFARM_ROOT", "data/StateFarm")))
    parser.add_argument("--run-dir", type=Path,
                        default=Path(os.environ.get("STATEFARM_RUN_DIR", "RUNS/STATEFARM_COVA_v6_3_MultiSeed5")))
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument(
        "--force-splits", action=argparse.BooleanOptionalAction, default=False,
        help="recreate the shared split CSV files",
    )
    parser.add_argument("--multi-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-per-seed", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "use split seed equal to each model seed, matching the supplied legacy "
            "StateFarm code; default keeps one shared split for clean training-seed variance"
        ),
    )
    parser.add_argument("--continue-on-seed-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--router-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--preview-size", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--scmc-hidden", type=int, default=128)
    parser.add_argument("--scmc-strength", type=float, default=0.25)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--sparse-epochs", type=int, default=20)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--sparse-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--subset-samples-per-batch", type=int, default=2)
    parser.add_argument("--subset-loss-weight", type=float, default=0.5)
    parser.add_argument("--scout-alignment-weight", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-confidence", type=float, default=0.6)
    parser.add_argument("--router-hidden", type=int, default=256)
    parser.add_argument("--router-smooth-weight", type=float, default=0.5)
    parser.add_argument("--router-sufficiency-weight", type=float, default=1.0)
    parser.add_argument("--router-listwise-weight", type=float, default=1.0)
    parser.add_argument("--router-temperature", type=float, default=0.1)
    parser.add_argument("--router-availability-dropout", type=float, default=0.15)
    parser.add_argument("--sparse-availability-dropout", type=float, default=0.15)
    parser.add_argument("--failure-dropout", type=float, default=0.2)
    parser.add_argument("--route-exploration-probability", type=float, default=0.25)
    parser.add_argument("--calibration-coverage", type=float, default=0.90)
    parser.add_argument("--require-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--modality-costs", default="1,1,1,1",
                        help="reporting only; never used by SSR")
    parser.add_argument("--latency-warmup-batches", type=int, default=3)
    parser.add_argument("--eval-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-all-subsets", action=argparse.BooleanOptionalAction, default=True)
    return parser


def configure_paths(args):
    args.teacher_ckpt = args.run_dir / "teacher" / "bestmacc.pt"
    args.oracle_cache_dir = args.run_dir / "oracle_cache"
    args.student_cache_dir = args.run_dir / "student_cache"
    args.router_ckpt = args.run_dir / "router" / "best.pt"
    args.teacher_calibration_path = args.run_dir / "router" / "calibration_teacher.json"
    args.student_calibration_path = args.run_dir / "router" / "calibration_student.json"
    args.sparse_ckpt = args.run_dir / "sparse" / "bestmacc.pt"
    return args


def finalize_args(args):
    args.data_root = args.data_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.seeds = parse_int_list(args.seeds)
    args.modality_costs = parse_float_list(args.modality_costs, M)
    if args.split_dir is None:
        args.split_dir = args.run_dir / f"splits_driver_group_seed{args.split_seed}_64-16-20"
    args.split_dir = args.split_dir.expanduser().resolve()
    if args.embed <= 0 or args.d_model <= 0 or args.router_hidden <= 0:
        raise ValueError("model dimensions must be positive")
    if args.d_model % args.attention_heads:
        raise ValueError("d-model must be divisible by attention-heads")
    for name in ("router_availability_dropout", "sparse_availability_dropout"):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    for name in ("failure_dropout", "route_exploration_probability"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0.5 < args.calibration_coverage < 1:
        raise ValueError("calibration coverage must be in (0.5, 1)")
    return configure_paths(args)


def shutdown_loaders(loaders):
    if not loaders:
        return
    for loader in loaders.values():
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            loader._iterator = None


def release_resources(loaders=None):
    shutdown_loaders(loaders)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_seed(args, device):
    seed_everything(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "config.json", vars(args))
    if args.stage == "smoke":
        smoke_test(args, device)
        return None
    loaders = None
    try:
        loaders = build_loaders(args, cache_order=False)
        if args.stage in ("teacher", "all"):
            train_teacher(args, loaders, device)
        if args.stage in ("cache", "all"):
            cache_loaders = build_loaders(args, cache_order=True)
            try:
                run_cache(args, cache_loaders, device)
            finally:
                shutdown_loaders(cache_loaders)
        if args.stage in ("router", "all"):
            train_router(args, device)
        if args.stage in ("calibrate", "all"):
            run_teacher_calibration(args, device)
        if args.stage in ("sparse", "all"):
            train_sparse(args, loaders, device)
        if args.stage in ("recalibrate", "all"):
            recalibration_loaders = build_loaders(args, cache_order=True)
            try:
                run_student_recalibration(args, recalibration_loaders, device)
            finally:
                shutdown_loaders(recalibration_loaders)
        if args.stage in ("eval", "all"):
            return run_final_evaluation(args, loaders, device)
        return None
    finally:
        release_resources(loaders)


def make_seed_args(base, seed):
    args = copy.deepcopy(base)
    args.seed = int(seed)
    args.run_dir = base.run_dir / f"seed_{seed}"
    if base.split_per_seed:
        args.split_seed = int(seed)
        args.split_dir = base.run_dir / f"splits_driver_group_seed{seed}_64-16-20"
    return configure_paths(args)


def result_row(seed, result):
    dense, dynamic = result["dense"], result["dynamic"]
    row = {
        "seed": seed, "dense_macc": dense["macc"], "dense_mf1": dense["mf1"],
        "dense_acc": dense["acc"]["DIR"], "dynamic_macc": dynamic["macc"],
        "dynamic_mf1": dynamic["mf1"], "dynamic_acc": dynamic["acc"]["DIR"],
        "macc_drop_pp": result["macc_drop_pp"],
        "dynamic_avg_modalities": dynamic["avg_modalities"],
        "dynamic_relative_branch_cost": dynamic["relative_branch_cost"],
        "dense_latency_ms": dense["model_latency_ms_per_sample"],
        "dynamic_latency_ms": dynamic["model_latency_ms_per_sample"],
        "accuracy_drop_pp": result["accuracy_drop_pp"],
        "macro_f1_drop_pp": result["macro_f1_drop_pp"],
        "branch_cost_reduction_percent": result["branch_cost_reduction_percent"],
        "latency_reduction_percent": result["latency_reduction_percent"],
    }
    for class_name in STATEFARM_CLASSES:
        row[f"dense_{class_name}_acc"] = dense["per_class_acc"][class_name]
        row[f"dynamic_{class_name}_acc"] = dynamic["per_class_acc"][class_name]
        row[f"dense_{class_name}_f1"] = dense["per_class_f1"][class_name]
        row[f"dynamic_{class_name}_f1"] = dynamic["per_class_f1"][class_name]
    for modality in MODALITY_NAMES:
        row[f"selection_rate_{modality}"] = dynamic["selection_rate"][modality]
    fixed = result.get("matched_cardinality_best_fixed_subset", {})
    if fixed:
        row["fixed_subset"] = "+".join(fixed["modalities"])
        row["fixed_test_macc"] = fixed["test_macc"]
        row["dynamic_minus_fixed_macc_pp"] = fixed["dynamic_minus_fixed_macc_pp"]
    for name, failure in result.get("single_branch_failure", {}).items():
        row[f"{name}_dynamic_macc"] = failure["dynamic"]["macc"]
        row[f"{name}_static_macc"] = failure["static_available_streams"]["macc"]
    unexpected = result.get("unexpected_post_route_failure", {})
    if unexpected:
        row["unexpected_post_route_failure_macc"] = unexpected["macc"]
        row["unexpected_post_route_drop_pp"] = 100.0 * (
            dynamic["macc"] - unexpected["macc"]
        )
        row["unexpected_post_route_failure_eligible_rate"] = unexpected[
            "unexpected_post_route_failure_eligible_rate"
        ]
    return row


def aggregate(args, require_all=True):
    rows, missing = [], []
    for seed in args.seeds:
        path = args.run_dir / f"seed_{seed}" / "final_test.json"
        if not path.exists():
            missing.append(seed)
            continue
        with open(path, "r", encoding="utf-8") as handle:
            rows.append(result_row(seed, json.load(handle)))
    if not rows:
        raise FileNotFoundError(f"no final_test.json under {args.run_dir}")
    if missing and require_all:
        raise FileNotFoundError(f"missing seeds: {missing}")
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(args.run_dir / "multi_seed_results.csv", index=False)
    summary = {}
    records = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column == "seed":
            continue
        values = frame[column].dropna().to_numpy(float)
        stats = {"n": len(values), "mean": float(values.mean()),
                 "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                 "min": float(values.min()), "max": float(values.max())}
        summary[column] = stats
        records.append({"metric": column, **stats})
    pd.DataFrame(records).to_csv(args.run_dir / "multi_seed_summary.csv", index=False)
    payload = {
        "method": "COVA-MTL v6.3 StateFarm static five-seed external validation",
        "requested_seeds": args.seeds, "completed_seeds": frame.seed.tolist(),
        "missing_seeds": missing, "split_seed": args.split_seed,
        "split_per_seed": bool(args.split_per_seed),
        "split_dir": args.split_dir, "std": "sample standard deviation (ddof=1)",
        "metrics": summary, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(args.run_dir / "multi_seed_summary.json", payload)
    for key in ("dense_macc", "dynamic_macc", "macc_drop_pp", "dynamic_avg_modalities",
                "dynamic_relative_branch_cost"):
        if key in summary:
            print(f"{key}: {summary[key]['mean']:.6f} ± {summary[key]['std']:.6f}")
    return payload


def main():
    args = finalize_args(build_parser().parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and args.stage not in ("smoke", "aggregate") and device.type != "cuda":
        raise RuntimeError("CUDA unavailable; fix CUDA/PyTorch or explicitly pass --no-require-cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    if args.stage == "aggregate":
        aggregate(args, require_all=False)
        return
    if args.stage != "smoke" and not args.split_per_seed:
        ensure_splits(args)
    if not args.multi_seed:
        run_single_seed(args, device)
        return
    if args.stage == "smoke":
        run_single_seed(make_seed_args(args, args.seeds[0]), device)
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "multi_seed_config.json", vars(args))
    status_path = args.run_dir / "multi_seed_status.json"
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as handle:
            status = json.load(handle)
    else:
        status = {"requested_seeds": args.seeds, "split_seed": args.split_seed, "seeds": {}}
    for index, seed in enumerate(args.seeds, 1):
        seed_args = make_seed_args(args, seed)
        final_path = seed_args.run_dir / "final_test.json"
        if args.skip_completed_seeds and args.stage == "all" and final_path.exists():
            print(f"[{index}/{len(args.seeds)}] seed {seed}: already complete; skip")
            status["seeds"][str(seed)] = {"status": "completed", "final_test": str(final_path)}
            write_json(status_path, status)
            continue
        print(f"\n[{index}/{len(args.seeds)}] seed {seed}: {seed_args.run_dir}")
        status["seeds"][str(seed)] = {"status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(status_path, status)
        try:
            run_single_seed(seed_args, device)
            status["seeds"][str(seed)] = {"status": "completed",
                                           "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                           "final_test": str(final_path) if final_path.exists() else None}
        except Exception as error:
            status["seeds"][str(seed)] = {"status": "failed", "error": repr(error),
                                           "traceback": traceback.format_exc()}
            write_json(status_path, status)
            if not args.continue_on_seed_error:
                raise
        write_json(status_path, status)
        release_resources()
    if args.stage in ("all", "eval"):
        aggregate(args, require_all=not args.continue_on_seed_error)


if __name__ == "__main__":
    main()
