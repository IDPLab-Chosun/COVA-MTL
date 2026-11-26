"""
COVA-MTL v6.3-ACSR-SRCR (5 streams) for AIDE
=======================================

Budget-free availability-conditioned sufficient routing built on a shared multi-task backbone.
The backbone keeps a shared ResNet-18 with stage attention, PS-MSBN, ETB, and
TSMF. Subset-Conditioned Modality Calibration (SCMC) adapts modality tokens
to the selected stream configuration before mask-aware TSMF. A cheap scout and
Availability-Conditioned Sufficient Router (ACSR) predicts dense-relative
four-task regret and uncertainty for all 31 non-empty subsets. Self-Referenced
Calibrated Routing (SRCR) compares every valid subset with the all-currently-
available reference evaluated by the same router. A subset is mean-admissible
when its predicted average four-task loss change does not exceed the reference.
SRCR selects the smallest mean-admissible subset and uses paired calibrated risk
only to break same-cardinality ties. This removes the over-conservative absolute-
zero upper-bound gate without adding a cost budget, performance-drop threshold,
or modality-frequency target.

Stages
------
1. teacher : train a dense, subset-capable teacher.
2. cache       : enumerate 31 non-empty subsets and cache their task losses.
3. router      : train ACSR with availability-relative regret supervision.
4. calibrate   : calibrate one-sided regret bounds on the validation split.
5. sparse      : fine-tune a sparse student under normal and missing inputs.
6. recalibrate : update calibration using the final sparse student.
7. eval        : compare dense, dynamic, fixed-subset, and failure inference.

v6.3 changes
------------
SRCR replaces the absolute ``upper regret <= 0`` gate with paired, self-
referenced mean admissibility and calibrated-risk tie-breaking. Sparse
exploration uses the next self-referenced mean-admissible subset. The matched
fixed-subset baseline is still selected only on validation data and then
evaluated unchanged on test data.

Examples
--------
Reuse the completed seed-42 COVA/plain checkpoints and export the 2x2 feature
distribution plus standalone PPT assets (the plain model is trained only when
its checkpoint is absent):
python analysis/final_visualization.py

Five-seed training/evaluation (default seeds: 42,43,44,45,46):
python analysis/final_visualization.py --stage all --data-root /path/to/AIDE --run-dir /path/to/RUNS/COVA_v6_3_MultiSeed5

Resume all unfinished seeds:
python analysis/final_visualization.py --stage all --data-root /path/to/AIDE --run-dir /path/to/RUNS/COVA_v6_3_MultiSeed5 --resume

Rebuild only the aggregate files from completed seed results:
python analysis/final_visualization.py --stage aggregate --run-dir /path/to/RUNS/COVA_v6_3_MultiSeed5

Original single-seed behavior:
python analysis/final_visualization.py --no-multi-seed --seed 42 --stage all --data-root /path/to/AIDE --run-dir /path/to/RUNS/COVA_v6_3_seed42

Smoke test:
python analysis/final_visualization.py --stage smoke --no-pretrained

Multi-seed protocol
-------------------
The train/validation/test split is fixed once with --split-seed (default 42).
Only initialization, minibatch shuffling, random subset sampling, availability
dropout, and route exploration change across --seeds. Each seed is stored in
an independent seed_<N> directory. After eval/all, the script writes per-seed
CSV results and mean/std summaries in the parent --run-dir.

Important scope note
--------------------
On AIDE, face/body/scene and posture/gesture are semantic branches derived
from shared raw sources. The reported selection cost is therefore branch-
encoder cost unless external detection/cropping/keypoint extraction is added
to --modality-costs. Do not describe the default experiment as physically
turning five independent sensors on and off.
"""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

try:
    import cv2
except ModuleNotFoundError:  # Model-only smoke tests do not need image decoding.
    cv2 = None


MODALITY_NAMES = ("face", "body", "scene", "posture", "gesture")
TASK_NAMES = ("emotion", "behavior", "scene", "vehicle")
TASK_CLASSES = (5, 7, 3, 5)

EMOTION_LABEL = ["Anxiety", "Peace", "Weariness", "Happiness", "Anger"]
DRIVER_BEHAVIOR_LABEL = [
    "Smoking", "Making Phone", "Looking Around", "Dozing Off",
    "Normal Driving", "Talking", "Body Movement",
]
SCENE_CENTRIC_CONTEXT_LABEL = ["Traffic Jam", "Waiting", "Smooth Traffic"]
VEHICLE_BASED_CONTEXT_LABEL = [
    "Parking", "Turning", "Backward Moving", "Changing Lane", "Forward Moving",
]


# -----------------------------------------------------------------------------
# Reproducibility and data
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_index_df(root: Path, ann_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            ann = ann_dir / f"{p.name}.json"
            if ann.exists():
                rows.append([str(p), str(ann), p.name])
    return pd.DataFrame(rows, columns=["frames_path", "label_path", "clip_id"])


def fixed_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio_in_train: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * train_ratio)
    train_all = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)
    train_all = train_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(train_all) * val_ratio_in_train)
    val = train_all.iloc[:n_val].reset_index(drop=True)
    train = train_all.iloc[n_val:].reset_index(drop=True)
    return train, val, test


def ensure_splits(data_root: Path, split_dir: Path, seed: int) -> None:
    required = [split_dir / f"{s}.csv" for s in ("train", "val", "test")]
    if all(p.exists() for p in required):
        return
    ann_dir = data_root / "annotation"
    if not data_root.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"AIDE path not found: {data_root} / {ann_dir}")
    train, val, test = fixed_splits(build_index_df(data_root, ann_dir), seed=seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame[["frames_path", "label_path"]].to_csv(
            split_dir / f"{name}.csv", index=False, header=False
        )


def base_candidate_indices(total_frames: int, stride: int = 3, target: int = 16) -> List[int]:
    idxs = list(range(0, total_frames, stride))
    if len(idxs) < target and (total_frames - 1) not in idxs:
        idxs.append(total_frames - 1)
    return idxs


def has_valid_result(pose_item: dict) -> bool:
    result = pose_item.get("result")
    return isinstance(result, list) and len(result) > 0


def select_valid_indices(pose_list: list, stride: int = 3, target: int = 16) -> List[int]:
    candidates = base_candidate_indices(len(pose_list), stride, target)
    valid = [i for i in candidates if i < len(pose_list) and has_valid_result(pose_list[i])]
    if len(valid) < target:
        seen = set(valid)
        for i, item in enumerate(pose_list):
            if i not in seen and has_valid_result(item):
                valid.append(i)
                seen.add(i)
            if len(valid) == target:
                break
    if not valid:
        raise RuntimeError("No valid pose result exists in the clip.")
    while len(valid) < target:
        valid.append(valid[-1])
    return valid[:target]


class CarDataset(Dataset):
    """Same tensor/label contract as the backbone data pipeline."""

    def __init__(
        self,
        csv_file: Path,
        t_target: int = 16,
        stride: int = 3,
        face_pad_top: int = 200,
        face_pad_bottom: int = 20,
        face_pad_left: int = 100,
        face_pad_right: int = 50,
    ) -> None:
        self.df = pd.read_csv(csv_file, header=None)
        self.t_target = int(t_target)
        self.stride = int(stride)
        self.face_pad = (
            int(face_pad_top), int(face_pad_bottom),
            int(face_pad_left), int(face_pad_right),
        )

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _list(path: Path) -> List[Path]:
        return sorted(path.glob("*.jpg"), key=lambda x: int(x.stem))

    @staticmethod
    def _rgb_resize(image: np.ndarray, size: int) -> np.ndarray:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _mosaic(self, front: np.ndarray, left: np.ndarray, right: np.ndarray, incar: np.ndarray) -> torch.Tensor:
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        canvas[:112, :112] = self._rgb_resize(front, 112)
        canvas[:112, 112:] = self._rgb_resize(left, 112)
        canvas[112:, :112] = self._rgb_resize(right, 112)
        canvas[112:, 112:] = self._rgb_resize(incar, 112)
        return torch.from_numpy(canvas)

    def __getitem__(self, index: int):
        frames_path, label_path = self.df.iloc[index]
        with open(label_path, "r", encoding="utf-8") as handle:
            ann = json.load(handle)
        pose_list = ann["pose_list"]
        selected = select_valid_indices(pose_list, self.stride, self.t_target)
        views = {
            name: self._list(Path(frames_path) / name)
            for name in ("incarframes", "frontframes", "leftframes", "rightframes")
        }
        for name, files in views.items():
            if len(files) < len(pose_list):
                raise RuntimeError(f"{name} has too few frames: {frames_path}")

        contexts, bodies, faces, keypoints = [], [], [], []
        top, bottom, left_pad, right_pad = self.face_pad
        for frame_index in selected:
            pose = pose_list[frame_index]
            incar = cv2.imread(str(views["incarframes"][frame_index]))
            if incar is None:
                raise RuntimeError(f"Cannot read in-cabin frame: {frames_path}/{frame_index}")
            height, width = incar.shape[:2]
            result = pose["result"][0]
            bx, by, bw, bh = map(int, result["bbox"])
            fx, fy, fw, fh = map(int, result["face_bbox"])
            body_crop = incar[max(0, by):min(height, by + bh), max(0, bx):min(width, bx + bw)]
            face_crop = incar[
                max(0, fy - top):min(height, fy + fh + bottom),
                max(0, fx - left_pad):min(width, fx + fw + right_pad),
            ]
            if body_crop.size == 0:
                body_crop = incar
            if face_crop.size == 0:
                face_crop = body_crop
            bodies.append(torch.from_numpy(self._rgb_resize(body_crop, 224)))
            faces.append(torch.from_numpy(self._rgb_resize(face_crop, 224)))

            front = cv2.imread(str(views["frontframes"][frame_index]))
            left = cv2.imread(str(views["leftframes"][frame_index]))
            right = cv2.imread(str(views["rightframes"][frame_index]))
            if front is None or left is None or right is None:
                raise RuntimeError(f"Cannot read outside view: {frames_path}/{frame_index}")
            contexts.append(self._mosaic(front, left, right, incar))
            keypoints.append(torch.tensor(result["keypoints"], dtype=torch.float32).reshape(-1, 3))

        context = torch.stack(contexts).permute(3, 0, 1, 2).contiguous()
        body = torch.stack(bodies).permute(3, 0, 1, 2).contiguous()
        face = torch.stack(faces).permute(3, 0, 1, 2).contiguous()
        kp = torch.stack(keypoints).permute(2, 0, 1).contiguous().unsqueeze(-1)

        emotion = EMOTION_LABEL.index(ann["emotion_label"].capitalize())
        behavior = DRIVER_BEHAVIOR_LABEL.index(ann["driver_behavior_label"])
        scene = SCENE_CENTRIC_CONTEXT_LABEL.index(ann["scene_centric_context_label"])
        vehicle_name = ann["vehicle_based_context_label"]
        if vehicle_name == "Forward":
            vehicle_name = "Forward Moving"
        vehicle = VEHICLE_BASED_CONTEXT_LABEL.index(vehicle_name)
        return kp, context, body, face, emotion, behavior, scene, vehicle


def build_loaders(args, cache_order: bool = False) -> Dict[str, DataLoader]:
    if cv2 is None:
        raise ModuleNotFoundError(
            "opencv-python is required for AIDE loading. Install it in the training environment."
        )
    # Keep the dataset partition identical across model seeds. Multi-seed
    # statistics should measure training stochasticity, not a mixture of
    # different train/validation/test partitions.
    ensure_splits(args.data_root, args.split_dir, args.split_seed)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CarDataset(
            args.split_dir / f"{split}.csv",
            t_target=args.t_target,
            stride=args.stride,
            face_pad_top=args.face_pad_top,
            face_pad_bottom=args.face_pad_bottom,
            face_pad_left=args.face_pad_left,
            face_pad_right=args.face_pad_right,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train" and not cache_order),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )
    return loaders


def move_batch(batch, device: torch.device):
    kp, ctx, body, face, ye, yb, ys, yv = batch
    return (
        kp.to(device, non_blocking=True),
        (ctx.float() / 255.0).to(device, non_blocking=True),
        (body.float() / 255.0).to(device, non_blocking=True),
        (face.float() / 255.0).to(device, non_blocking=True),
        ye.to(device, non_blocking=True),
        yb.to(device, non_blocking=True),
        ys.to(device, non_blocking=True),
        yv.to(device, non_blocking=True),
    )


# -----------------------------------------------------------------------------
# Shared backbone and COVA-MTL modules
# -----------------------------------------------------------------------------
class IdentityAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(x, 1).flatten(1)
        weight = torch.sigmoid(self.mlp(avg) + self.mlp(maximum))[:, :, None, None]
        return x * weight


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.conv(pooled))


class DirectionalSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.vertical = nn.Conv2d(2, 1, (kernel_size, 1), padding=(pad, 0), bias=False)
        self.horizontal = nn.Conv2d(2, 1, (1, kernel_size), padding=(0, pad), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.vertical(pooled) + self.horizontal(pooled))


class TokenSE(nn.Module):
    def __init__(self, dim: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, dim // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.mlp(x.mean(1)) + self.mlp(x.amax(1)))
        return x * weight[:, None]


class FrameKeypointMLP(nn.Module):
    def __init__(self, keypoints: int, embed: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.keypoints = keypoints
        self.mlp = nn.Sequential(
            nn.Linear(3 * keypoints, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, embed), nn.GELU(),
        )
        self.se = TokenSE(embed)

    def forward(self, kp: torch.Tensor) -> torch.Tensor:
        kp = kp.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        batch, steps = kp.shape[:2]
        return self.se(self.mlp(kp.reshape(batch * steps, -1)).reshape(batch, steps, -1))


# =============================================================================
# COVA-MTL v6: PS-MSBN backbone plus calibrated sufficient routing and SCMC
# =============================================================================
class PSMSBN2d(nn.Module):
    """Partial-shared modality-specific BN of the shared backbone."""

    def __init__(self, reference: nn.BatchNorm2d, domains: int = 3, mix: float = 0.2):
        super().__init__()
        self.domains = int(domains)
        self.channels = int(reference.num_features)
        self.eps = float(reference.eps)
        self.momentum = reference.momentum
        self.weight = nn.Parameter(reference.weight.detach()[None].repeat(domains, 1))
        self.bias = nn.Parameter(reference.bias.detach()[None].repeat(domains, 1))
        self.register_buffer(
            "running_mean_dom", reference.running_mean.detach()[None].repeat(domains, 1)
        )
        self.register_buffer(
            "running_var_dom", reference.running_var.detach()[None].repeat(domains, 1)
        )
        self.register_buffer("running_mean_global", reference.running_mean.detach().clone())
        self.register_buffer("running_var_global", reference.running_var.detach().clone())
        self.register_buffer("batches_dom", torch.zeros(domains, dtype=torch.long))
        self.register_buffer("batches_global", torch.zeros((), dtype=torch.long))
        mix = float(np.clip(mix, 1e-4, 1.0 - 1e-4))
        self.mix_logit = nn.Parameter(
            torch.full((self.channels,), math.log(mix / (1.0 - mix)))
        )

    def _factor(self, tracked: torch.Tensor) -> float:
        return 1.0 / float(tracked.item()) if self.momentum is None else float(self.momentum)

    def forward(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        domain = int(domain)
        if not 0 <= domain < self.domains:
            raise ValueError(f"invalid PS-MSBN modality {domain}")
        if self.training:
            xf = x.float()
            local_mean = xf.mean((0, 2, 3))
            local_var = xf.var((0, 2, 3), unbiased=False)
            global_mean = self.running_mean_global.detach().clone().float()
            global_var = self.running_var_global.detach().clone().float()
            with torch.no_grad():
                self.batches_dom[domain] += 1
                self.batches_global += 1
                fd = self._factor(self.batches_dom[domain])
                fg = self._factor(self.batches_global)
                self.running_mean_dom[domain].lerp_(local_mean.detach(), fd)
                self.running_var_dom[domain].lerp_(local_var.detach(), fd)
                self.running_mean_global.lerp_(local_mean.detach(), fg)
                self.running_var_global.lerp_(local_var.detach(), fg)
        else:
            local_mean = self.running_mean_dom[domain].detach().clone().float()
            local_var = self.running_var_dom[domain].detach().clone().float()
            global_mean = self.running_mean_global.detach().clone().float()
            global_var = self.running_var_global.detach().clone().float()
        mix = torch.sigmoid(self.mix_logit).float()
        mean = (1.0 - mix) * local_mean + mix * global_mean
        var = (1.0 - mix) * local_var + mix * global_var
        out = (x.float() - mean[None, :, None, None]) / torch.sqrt(
            var[None, :, None, None] + self.eps
        )
        out = out.to(x.dtype)
        return out * self.weight[domain][None, :, None, None].to(x.dtype) + self.bias[
            domain
        ][None, :, None, None].to(x.dtype)


class DomainIdentity(nn.Module):
    def forward(self, x: torch.Tensor, domain: int = 0) -> torch.Tensor:
        return x


class PairDeltaFusion1D(nn.Module):
    """ETB for keypoint sequences: pairwise 16 -> 8 temporal fusion."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.delta = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(4 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        if steps % 2:
            raise ValueError("ETB requires an even number of frames")
        paired = x.reshape(batch, steps // 2, 2, dim)
        first, second = paired[:, :, 0], paired[:, :, 1]
        mean, difference = 0.5 * (first + second), 0.5 * (second - first)
        mf, df = mean.float(), difference.float()
        stats = torch.cat((mf, mf.abs(), (mf.square() + 1e-6).sqrt(), df.abs()), -1)
        gate = torch.sigmoid(self.gate(torch.nan_to_num(stats))).to(x.dtype)
        return mean + gate * self.delta(difference)


class PairDeltaFusion2D(nn.Module):
    """ETB for stage-2 visual maps: pairwise 16 -> 8 temporal fusion."""

    def __init__(self, channels: int = 128, hidden: int = 64):
        super().__init__()
        self.channels = channels
        self.delta = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(4 * channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        if steps % 2:
            raise ValueError("ETB requires an even number of frames")
        paired = x.reshape(batch, steps // 2, 2, channels, height, width)
        first, second = paired[:, :, 0], paired[:, :, 1]
        mean, difference = 0.5 * (first + second), 0.5 * (second - first)
        mu = mean.mean((3, 4), dtype=torch.float32)
        second_moment = (mean * mean).mean((3, 4), dtype=torch.float32)
        std = (second_moment - mu.square()).clamp_min(0.0).add(1e-6).sqrt()
        absolute = mean.abs().mean((3, 4), dtype=torch.float32)
        rms = second_moment.add(1e-6).sqrt()
        gate = torch.sigmoid(self.gate(torch.cat((mu, std, absolute, rms), -1))).to(x.dtype)
        gate = gate[:, :, :, None, None]
        delta = self.delta(difference.reshape(-1, channels, height, width)).reshape_as(difference)
        return mean + gate * delta


class SharedResNet18PSMSBN(nn.Module):
    """Shared ResNet-18 with stage attention and PS-MSBN."""

    def __init__(self, pretrained: bool = True, train_base: bool = True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1, self.relu, self.maxpool = base.conv1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.stem_bn = PSMSBN2d(base.bn1)
        self.attn = nn.ModuleList(
            (IdentityAttention(), DirectionalSpatialAttention(7), SpatialAttention(7), ChannelAttention(512, 16))
        )

        def domain_bns(layer):
            bn1 = nn.ModuleList([PSMSBN2d(block.bn1) for block in layer])
            bn2 = nn.ModuleList([PSMSBN2d(block.bn2) for block in layer])
            down = nn.ModuleList([
                PSMSBN2d(block.downsample[1])
                if block.downsample is not None and len(block.downsample) > 1
                else DomainIdentity()
                for block in layer
            ])
            return bn1, bn2, down

        self.l1_bn1, self.l1_bn2, self.l1_down = domain_bns(self.layer1)
        self.l2_bn1, self.l2_bn2, self.l2_down = domain_bns(self.layer2)
        self.l3_bn1, self.l3_bn2, self.l3_down = domain_bns(self.layer3)
        self.l4_bn1, self.l4_bn2, self.l4_down = domain_bns(self.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if not train_base:
            for module in (self.conv1, self.layer1, self.layer2, self.layer3, self.layer4):
                module.requires_grad_(False)

    @staticmethod
    def _shortcut(block, x, down_bn, domain):
        if block.downsample is None:
            return x
        return down_bn(block.downsample[0](x), domain)

    def _basic(self, block, x, bn1, bn2, down, domain):
        residual = self._shortcut(block, x, down, domain)
        out = block.relu(bn1(block.conv1(x), domain))
        out = bn2(block.conv2(out), domain)
        return block.relu(out + residual)

    def stage12(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        x = self.maxpool(self.relu(self.stem_bn(self.conv1(x), domain)))
        for i, block in enumerate(self.layer1):
            x = self.attn[0](self._basic(block, x, self.l1_bn1[i], self.l1_bn2[i], self.l1_down[i], domain))
        for i, block in enumerate(self.layer2):
            x = self.attn[1](self._basic(block, x, self.l2_bn1[i], self.l2_bn2[i], self.l2_down[i], domain))
        return x

    def stage34(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        for i, block in enumerate(self.layer3):
            x = self.attn[2](self._basic(
                block, x, self.l3_bn1[i], self.l3_bn2[i], self.l3_down[i], domain
            ))
        for i, block in enumerate(self.layer4):
            x = self.attn[3](self._basic(
                block, x, self.l4_bn1[i], self.l4_bn2[i], self.l4_down[i], domain
            ))
        return x


class MultiStreamScout(nn.Module):
    """Cheap per-stream previews used before any heavy branch is executed."""

    def __init__(self, embed: int = 128, frames: int = 4, size: int = 64):
        super().__init__()
        self.frames, self.size = int(frames), int(size)
        self.visual = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.visual_proj = nn.ModuleList([nn.Linear(64, embed) for _ in range(3)])
        # Positions alone make a preview nearly static. Concatenating first
        # differences exposes inexpensive posture/gesture motion without a
        # heavy video encoder (3 coordinates + 3 temporal differences).
        self.posture_proj = nn.Sequential(nn.Linear(6 * 26, 64), nn.GELU(), nn.Linear(64, embed))
        self.gesture_proj = nn.Sequential(nn.Linear(6 * 42, 64), nn.GELU(), nn.Linear(64, embed))
        self.temporal_score = nn.ModuleList([nn.Linear(embed, 1) for _ in range(5)])
        self.type_embed = nn.Parameter(torch.randn(5, embed) * 0.02)

    def _indices(self, steps: int, device: torch.device) -> torch.Tensor:
        count = min(self.frames, steps)
        return torch.linspace(0, steps - 1, count, device=device).round().long()

    def _visual_token(self, x: torch.Tensor, stream: int, indices: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = x.shape
        preview = x.index_select(2, indices).permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)
        preview = F.interpolate(preview, (self.size, self.size), mode="bilinear", align_corners=False)
        sequence = self.visual_proj[stream](self.visual(preview).flatten(1)).reshape(batch, -1, self.type_embed.size(1))
        weight = torch.softmax(self.temporal_score[stream](sequence).squeeze(-1), 1)
        return (sequence * weight[:, :, None]).sum(1)

    def _keypoint_token(self, kp, start, end, stream, projection, indices):
        chosen = kp[:, :, :, start:end, :].index_select(2, indices).squeeze(-1)
        sequence = chosen.permute(0, 2, 1, 3).reshape(chosen.size(0), len(indices), -1)
        velocity = torch.diff(sequence, dim=1, prepend=sequence[:, :1])
        sequence = torch.cat((sequence, velocity), dim=-1)
        sequence = projection(sequence)
        weight = torch.softmax(self.temporal_score[stream](sequence).squeeze(-1), 1)
        return (sequence * weight[:, :, None]).sum(1)

    def forward(self, kp, context, body, face) -> torch.Tensor:
        indices = self._indices(context.size(2), context.device)
        tokens = (
            self._visual_token(face, 0, indices),
            self._visual_token(body, 1, indices),
            self._visual_token(context, 2, indices),
            self._keypoint_token(kp, 0, 26, 3, self.posture_proj, indices),
            self._keypoint_token(kp, 94, 136, 4, self.gesture_proj, indices),
        )
        return torch.stack(tokens, 1) + self.type_embed[None]


class SubsetConditionedModalityCalibration(nn.Module):
    """Identity-initialized token calibration conditioned on the active subset."""

    def __init__(self, embed: int = 128, hidden: int = 128, strength: float = 0.25):
        super().__init__()
        self.embed = int(embed)
        self.strength = float(strength)
        self.token_norm = nn.LayerNorm(embed)
        self.configuration = nn.Sequential(
            nn.Linear(embed + 5, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.modality_embedding = nn.Parameter(torch.randn(5, hidden) * 0.02)
        self.affine = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * embed),
        )
        # SCMC begins as an exact identity mapping. This permits initialization
        # from an earlier checkpoint even though the new calibration did not exist.
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(
        self,
        modality_tokens: torch.Tensor,
        scout_tokens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = active_mask.bool()
        if (~mask).all(1).any():
            raise ValueError("SCMC requires at least one active modality per sample")
        mask_value = mask.to(scout_tokens.dtype)
        selected_preview = (
            scout_tokens * mask_value[:, :, None]
        ).sum(1) / mask_value.sum(1, keepdim=True).clamp_min(1.0)
        configuration = self.configuration(
            torch.cat((selected_preview, mask_value), dim=1)
        )
        condition = configuration[:, None] + self.modality_embedding[None]
        delta_scale, delta_shift = self.affine(condition).chunk(2, dim=-1)
        delta_scale = self.strength * torch.tanh(delta_scale)
        delta_shift = self.strength * torch.tanh(delta_shift)
        calibrated = (
            modality_tokens
            + delta_scale[:, None] * self.token_norm(modality_tokens)
            + delta_shift[:, None]
        )
        return calibrated * mask_value[:, None, :, None]


class MaskAwareTSMF(nn.Module):
    """Original task-token fusion with absent streams hard-masked from attention."""

    def __init__(self, embed=128, d_model=256, heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.project = nn.Linear(embed, d_model)
        self.task_tokens = nn.Parameter(torch.randn(4, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, return_attention=False):
        batch, steps, modalities, _ = tokens.shape
        if (~mask.bool()).all(1).any():
            raise ValueError("TSMF requires at least one available modality per sample")
        kv = self.norm1(self.project(tokens).reshape(batch * steps, modalities, self.d_model))
        query = self.task_tokens[None].expand(batch * steps, -1, -1)
        key_padding = (~mask.bool())[:, None].expand(batch, steps, modalities).reshape(batch * steps, modalities)
        attended, weights = self.attention(
            self.norm1(query), kv, kv, key_padding_mask=key_padding,
            need_weights=return_attention, average_attn_weights=True
        )
        output = query + attended
        output = output + self.ffn(self.norm2(output))
        output = output.reshape(batch, steps, 4, self.d_model).permute(0, 2, 1, 3)
        if return_attention:
            return output, weights.reshape(batch, steps, 4, modalities)
        return output


class COVAMTLSCMC(nn.Module):
    """Budget-free COVA-MTL with PS-MSBN and subset-conditioned calibration."""

    def __init__(self, embed=128, d_model=256, pretrained=True, dropout=0.1,
                 preview_frames=4, preview_size=64, temporal_steps=16,
                 attention_heads=4, train_base=True, use_scmc=True,
                 scmc_hidden=128, scmc_strength=0.25):
        super().__init__()
        if temporal_steps % 2:
            raise ValueError("temporal_steps must be even")
        self.embed, self.d_model, self.temporal_steps = embed, d_model, temporal_steps
        self.reduced_steps = temporal_steps // 2
        self.backbone = SharedResNet18PSMSBN(pretrained, train_base)
        self.projections = nn.ModuleList([nn.Linear(512, embed) for _ in range(3)])
        self.posture = FrameKeypointMLP(26, embed)
        self.gesture = FrameKeypointMLP(42, embed)
        self.visual_etb = nn.ModuleList([PairDeltaFusion2D() for _ in range(3)])
        self.posture_etb, self.gesture_etb = PairDeltaFusion1D(embed), PairDeltaFusion1D(embed)
        self.use_scmc = bool(use_scmc)
        self.scmc = (
            SubsetConditionedModalityCalibration(
                embed=embed, hidden=scmc_hidden, strength=scmc_strength
            )
            if self.use_scmc else None
        )
        self.tsmf = MaskAwareTSMF(embed, d_model, attention_heads, dropout)
        self.pool_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, classes) for classes in TASK_CLASSES])
        self.scout = MultiStreamScout(embed, preview_frames, preview_size)

    def encode_modality(self, modality_id, kp, context, body, face):
        batch, steps = kp.size(0), kp.size(2)
        if steps != self.temporal_steps:
            raise ValueError(f"model expects {self.temporal_steps} frames, got {steps}")
        if modality_id < 3:
            visual = (face, body, context)[modality_id]
            _, _, _, height, width = visual.shape
            frames = visual.permute(0, 2, 1, 3, 4).reshape(batch * steps, 3, height, width)
            stage2 = self.backbone.stage12(frames, modality_id)
            _, channels, h2, w2 = stage2.shape
            stage2 = self.visual_etb[modality_id](stage2.reshape(batch, steps, channels, h2, w2))
            stage4 = self.backbone.stage34(stage2.reshape(-1, channels, h2, w2), modality_id)
            pooled = self.backbone.pool(stage4).flatten(1)
            return self.projections[modality_id](pooled).reshape(batch, self.reduced_steps, self.embed)
        if modality_id == 3:
            return self.posture_etb(self.posture(kp[:, :, :, :26, :]))
        if modality_id == 4:
            return self.gesture_etb(self.gesture(kp[:, :, :, 94:, :]))
        raise ValueError(f"unknown modality {modality_id}")

    def encode_all(self, kp, context, body, face):
        scout_tokens = self.scout(kp, context, body, face)
        order = torch.randperm(3).tolist() if self.training else [0, 1, 2]
        encoded = [None] * 5
        for modality in order:
            encoded[modality] = self.encode_modality(modality, kp, context, body, face)
        encoded[3] = self.encode_modality(3, kp, context, body, face)
        encoded[4] = self.encode_modality(4, kp, context, body, face)
        return scout_tokens, torch.stack(encoded, 2)

    def encode_selected(self, mask, kp, context, body, face):
        batch = mask.size(0)
        streams = []
        for modality in range(5):
            indices = torch.nonzero(mask[:, modality], as_tuple=False).flatten()
            if indices.numel() == 0:
                stream = context.new_zeros(batch, self.reduced_steps, self.embed)
            else:
                sequence = self.encode_modality(
                    modality, kp.index_select(0, indices), context.index_select(0, indices),
                    body.index_select(0, indices), face.index_select(0, indices)
                )
                stream = sequence.new_zeros(batch, self.reduced_steps, self.embed)
                stream = stream.index_copy(0, indices, sequence)
            streams.append(stream)
        return torch.stack(streams, 2)

    def predict(self, scout_tokens, tokens, mask, return_attention=False):
        if self.use_scmc:
            tokens = self.scmc(tokens, scout_tokens, mask)
        else:
            tokens = tokens * mask[:, None, :, None].to(tokens.dtype)
        fused = self.tsmf(tokens, mask, return_attention)
        attention = None
        if return_attention:
            fused, attention = fused
        features = self.pool_norm(fused.mean(2))
        logits = tuple(head(features[:, i]) for i, head in enumerate(self.heads))
        state = features.mean(1)
        if return_attention:
            return logits, state, features, attention
        return logits, state, features

    def forward_dense(self, kp, context, body, face):
        scout, tokens = self.encode_all(kp, context, body, face)
        mask = torch.ones(kp.size(0), 5, dtype=torch.bool, device=kp.device)
        return self.predict(scout, tokens, mask)[0]


class AvailabilityConditionedSufficientRouter(nn.Module):
    """Predicts task-wise regret distributions for all non-empty subsets.

    The target is relative to the set of streams that is physically available
    for the current sample. Explicit availability conditioning separates
    "not selected" from "not observable" and lets one 31-subset cache supervise
    normal, single-failure, and multiple-failure scenarios.
    """

    def __init__(self, scout_dim: int = 128, hidden: int = 256):
        super().__init__()
        token_dim = max(hidden // 2, 32)
        self.token_dim = token_dim
        self.token = nn.Sequential(
            nn.LayerNorm(scout_dim), nn.Linear(scout_dim, token_dim), nn.GELU()
        )
        self.modality_embedding = nn.Parameter(torch.randn(5, token_dim) * 0.02)
        self.availability_embedding = nn.Parameter(torch.randn(2, token_dim) * 0.02)
        candidate_masks = torch.tensor(
            [
                [bool(subset & (1 << modality)) for modality in range(5)]
                for subset in range(1, 32)
            ],
            dtype=torch.bool,
        )
        self.register_buffer("candidate_masks", candidate_masks, persistent=True)
        self.register_buffer(
            "candidate_cardinality", candidate_masks.sum(1), persistent=True
        )
        # Set by validation calibration. It is intentionally not part of the
        # checkpoint because teacher and sparse-student calibrations differ.
        # Rows condition on the number of physically available streams and
        # columns on the number of selected streams. This prevents severe
        # failure cases from making normal full-availability routing needlessly
        # conservative.
        self.register_buffer("calibration_q", torch.ones(5, 5), persistent=False)
        # Five identity-preserving slots, selected/remaining/available means,
        # candidate bits, availability bits, and two cardinality ratios.
        feature_dim = 8 * token_dim + 12
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 8),
        )

    def forward(
        self, scout_tokens: torch.Tensor, available: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scout_tokens.ndim != 3 or scout_tokens.size(1) != 5:
            raise ValueError("ACSR expects scout tokens with shape [batch, 5, dim]")
        if available.shape != scout_tokens.shape[:2]:
            raise ValueError("available must have shape [batch, 5]")
        available = available.bool()
        if (~available).all(1).any():
            raise ValueError("Every sample needs at least one available stream")
        availability_index = available.long()
        encoded = (
            self.token(scout_tokens)
            + self.modality_embedding[None]
            + self.availability_embedding[availability_index]
        )
        encoded = encoded * available[:, :, None].to(encoded.dtype)
        candidate = self.candidate_masks.to(encoded.dtype)
        available_value = available.to(encoded.dtype)
        selected_mask = candidate[None] * available_value[:, None]
        remaining_mask = (1.0 - candidate)[None] * available_value[:, None]
        selected = encoded[:, None] * selected_mask[:, :, :, None]
        remaining = encoded[:, None] * remaining_mask[:, :, :, None]
        available_tokens = encoded * available_value[:, :, None]
        selected_count = selected_mask.sum(2, keepdim=True).clamp_min(1.0)
        remaining_count = remaining_mask.sum(2, keepdim=True).clamp_min(1.0)
        available_count = available_value.sum(1, keepdim=True).clamp_min(1.0)
        selected_mean = selected.sum(2) / selected_count
        remaining_mean = remaining.sum(2) / remaining_count
        available_mean = (
            available_tokens.sum(1) / available_count
        )[:, None].expand(-1, 31, -1)
        cardinality = candidate.sum(1, keepdim=True)
        features = torch.cat(
            (
                selected.flatten(2), selected_mean, remaining_mean, available_mean,
                candidate[None].expand(encoded.size(0), -1, -1),
                available_value[:, None].expand(-1, 31, -1),
                (cardinality / 5.0)[None].expand(encoded.size(0), -1, -1),
                (available_count / 5.0)[:, None].expand(-1, 31, -1),
            ),
            dim=-1,
        )
        raw = self.net(features)
        mean, raw_scale = raw.chunk(2, dim=-1)
        scale = F.softplus(raw_scale.float()).to(mean.dtype).clamp_min(1e-4)
        return mean, scale


# -----------------------------------------------------------------------------
# Losses, routing, and metrics
# -----------------------------------------------------------------------------
def labels_tuple(batch_on_device) -> Tuple[torch.Tensor, ...]:
    return tuple(batch_on_device[4:8])


def multitask_ce(logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum(F.cross_entropy(logit, target) for logit, target in zip(logits, labels))


def per_task_ce(logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [F.cross_entropy(logit, target, reduction="none") for logit, target in zip(logits, labels)], dim=1
    )


def normalized_per_task_ce(
    logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Comparable CE scale across tasks with different class counts."""
    return torch.stack(
        [
            F.cross_entropy(logit, target, reduction="none") / math.log(classes)
            for logit, target, classes in zip(logits, labels, TASK_CLASSES)
        ],
        dim=1,
    )


def normalized_entropy(logits: Sequence[torch.Tensor]) -> torch.Tensor:
    values = []
    for logit, classes in zip(logits, TASK_CLASSES):
        prob = torch.softmax(logit.float(), dim=1).clamp_min(1e-8)
        values.append(-(prob * prob.log()).sum(1) / math.log(classes))
    return torch.stack(values, dim=1)


def confidence_gated_kd(
    student: Sequence[torch.Tensor],
    teacher: Sequence[torch.Tensor],
    temperature: float = 2.0,
    confidence_threshold: float = 0.6,
) -> torch.Tensor:
    total = student[0].new_zeros(())
    used = 0
    for s_logit, t_logit in zip(student, teacher):
        confidence = torch.softmax(t_logit.detach().float(), dim=1).amax(1)
        valid = confidence >= confidence_threshold
        if valid.any():
            target = torch.softmax(t_logit.detach()[valid] / temperature, dim=1)
            loss = F.kl_div(
                torch.log_softmax(s_logit[valid] / temperature, dim=1),
                target,
                reduction="batchmean",
            ) * (temperature ** 2)
            total = total + loss
            used += 1
    return total / max(used, 1)


def mask_table(device: Optional[torch.device] = None) -> torch.Tensor:
    table = [[bool(mask & (1 << m)) for m in range(5)] for mask in range(32)]
    return torch.tensor(table, dtype=torch.bool, device=device)


def sample_balanced_masks(batch: int, device: torch.device, include_full: bool = True) -> torch.Tensor:
    masks = torch.zeros(batch, 5, dtype=torch.bool, device=device)
    maximum = 5 if include_full else 4
    # Empty input has no meaning for stage attention. Train cardinalities 1..5.
    cardinalities = torch.randint(1, maximum + 1, (batch,), device=device)
    random_score = torch.rand(batch, 5, device=device)
    order = random_score.argsort(dim=1)
    for i in range(batch):
        masks[i, order[i, : int(cardinalities[i])]] = True
    return masks


def masks_to_subset_ids(mask: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([1, 2, 4, 8, 16], device=mask.device, dtype=torch.long)
    return (mask.long() * weights).sum(1)


def availability_relative_targets(
    subset_loss: torch.Tensor,
    available: torch.Tensor,
    candidate_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return regret to the all-available baseline and valid candidates."""
    available = available.bool()
    baseline_ids = masks_to_subset_ids(available)
    rows = torch.arange(subset_loss.size(0), device=subset_loss.device)
    baseline = subset_loss[rows, baseline_ids]
    regret = subset_loss[:, 1:32] - baseline[:, None]
    valid = ~(
        candidate_masks[None] & (~available)[:, None]
    ).any(-1)
    return regret, valid


def predicted_regret_statistics(
    task_mean: torch.Tensor, task_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = task_mean.float().mean(-1)
    # Root-sum-square propagation is conservative relative to simply averaging
    # the four predicted task scales and remains differentiable.
    scale = task_scale.float().square().sum(-1).sqrt() / task_scale.size(-1)
    return mean, scale.clamp_min(1e-4)


def calibrated_relative_upper_regret(
    router: AvailabilityConditionedSufficientRouter,
    task_mean: torch.Tensor,
    task_scale: torch.Tensor,
    available: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return paired candidate-minus-reference mean, scale, and upper bound."""
    mean, scale = predicted_regret_statistics(task_mean, task_scale)
    rows = torch.arange(mean.size(0), device=mean.device)
    reference_candidate = masks_to_subset_ids(available) - 1
    reference_mean = mean[rows, reference_candidate]
    reference_scale = scale[rows, reference_candidate]
    relative_mean = mean - reference_mean[:, None]
    # A paired covariance is not available from the router. RSS is used as a
    # stable normalization; the validation quantile below calibrates its scale.
    relative_scale = (
        scale.square() + reference_scale[:, None].square()
    ).sqrt().clamp_min(1e-4)
    availability_cardinality = available.sum(1).long() - 1
    candidate_cardinality = router.candidate_cardinality.long() - 1
    q = router.calibration_q.to(mean.device)[
        availability_cardinality[:, None], candidate_cardinality[None]
    ]
    relative_upper = relative_mean + q * relative_scale
    # The reference is the decision baseline, not a candidate that has to prove
    # superiority over itself. Pinning its paired difference to zero guarantees
    # a valid route whenever at least one stream is physically available.
    relative_mean = relative_mean.clone()
    relative_upper = relative_upper.clone()
    relative_mean[rows, reference_candidate] = 0.0
    relative_upper[rows, reference_candidate] = 0.0
    return relative_mean, relative_scale, relative_upper, mean, scale


# Budget-free, self-referenced calibrated routing.
@torch.no_grad()
def select_self_referenced_subset(
    router: AvailabilityConditionedSufficientRouter,
    scout_tokens: torch.Tensor,
    available: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Choose the smallest mean-admissible subset; use calibrated risk as tie-break."""
    batch = scout_tokens.size(0)
    if available is None:
        available = torch.ones(batch, 5, dtype=torch.bool, device=scout_tokens.device)
    else:
        available = available.bool()
    if (~available).all(1).any():
        raise ValueError("Every sample needs at least one physically available stream.")

    routed_scout = scout_tokens * available[:, :, None].to(scout_tokens.dtype)
    predicted_task_regret, predicted_task_scale = router(routed_scout, available)
    candidate_masks = router.candidate_masks
    valid = ~(
        candidate_masks[None] & (~available)[:, None]
    ).any(-1)
    (
        relative_mean,
        relative_scale,
        relative_upper,
        predicted_mean,
        predicted_scale,
    ) = calibrated_relative_upper_regret(
        router, predicted_task_regret, predicted_task_scale, available
    )
    relative_upper = relative_upper.masked_fill(~valid, torch.inf)
    fallback_candidate = masks_to_subset_ids(available) - 1
    rows = torch.arange(batch, device=scout_tokens.device)
    reference_mean = predicted_mean[rows, fallback_candidate]

    # The paired candidate-minus-reference difference cancels common router
    # bias. Mean admissibility is the selection gate. The calibrated upper bound
    # is deliberately a same-cardinality tie-break, because using it again as a
    # hard zero gate caused the empirically observed all-modalities collapse.
    # The reference is explicitly pinned to zero above.
    finite = torch.isfinite(relative_mean) & torch.isfinite(relative_upper)
    admissible = (
        valid
        & finite
        & (relative_mean <= 0.0)
    )
    cardinality = router.candidate_cardinality[None].expand(batch, -1)
    minimum_cardinality = cardinality.masked_fill(~admissible, 6).min(1).values
    eligible = admissible & (cardinality == minimum_cardinality[:, None])
    admissible_candidate = relative_upper.masked_fill(~eligible, torch.inf).argmin(1)
    has_admissible = admissible.any(1)
    best_candidate = torch.where(
        has_admissible, admissible_candidate, fallback_candidate
    )
    # The sparse student also receives the next smallest self-referenced
    # admissible configuration. It is a counterfactual sibling, not a random
    # subset and never introduces a cost or modality-frequency target.
    chosen = torch.arange(31, device=scout_tokens.device)[None] == best_candidate[:, None]
    alternative_pool = admissible & (~chosen)
    alternative_cardinality = cardinality.masked_fill(~alternative_pool, 6).min(1).values
    alternative_eligible = alternative_pool & (
        cardinality == alternative_cardinality[:, None]
    )
    alternative_candidate = relative_upper.masked_fill(
        ~alternative_eligible, torch.inf
    ).argmin(1)
    has_alternative = alternative_pool.any(1)
    alternative_candidate = torch.where(
        has_alternative, alternative_candidate, best_candidate
    )
    selected_mask = candidate_masks.index_select(0, best_candidate)
    alternative_mask = candidate_masks.index_select(0, alternative_candidate)
    selected_task_regret = predicted_task_regret.float()[rows, best_candidate]
    selected_task_scale = predicted_task_scale.float()[rows, best_candidate]
    selected_mean = predicted_mean[rows, best_candidate]
    selected_relative_mean = relative_mean[rows, best_candidate]
    selected_relative_scale = relative_scale[rows, best_candidate]
    selected_relative_upper = relative_upper[rows, best_candidate]
    selected_full_available = best_candidate.eq(fallback_candidate)
    confidence_supported = selected_relative_upper <= 0.0
    return {
        "mask": selected_mask,
        "subset_id": best_candidate + 1,
        "available": available,
        "predicted_task_regret": selected_task_regret,
        "predicted_task_scale": selected_task_scale,
        "predicted_mean_regret": selected_mean,
        "reference_mean_regret": reference_mean,
        "predicted_relative_mean_regret": selected_relative_mean,
        "predicted_relative_scale": selected_relative_scale,
        "predicted_relative_upper_regret": selected_relative_upper,
        "confidence_supported": confidence_supported,
        "reference_admissible": has_admissible,
        "selected_full_available": selected_full_available,
        "fallback": ~has_admissible,
        "alternative_mask": alternative_mask,
        "alternative_subset_id": alternative_candidate + 1,
        "has_admissible_alternative": has_alternative,
    }


@torch.no_grad()
def route_from_precomputed(
    model: COVAMTLSCMC,
    router: AvailabilityConditionedSufficientRouter,
    scout_tokens: torch.Tensor,
    all_tokens: torch.Tensor,
    available: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    route = select_self_referenced_subset(router, scout_tokens, available)
    logits, state, task_features = model.predict(scout_tokens, all_tokens, route["mask"])
    route.update({"logits": logits, "state": state, "task_features": task_features, "tokens": all_tokens})
    return route


@torch.no_grad()
def route_from_raw(
    model: COVAMTLSCMC,
    router: AvailabilityConditionedSufficientRouter,
    kp: torch.Tensor,
    context: torch.Tensor,
    body: torch.Tensor,
    face: torch.Tensor,
    available: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # Crucially, no heavy feature is computed before this decision.
    scout_tokens = model.scout(kp, context, body, face)
    route = select_self_referenced_subset(router, scout_tokens, available)
    selected_tokens = model.encode_selected(route["mask"], kp, context, body, face)
    logits, state, task_features = model.predict(scout_tokens, selected_tokens, route["mask"])
    route.update({
        "logits": logits, "state": state, "task_features": task_features,
        "tokens": selected_tokens, "scout_tokens": scout_tokens,
    })
    return route


def update_predictions(store: dict, logits: Sequence[torch.Tensor], labels: Sequence[torch.Tensor]) -> None:
    for task_id in range(4):
        store["true"][task_id].extend(labels[task_id].detach().cpu().tolist())
        store["pred"][task_id].extend(logits[task_id].argmax(1).detach().cpu().tolist())


def finish_metrics(store: dict) -> dict:
    acc, mf1 = {}, {}
    for task_id, name in enumerate(TASK_NAMES):
        acc[name] = accuracy_score(store["true"][task_id], store["pred"][task_id])
        mf1[name] = f1_score(store["true"][task_id], store["pred"][task_id], average="macro")
    return {
        "acc": acc, "f1": mf1,
        "macc": float(np.mean(list(acc.values()))),
        "mf1": float(np.mean(list(mf1.values()))),
    }


def new_prediction_store() -> dict:
    return {"true": {i: [] for i in range(4)}, "pred": {i: [] for i in range(4)}}


# -----------------------------------------------------------------------------
# Checkpoints and optional compatible initialization
# -----------------------------------------------------------------------------
def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, scaler, best: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "best": best,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scaler=None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def initialize_from_checkpoint(model: COVAMTLSCMC, checkpoint_path: Path) -> None:
    """Shape-compatible initialization, including renamed backbone components."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    copied = {}
    replacements = [
        ("backbone.dsbn_stem.", "backbone.stem_bn."),
        ("backbone.dsbn_l1_bn1.", "backbone.l1_bn1."),
        ("backbone.dsbn_l1_bn2.", "backbone.l1_bn2."),
        ("backbone.dsbn_l1_down.", "backbone.l1_down."),
        ("backbone.dsbn_l2_bn1.", "backbone.l2_bn1."),
        ("backbone.dsbn_l2_bn2.", "backbone.l2_bn2."),
        ("backbone.dsbn_l2_down.", "backbone.l2_down."),
        ("backbone.dsbn_l3_bn1.", "backbone.l3_bn1."),
        ("backbone.dsbn_l3_bn2.", "backbone.l3_bn2."),
        ("backbone.dsbn_l3_down.", "backbone.l3_down."),
        ("backbone.dsbn_l4_bn1.", "backbone.l4_bn1."),
        ("backbone.dsbn_l4_bn2.", "backbone.l4_bn2."),
        ("backbone.dsbn_l4_down.", "backbone.l4_down."),
        ("backbone.attn1.", "backbone.attn.0."),
        ("backbone.attn2.", "backbone.attn.1."),
        ("backbone.attn3.", "backbone.attn.2."),
        ("backbone.attn4.", "backbone.attn.3."),
        ("fp.", "projections.0."), ("bp.", "projections.1."), ("sp.", "projections.2."),
        ("post.", "posture."), ("gest.", "gesture."),
        ("feat2_fuser.", "visual_etb."),
        ("post_fuser.", "posture_etb."), ("gest_fuser.", "gesture_etb."),
        ("tsmf_final.", "tsmf."),
        ("head_emotion.", "heads.0."), ("head_behavior.", "heads.1."),
        ("head_scene.", "heads.2."), ("head_vehicle.", "heads.3."),
    ]
    suffix_replacements = [
        (".lambda_logit", ".mix_logit"),
        (".running_mean_g", ".running_mean_global"),
        (".running_var_g", ".running_var_global"),
        (".num_batches_tracked_dom", ".batches_dom"),
        (".num_batches_tracked_g", ".batches_global"),
    ]
    for old_key, value in source.items():
        new_key = old_key.removeprefix("module.")
        for old_prefix, new_prefix in replacements:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
        for old_suffix, new_suffix in suffix_replacements:
            new_key = new_key.replace(old_suffix, new_suffix)
        if new_key.startswith(("visual_etb.", "posture_etb.", "gesture_etb.")):
            new_key = new_key.replace(".delta_proj.", ".delta.").replace(
                ".gate_mlp.", ".gate."
            )
        if new_key.startswith("tsmf."):
            new_key = new_key.replace(".mod_proj.", ".project.").replace(
                ".attn.", ".attention."
            )
        if new_key.startswith("backbone.attn.1."):
            new_key = new_key.replace(".conv_v.", ".vertical.").replace(
                ".conv_h.", ".horizontal."
            )
        if new_key in target and target[new_key].shape == value.shape:
            copied[new_key] = value
    target.update(copied)
    model.load_state_dict(target)
    print(f"Compatible initialization: {len(copied)} tensors loaded from {checkpoint_path}")


def model_from_args(args, device: torch.device) -> COVAMTLSCMC:
    return COVAMTLSCMC(
        embed=args.embed,
        d_model=args.d_model,
        pretrained=args.pretrained,
        dropout=args.dropout,
        preview_frames=args.preview_frames,
        preview_size=args.preview_size,
        temporal_steps=args.t_target,
        attention_heads=args.attention_heads,
        train_base=args.train_base,
        use_scmc=args.use_scmc,
        scmc_hidden=args.scmc_hidden,
        scmc_strength=args.scmc_strength,
    ).to(device)


def freeze_batchnorm_statistics(module: nn.Module) -> None:
    """Sparse groups vary by route; retain the dense teacher's running statistics."""
    for child in module.modules():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, PSMSBN2d)):
            child.eval()


# -----------------------------------------------------------------------------
# Stage 1: dense/subset-capable teacher
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_masks(
    model: COVAMTLSCMC,
    loader: DataLoader,
    device: torch.device,
    dynamic: bool = False,
    router: Optional[AvailabilityConditionedSufficientRouter] = None,
    args=None,
    forced_available: Optional[Sequence[bool]] = None,
) -> dict:
    model.eval()
    if router is not None:
        router.eval()
    store = new_prediction_store()
    selected_sum = torch.zeros(5)
    subset_count = torch.zeros(32, dtype=torch.long)
    count = 0
    cost_sum = 0.0
    modality_count_sum = 0.0
    available_cost_sum = 0.0
    numerical_fallback_sum = 0.0
    selected_full_available_sum = 0.0
    confidence_supported_sum = 0.0
    relative_mean_regret_sum = 0.0
    relative_upper_regret_sum = 0.0
    elapsed = 0.0
    measured_batches = 0
    costs = torch.tensor(args.modality_costs, device=device)

    for batch_index, batch in enumerate(tqdm(loader, desc="evaluation", leave=False)):
        data = move_batch(batch, device)
        kp, context, body, face = data[:4]
        labels = labels_tuple(data)
        available = None
        if forced_available is not None:
            available = torch.tensor(
                forced_available, dtype=torch.bool, device=device
            )[None].expand(kp.size(0), -1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast("cuda", enabled=device.type == "cuda"):
            if dynamic:
                result = route_from_raw(
                    model, router, kp, context, body, face,
                    available=available,
                )
                logits = result["logits"]
            else:
                if available is None:
                    logits = model.forward_dense(kp, context, body, face)
                else:
                    scout = model.scout(kp, context, body, face)
                    tokens = model.encode_selected(available, kp, context, body, face)
                    logits = model.predict(scout, tokens, available)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        if batch_index >= args.latency_warmup_batches:
            elapsed += time.perf_counter() - start
            measured_batches += kp.size(0)
        update_predictions(store, logits, labels)
        if dynamic:
            mask = result["mask"].detach().cpu()
            selected_sum += mask.float().sum(0)
            subset_ids = (
                mask.long() * torch.tensor([1, 2, 4, 8, 16], dtype=torch.long)[None]
            ).sum(1)
            subset_count += torch.bincount(subset_ids, minlength=32)
            count += mask.size(0)
            cost_sum += float((mask.float() * costs.cpu()[None]).sum())
            modality_count_sum += float(mask.sum().cpu())
            current_available = result["available"].detach().cpu()
            available_cost_sum += float(
                (current_available.float() * costs.cpu()[None]).sum()
            )
            numerical_fallback_sum += float(result["fallback"].float().sum().cpu())
            selected_full_available_sum += float(
                result["selected_full_available"].float().sum().cpu()
            )
            confidence_supported_sum += float(
                result["confidence_supported"].float().sum().cpu()
            )
            relative_mean_regret_sum += float(
                result["predicted_relative_mean_regret"].float().sum().cpu()
            )
            relative_upper_regret_sum += float(
                result["predicted_relative_upper_regret"].float().sum().cpu()
            )

    metrics = finish_metrics(store)
    metrics["model_latency_ms_per_clip"] = 1000.0 * elapsed / max(measured_batches, 1)
    if dynamic:
        metrics.update(
            {
                "avg_modalities": modality_count_sum / max(count, 1),
                "avg_cost": cost_sum / max(count, 1),
                "full_cost": float(costs.sum().cpu()),
                "relative_branch_cost": cost_sum / max(count * float(costs.sum().cpu()), 1e-8),
                "available_baseline_cost": available_cost_sum / max(count, 1),
                "relative_to_available_cost": cost_sum / max(available_cost_sum, 1e-8),
                "numerical_fallback_rate": numerical_fallback_sum / max(count, 1),
                "selected_full_available_rate": selected_full_available_sum / max(count, 1),
                "confidence_supported_selection_rate": confidence_supported_sum / max(count, 1),
                "mean_selected_relative_regret": relative_mean_regret_sum / max(count, 1),
                "mean_selected_relative_upper_regret": relative_upper_regret_sum / max(count, 1),
                "selection_rate": {
                    name: float(selected_sum[i] / max(count, 1)) for i, name in enumerate(MODALITY_NAMES)
                },
                "selection_pattern_rate": {
                    "+".join(
                        MODALITY_NAMES[index]
                        for index in range(5)
                        if subset_id & (1 << index)
                    ): float(subset_count[subset_id] / max(count, 1))
                    for subset_id in range(1, 32)
                    if subset_count[subset_id] > 0
                },
            }
        )
    return metrics


@torch.no_grad()
def evaluate_all_subsets(
    model: COVAMTLSCMC,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate every non-empty fixed subset with one dense encoding per batch."""
    model.eval()
    masks = mask_table(device)
    stores = {subset_id: new_prediction_store() for subset_id in range(1, 32)}
    for batch in tqdm(loader, desc="31-subset evaluation", leave=False):
        data = move_batch(batch, device)
        kp, context, body, face = data[:4]
        labels = labels_tuple(data)
        with autocast("cuda", enabled=device.type == "cuda"):
            scout_tokens, tokens = model.encode_all(kp, context, body, face)
            for subset_id in range(1, 32):
                subset_mask = masks[subset_id][None].expand(kp.size(0), -1)
                logits, _, _ = model.predict(scout_tokens, tokens, subset_mask)
                update_predictions(stores[subset_id], logits, labels)

    subset_results = {}
    cardinality_results = {}
    for subset_id, store in stores.items():
        metrics = finish_metrics(store)
        names = [
            MODALITY_NAMES[index]
            for index in range(5)
            if subset_id & (1 << index)
        ]
        subset_results[str(subset_id)] = {
            "modalities": names,
            "cardinality": len(names),
            **metrics,
        }
    for cardinality in range(1, 6):
        group = [
            result for result in subset_results.values()
            if result["cardinality"] == cardinality
        ]
        macc = np.array([result["macc"] for result in group], dtype=np.float64)
        mf1 = np.array([result["mf1"] for result in group], dtype=np.float64)
        worst_index = int(macc.argmin())
        cardinality_results[str(cardinality)] = {
            "subset_count": len(group),
            "mean_macc": float(macc.mean()),
            "worst_macc": float(macc.min()),
            "mean_mf1": float(mf1.mean()),
            "worst_mf1": float(mf1.min()),
            "worst_subset": group[worst_index]["modalities"],
        }
    return {
        "by_cardinality": cardinality_results,
        "by_subset": subset_results,
    }


def train_teacher(args, loaders: Dict[str, DataLoader], device: torch.device) -> Path:
    stage_dir = args.run_dir / "teacher"
    last_path, best_path = stage_dir / "last.pt", stage_dir / "bestmacc.pt"
    model = model_from_args(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_macc = 1, -1.0
    if args.resume and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer, scaler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macc = float(checkpoint.get("best", {}).get("macc", -1.0))
    elif args.init_ckpt is not None:
        initialize_from_checkpoint(model, args.init_ckpt)

    for epoch in range(start_epoch, args.teacher_epochs + 1):
        model.train()
        running, iteration = 0.0, 0
        for iteration, batch in enumerate(tqdm(loaders["train"], desc=f"teacher {epoch}", leave=False), 1):
            data = move_batch(batch, device)
            kp, context, body, face = data[:4]
            labels = labels_tuple(data)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                scout, tokens = model.encode_all(kp, context, body, face)
                full_mask = torch.ones(kp.size(0), 5, dtype=torch.bool, device=device)
                full_logits, _, _ = model.predict(scout, tokens, full_mask)
                loss_full = multitask_ce(full_logits, labels)
                loss_subset = full_logits[0].new_zeros(())
                loss_kd = full_logits[0].new_zeros(())
                for _ in range(args.subset_samples_per_batch):
                    subset_mask = sample_balanced_masks(
                        kp.size(0), device, include_full=True
                    )
                    subset_logits, _, _ = model.predict(scout, tokens, subset_mask)
                    loss_subset = loss_subset + multitask_ce(subset_logits, labels)
                    loss_kd = loss_kd + confidence_gated_kd(
                        subset_logits, full_logits,
                        args.kd_temperature, args.kd_confidence,
                    )
                loss_subset = loss_subset / args.subset_samples_per_batch
                loss_kd = loss_kd / args.subset_samples_per_batch
                # Give the cheap scout semantic content without letting its
                # alignment objective alter the heavy representation.
                heavy_summary = tokens.mean(1).detach()
                loss_scout = (
                    1.0 - F.cosine_similarity(scout, heavy_summary, dim=-1)
                ).mean() + 0.1 * F.smooth_l1_loss(scout, heavy_summary)
                loss = (
                    loss_full
                    + args.subset_loss_weight * loss_subset
                    + args.kd_weight * loss_kd
                    + args.scout_alignment_weight * loss_scout
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
        val = evaluate_masks(model, loaders["val"], device, dynamic=False, args=args)
        print(f"[teacher {epoch}] loss={running/max(iteration,1):.4f} val_mAcc={val['macc']:.4f}")
        if val["macc"] > best_macc:
            best_macc = val["macc"]
            save_checkpoint(best_path, epoch, model, optimizer, scaler, {"macc": best_macc})
        save_checkpoint(last_path, epoch, model, optimizer, scaler, {"macc": best_macc})
    return best_path


# -----------------------------------------------------------------------------
# Stage 2a: enumerate all semantic subsets and cache counterfactual targets
# -----------------------------------------------------------------------------
# =============================================================================
# ACSR oracle/cache/router definitions (availability-relative, no budget)
# =============================================================================
@torch.no_grad()
def build_oracle_cache_for_split(
    args, split: str, loader: DataLoader, model: COVAMTLSCMC,
    device: torch.device, cache_dir: Optional[Path] = None,
    source_checkpoint: Optional[Path] = None,
) -> Path:
    model.eval()
    masks = mask_table(device)
    collected = {key: [] for key in (
        "scout_tokens", "subset_loss", "predictions", "labels"
    )}
    for batch in tqdm(loader, desc=f"removal oracle {split}", leave=False):
        data = move_batch(batch, device)
        kp, context, body, face = data[:4]
        labels = labels_tuple(data)
        with autocast("cuda", enabled=device.type == "cuda"):
            scout_tokens, tokens = model.encode_all(kp, context, body, face)
            losses = torch.full((kp.size(0), 32, 4), torch.nan, device=device)
            predictions = torch.full(
                (kp.size(0), 32, 4), -1, dtype=torch.long, device=device
            )
            # Empty subset is deliberately excluded: TSMF always needs one stream.
            for subset_id in range(1, 32):
                current = masks[subset_id][None].expand(kp.size(0), -1)
                logits, _, _ = model.predict(scout_tokens, tokens, current)
                losses[:, subset_id] = normalized_per_task_ce(logits, labels)
                predictions[:, subset_id] = torch.stack(
                    [logit.argmax(1) for logit in logits], 1
                )
        collected["scout_tokens"].append(scout_tokens.half().cpu())
        collected["subset_loss"].append(losses.float().cpu())
        collected["predictions"].append(predictions.cpu())
        collected["labels"].append(torch.stack(labels, 1).cpu())

    output = {key: torch.cat(parts, 0) for key, parts in collected.items()}
    output.update({
        "mask_table": masks.cpu(), "modality_names": MODALITY_NAMES,
        "task_names": TASK_NAMES,
        "source_checkpoint": str(source_checkpoint or args.teacher_ckpt),
        "target_semantics": "normalized per-task loss for each of 31 non-empty subsets",
    })
    cache_dir = cache_dir or args.oracle_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}.pt"
    torch.save(output, path)
    print(f"removal oracle saved: {path} | samples={output['scout_tokens'].size(0)}")
    return path


def metrics_from_label_prediction(labels: torch.Tensor, predictions: torch.Tensor) -> dict:
    accuracy, macro_f1 = {}, {}
    labels_np, predictions_np = labels.numpy(), predictions.numpy()
    for task_id, name in enumerate(TASK_NAMES):
        accuracy[name] = accuracy_score(labels_np[:, task_id], predictions_np[:, task_id])
        macro_f1[name] = f1_score(
            labels_np[:, task_id], predictions_np[:, task_id], average="macro"
        )
    return {
        "acc": accuracy, "f1": macro_f1,
        "macc": float(np.mean(list(accuracy.values()))),
        "mf1": float(np.mean(list(macro_f1.values()))),
    }


def analyze_oracle_cache(cache_path: Path, *_unused) -> dict:
    """Checks whether clip-specific subsets beat one globally fixed subset."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    total_loss = cache["subset_loss"].mean(2)
    valid_ids = torch.arange(1, 32)
    valid_loss = total_loss[:, 1:]
    oracle_local = valid_loss.argmin(1)
    oracle_ids = valid_ids[oracle_local]
    fixed_id = int(valid_ids[valid_loss.mean(0).argmin()])
    rows = torch.arange(total_loss.size(0))
    oracle_predictions = cache["predictions"][rows, oracle_ids]
    fixed_predictions = cache["predictions"][:, fixed_id]
    unique, counts = torch.unique(oracle_ids, return_counts=True)
    order = counts.argsort(descending=True)
    dense_loss = total_loss[:, 31]
    non_inferior = valid_loss <= dense_loss[:, None]
    cardinality = cache["mask_table"][1:32].sum(1)
    minimum_cardinality = cardinality[None].expand_as(valid_loss).masked_fill(
        ~non_inferior, 6
    ).min(1).values
    report = {
        "selection_rule": "minimum mean normalized four-task loss; no cost constraint",
        "unique_oracle_subsets": int(unique.numel()),
        "dominant_oracle_subset_rate": float(counts.max() / len(oracle_ids)),
        "fixed_subset": [MODALITY_NAMES[m] for m in range(5) if fixed_id & (1 << m)],
        "oracle": metrics_from_label_prediction(cache["labels"], oracle_predictions),
        "fixed": metrics_from_label_prediction(cache["labels"], fixed_predictions),
        "oracle_minus_fixed_mean_loss": float(
            valid_loss[:, fixed_id - 1].mean() - total_loss[rows, oracle_ids].mean()
        ),
        "dense_relative_sufficiency": {
            "mean_minimum_cardinality": float(minimum_cardinality.float().mean()),
            "minimum_cardinality_rate": {
                str(k): float((minimum_cardinality == k).float().mean())
                for k in range(1, 6)
            },
        },
        "top_oracle_subsets": [
            {
                "subset_id": int(unique[i]),
                "modalities": [
                    MODALITY_NAMES[m] for m in range(5) if int(unique[i]) & (1 << m)
                ],
                "rate": float(counts[i] / len(oracle_ids)),
            }
            for i in order[:10]
        ],
    }
    report_path = cache_path.with_name(cache_path.stem + "_oracle_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def run_cache(args, loaders, device) -> None:
    model = model_from_args(args, device)
    load_checkpoint(args.teacher_ckpt, model)
    for split in ("train", "val", "test"):
        path = build_oracle_cache_for_split(
            args, split, loaders[split], model, device,
            args.oracle_cache_dir, args.teacher_ckpt,
        )
        if split == "val":
            analyze_oracle_cache(path)


class AvailabilityRegretDataset(Dataset):
    """Scout previews and all subset losses; availability is sampled online."""

    def __init__(self, cache_path: Path):
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "scout_tokens" not in self.cache or "subset_loss" not in self.cache:
            raise KeyError(f"Invalid subset cache: {cache_path}")

    def __len__(self) -> int:
        return self.cache["scout_tokens"].size(0)

    def __getitem__(self, index: int):
        return (
            self.cache["scout_tokens"][index].float(),
            self.cache["subset_loss"][index].float(),
        )


def random_available_mask(batch: int, device: torch.device, drop_probability: float) -> torch.Tensor:
    available = torch.ones(batch, 5, dtype=torch.bool, device=device)
    if drop_probability <= 0:
        return available
    available &= torch.rand(batch, 5, device=device) >= drop_probability
    empty = ~available.any(1)
    if empty.any():
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        keep = torch.randint(0, 5, (rows.numel(),), device=device)
        available[rows, keep] = True
    return available


def deterministic_availability_patterns(
    device: torch.device, all_patterns: bool = True
) -> torch.Tensor:
    """Return deterministic availability states with the dense state first."""
    table = mask_table(device)[1:32]
    ordered = torch.cat((table[-1:], table[:-1]), 0)
    if all_patterns:
        return ordered
    # Normal operation and the five single-stream failures are the primary
    # validation scenarios; severe multi-failure cases remain calibration data.
    return ordered[ordered.sum(1) >= 4]


def router_batch_loss(
    router, batch, args, device, train: bool = True,
    forced_available: Optional[torch.Tensor] = None,
):
    scout_tokens, subset_loss = [item.to(device) for item in batch]
    if forced_available is None:
        available = random_available_mask(
            scout_tokens.size(0), device,
            args.router_availability_dropout if train else 0.0,
        )
    else:
        available = forced_available.bool().to(device)
        if available.ndim == 1:
            available = available[None].expand(scout_tokens.size(0), -1)
    routed_scout = scout_tokens * available[:, :, None].to(scout_tokens.dtype)
    predicted_task_mean, predicted_task_scale = router(routed_scout, available)
    target_task_regret, valid = availability_relative_targets(
        subset_loss, available, router.candidate_masks
    )
    valid_tasks = valid[:, :, None].expand_as(predicted_task_mean)
    error = predicted_task_mean.float() - target_task_regret.float()
    scale = predicted_task_scale.float().clamp_min(1e-4)
    nll_values = torch.log(scale) + 0.5 * (error / scale).square()
    nll = nll_values[valid_tasks].mean()
    smooth = F.smooth_l1_loss(
        predicted_task_mean.float()[valid_tasks],
        target_task_regret.float()[valid_tasks],
    )

    predicted_mean, predicted_scale = predicted_regret_statistics(
        predicted_task_mean, predicted_task_scale
    )
    target_mean = target_task_regret.float().mean(-1)
    sufficient_target = (target_mean <= 0.0).float()
    sufficiency_logit = -predicted_mean / predicted_scale.detach().clamp_min(1e-4)
    sufficiency = F.binary_cross_entropy_with_logits(
        sufficiency_logit[valid], sufficient_target[valid]
    )

    # Within each availability context, retain the relative ordering of
    # counterfactual regrets without inserting a cardinality or cost penalty.
    temperature = max(float(args.router_temperature), 1e-4)
    predicted_logits = (-predicted_mean / temperature).masked_fill(~valid, -torch.inf)
    target_logits = (-target_mean / temperature).masked_fill(~valid, -torch.inf)
    target_probability = torch.softmax(target_logits, dim=1)
    predicted_log_probability = torch.log_softmax(
        predicted_logits, dim=1
    ).masked_fill(~valid, 0.0)
    listwise = -(target_probability * predicted_log_probability).sum(1).mean()
    loss = (
        nll
        + args.router_smooth_weight * smooth
        + args.router_sufficiency_weight * sufficiency
        + args.router_listwise_weight * listwise
    )

    with torch.no_grad():
        predicted_sufficient = valid & (predicted_mean <= 0.0)
        sufficient_accuracy = (
            predicted_sufficient[valid] == sufficient_target[valid].bool()
        ).float().mean()
        cardinality = router.candidate_cardinality[None].expand_as(valid)
        oracle_cardinality = cardinality.masked_fill(
            ~(valid & sufficient_target.bool()), 6
        ).min(1).values
        predicted_cardinality = cardinality.masked_fill(
            ~predicted_sufficient, 6
        ).min(1).values
        fallback = predicted_cardinality == 6
        predicted_cardinality = torch.where(
            fallback, available.sum(1), predicted_cardinality
        )
        cardinality_error = (
            predicted_cardinality.float() - oracle_cardinality.float()
        ).abs().mean()
    return (
        loss, nll.detach(), smooth.detach(), sufficiency.detach(),
        listwise.detach(), sufficient_accuracy.detach(), cardinality_error.detach(),
    )


@torch.no_grad()
def evaluate_router(router, loader, args, device) -> dict:
    router.eval()
    sums, count = np.zeros(7, dtype=np.float64), 0
    patterns = deterministic_availability_patterns(device, all_patterns=False)
    for batch in loader:
        for pattern in patterns:
            values = router_batch_loss(
                router, batch, args, device, False, pattern
            )
            sums += np.array([float(value) for value in values])
            count += 1
    return dict(zip(
        (
            "loss", "nll", "smooth", "sufficiency", "listwise",
            "sufficiency_accuracy", "cardinality_error",
        ),
        (sums / max(count, 1)).tolist(),
    ))


def train_router(args, device: torch.device) -> Path:
    train_ds = AvailabilityRegretDataset(args.oracle_cache_dir / "train.pt")
    val_ds = AvailabilityRegretDataset(args.oracle_cache_dir / "val.pt")
    train_loader = DataLoader(
        train_ds, batch_size=args.router_batch_size, shuffle=True,
        num_workers=min(args.num_workers, 4), pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.router_batch_size, shuffle=False,
        num_workers=min(args.num_workers, 4), pin_memory=True
    )
    router = AvailabilityConditionedSufficientRouter(
        args.embed, args.router_hidden
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.router_lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    stage_dir = args.run_dir / "router"
    last_path, best_path = stage_dir / "last.pt", stage_dir / "best.pt"
    start_epoch, best_loss = 1, float("inf")
    if args.resume and last_path.exists():
        checkpoint = load_checkpoint(last_path, router, optimizer, scaler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best", {}).get("loss", float("inf")))
    for epoch in range(start_epoch, args.router_epochs + 1):
        router.train()
        running, iteration = 0.0, 0
        for iteration, batch in enumerate(tqdm(train_loader, desc=f"router {epoch}", leave=False), 1):
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                values = router_batch_loss(router, batch, args, device, True)
                loss = values[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
        val = evaluate_router(router, val_loader, args, device)
        print(
            f"[router {epoch}] train={running/max(iteration,1):.4f} "
            f"val={val['loss']:.4f} suff_acc={val['sufficiency_accuracy']:.4f} "
            f"card_err={val['cardinality_error']:.4f}"
        )
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            save_checkpoint(best_path, epoch, router, optimizer, scaler, {"loss": best_loss})
        save_checkpoint(last_path, epoch, router, optimizer, scaler, {"loss": best_loss})
    return best_path


def one_sided_calibration_quantile(
    values: torch.Tensor, coverage: float
) -> float:
    values = values.float().flatten()
    if values.numel() == 0:
        raise ValueError("Cannot calibrate an empty residual set")
    rank = min(values.numel(), math.ceil((values.numel() + 1) * coverage))
    quantile = float(values.kthvalue(rank).values)
    # A negative multiplier would make the upper bound narrower than the
    # predicted mean. Keeping zero as the lower limit is deliberately safe.
    return max(0.0, quantile)


@torch.no_grad()
def calibrate_router_from_cache(
    args, device: torch.device, cache_path: Path, output_path: Path,
    source_model_checkpoint: Path,
) -> dict:
    dataset = AvailabilityRegretDataset(cache_path)
    loader = DataLoader(
        dataset, batch_size=args.router_batch_size, shuffle=False,
        num_workers=min(args.num_workers, 4), pin_memory=True,
    )
    router = AvailabilityConditionedSufficientRouter(
        args.embed, args.router_hidden
    ).to(device)
    load_checkpoint(args.router_ckpt, router)
    router.eval()
    patterns = deterministic_availability_patterns(device, all_patterns=True)
    residuals = {
        (available_cardinality, candidate_cardinality): []
        for available_cardinality in range(1, 6)
        for candidate_cardinality in range(1, available_cardinality + 1)
    }
    for batch in tqdm(loader, desc="router calibration", leave=False):
        scout_tokens, subset_loss = [item.to(device) for item in batch]
        for pattern in patterns:
            available = pattern[None].expand(scout_tokens.size(0), -1)
            routed_scout = scout_tokens * available[:, :, None].to(scout_tokens.dtype)
            with autocast("cuda", enabled=device.type == "cuda"):
                task_mean, task_scale = router(routed_scout, available)
            target_regret, valid = availability_relative_targets(
                subset_loss, available, router.candidate_masks
            )
            predicted_mean, predicted_scale = predicted_regret_statistics(
                task_mean, task_scale
            )
            target_mean = target_regret.float().mean(-1)
            rows = torch.arange(scout_tokens.size(0), device=device)
            reference_candidate = masks_to_subset_ids(available) - 1
            predicted_reference = predicted_mean[rows, reference_candidate]
            target_reference = target_mean[rows, reference_candidate]
            reference_scale = predicted_scale[rows, reference_candidate]
            predicted_relative = predicted_mean - predicted_reference[:, None]
            target_relative = target_mean - target_reference[:, None]
            relative_scale = (
                predicted_scale.square() + reference_scale[:, None].square()
            ).sqrt().clamp_min(1e-4)
            standardized = (
                target_relative - predicted_relative
            ) / relative_scale
            available_cardinality = int(pattern.sum().item())
            for candidate_cardinality in range(1, available_cardinality + 1):
                use = valid & (
                    router.candidate_cardinality[None] == candidate_cardinality
                )
                if use.any():
                    residuals[(available_cardinality, candidate_cardinality)].append(
                        standardized[use].cpu()
                    )

    q_matrix = torch.ones(5, 5, dtype=torch.float32)
    realized_coverage, counts = {}, {}
    for available_cardinality in range(1, 6):
        for candidate_cardinality in range(1, available_cardinality + 1):
            key = (available_cardinality, candidate_cardinality)
            values = torch.cat(residuals[key])
            q = one_sided_calibration_quantile(values, args.calibration_coverage)
            q_matrix[available_cardinality - 1, candidate_cardinality - 1] = q
            report_key = f"A{available_cardinality}_S{candidate_cardinality}"
            realized_coverage[report_key] = float((values <= q).float().mean())
            counts[report_key] = int(values.numel())
    report = {
        "method": (
            "one-sided availability-and-cardinality conditional calibration "
            "of paired candidate-minus-all-available regret"
        ),
        "coverage_target": float(args.calibration_coverage),
        "q_by_available_and_selected_cardinality": q_matrix.tolist(),
        "realized_coverage": realized_coverage,
        "counts": counts,
        "availability_contexts": [
            pattern.int().tolist() for pattern in patterns.cpu()
        ],
        "router_checkpoint": str(args.router_ckpt),
        "source_model_checkpoint": str(source_model_checkpoint),
        "source_cache": str(cache_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def run_teacher_calibration(args, device: torch.device) -> dict:
    return calibrate_router_from_cache(
        args, device, args.oracle_cache_dir / "val.pt",
        args.teacher_calibration_path, args.teacher_ckpt,
    )


def run_student_recalibration(args, loaders, device: torch.device) -> dict:
    if not args.sparse_ckpt.exists():
        raise FileNotFoundError(f"Train the sparse student first: {args.sparse_ckpt}")
    student = model_from_args(args, device)
    load_checkpoint(args.sparse_ckpt, student)
    cache_path = build_oracle_cache_for_split(
        args, "val", loaders["val"], student, device,
        args.student_cache_dir, args.sparse_ckpt,
    )
    return calibrate_router_from_cache(
        args, device, cache_path,
        args.student_calibration_path, args.sparse_ckpt,
    )


# -----------------------------------------------------------------------------
# Stage 3: sparse fine-tuning and robust unexpected-failure augmentation
# -----------------------------------------------------------------------------
def randomly_drop_one_selected(mask: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return mask
    output = mask.clone()
    for i in range(mask.size(0)):
        candidates = torch.nonzero(output[i], as_tuple=False).flatten()
        if candidates.numel() > 1 and random.random() < probability:
            selected = candidates[torch.randint(candidates.numel(), (1,), device=mask.device)]
            output[i, selected] = False
    return output


def mix_route_with_admissible_alternative(
    route_mask: torch.Tensor,
    alternative_mask: torch.Tensor,
    has_alternative: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Explore only another self-referenced mean-admissible subset."""
    if probability <= 0:
        return route_mask
    replace = (
        torch.rand(route_mask.size(0), device=route_mask.device) < probability
    ) & has_alternative.bool()
    return torch.where(replace[:, None], alternative_mask, route_mask)


def load_router_from_args(
    args, device: torch.device, prefer_student_calibration: bool = True
) -> AvailabilityConditionedSufficientRouter:
    router = AvailabilityConditionedSufficientRouter(
        args.embed, args.router_hidden
    ).to(device)
    load_checkpoint(args.router_ckpt, router)
    calibration_path = (
        args.student_calibration_path
        if prefer_student_calibration and args.student_calibration_path.exists()
        else args.teacher_calibration_path
    )
    if calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        q = torch.tensor(
            calibration["q_by_available_and_selected_cardinality"],
            dtype=torch.float32,
            device=device,
        )
        if q.shape != (5, 5):
            raise ValueError(f"Invalid calibration file: {calibration_path}")
        router.calibration_q.copy_(q)
    elif args.require_calibration:
        raise FileNotFoundError(
            f"Run --stage calibrate first; calibration not found: {calibration_path}"
        )
    router.loaded_calibration = calibration_path if calibration_path.exists() else None
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad = False
    return router


def train_sparse(args, loaders, device) -> Path:
    teacher = model_from_args(args, device)
    load_checkpoint(args.teacher_ckpt, teacher)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student = copy.deepcopy(teacher).to(device)
    # deepcopy preserves the teacher's requires_grad=False flags. The sparse
    # student must be explicitly unfrozen before its optimizer is constructed.
    student.requires_grad_(True)
    # The router was fitted to the teacher's scout representation. Keep this
    # cheap preview fixed while adapting the selected heavy branches.
    student.scout.requires_grad_(False)
    trainable_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if trainable_parameters == 0:
        raise RuntimeError("Sparse student has no trainable parameters.")
    print(f"sparse student trainable parameters: {trainable_parameters:,}")
    router = load_router_from_args(
        args, device, prefer_student_calibration=False
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.sparse_lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    stage_dir = args.run_dir / "sparse"
    last_path, best_path = stage_dir / "last.pt", stage_dir / "bestmacc.pt"
    start_epoch, best_macc = 1, -1.0
    if args.resume and last_path.exists():
        checkpoint = load_checkpoint(last_path, student, optimizer, scaler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macc = float(checkpoint.get("best", {}).get("macc", -1.0))

    for epoch in range(start_epoch, args.sparse_epochs + 1):
        student.train()
        freeze_batchnorm_statistics(student)
        router.eval()
        running, iteration = 0.0, 0
        for iteration, batch in enumerate(tqdm(loaders["train"], desc=f"sparse {epoch}", leave=False), 1):
            data = move_batch(batch, device)
            kp, context, body, face = data[:4]
            labels = labels_tuple(data)
            with torch.no_grad(), autocast("cuda", enabled=device.type == "cuda"):
                teacher_scout, teacher_tokens = teacher.encode_all(kp, context, body, face)
                available = random_available_mask(
                    kp.size(0), device, args.sparse_availability_dropout
                )
                teacher_logits, _, _ = teacher.predict(
                    teacher_scout, teacher_tokens, available
                )
                route = route_from_precomputed(
                    teacher, router, teacher_scout, teacher_tokens,
                    available=available,
                )
                route_mask = route["mask"]

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                student_scout = student.scout(kp, context, body, face)
                train_mask = mix_route_with_admissible_alternative(
                    route_mask,
                    route["alternative_mask"],
                    route["has_admissible_alternative"],
                    args.route_exploration_probability,
                )
                student_tokens = student.encode_selected(train_mask, kp, context, body, face)
                predictor_mask = randomly_drop_one_selected(train_mask, args.failure_dropout)
                student_logits, _, _ = student.predict(student_scout, student_tokens, predictor_mask)
                ce = multitask_ce(student_logits, labels)
                kd = confidence_gated_kd(
                    student_logits, teacher_logits,
                    args.kd_temperature, args.kd_confidence,
                )
                loss = ce + args.kd_weight * kd
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())

        val = evaluate_masks(student, loaders["val"], device, True, router, args)
        print(f"[sparse {epoch}] loss={running/max(iteration,1):.4f} val_mAcc={val['macc']:.4f}")
        if val["macc"] > best_macc:
            best_macc = val["macc"]
            save_checkpoint(best_path, epoch, student, optimizer, scaler, {"macc": best_macc})
        save_checkpoint(last_path, epoch, student, optimizer, scaler, {"macc": best_macc})
    return best_path


# -----------------------------------------------------------------------------
# Final evaluation and go/no-go evidence
# -----------------------------------------------------------------------------
def run_final_evaluation(args, loaders, device) -> dict:
    model = model_from_args(args, device)
    checkpoint = args.sparse_ckpt if args.sparse_ckpt.exists() else args.teacher_ckpt
    load_checkpoint(checkpoint, model)
    router = load_router_from_args(args, device)
    dense = evaluate_masks(model, loaders["test"], device, False, None, args)
    dynamic = evaluate_masks(model, loaders["test"], device, True, router, args)
    validation_dynamic = {}
    validation_subsets = {}
    exhaustive_subsets = {}
    if args.eval_all_subsets:
        # Select every deployable fixed baseline using validation labels only.
        # Test labels are used once, after the subset identity is frozen.
        validation_dynamic = evaluate_masks(
            model, loaders["val"], device, True, router, args
        )
        validation_subsets = evaluate_all_subsets(
            model, loaders["val"], device
        )
        exhaustive_subsets = evaluate_all_subsets(
            model, loaders["test"], device
        )
    matched_fixed = {}
    if validation_subsets and exhaustive_subsets:
        rounded_cardinality = int(np.clip(
            round(validation_dynamic["avg_modalities"]), 1, 5
        ))
        group = [
            (subset_id, value)
            for subset_id, value in validation_subsets["by_subset"].items()
            if value["cardinality"] == rounded_cardinality
        ]
        best_subset_id, best_validation = max(
            group, key=lambda item: item[1]["macc"]
        )
        best_test = exhaustive_subsets["by_subset"][best_subset_id]
        matched_fixed = {
            "selection_split": "validation",
            "selection_metric": "macc",
            "validation_dynamic_avg_modalities": validation_dynamic["avg_modalities"],
            "rounded_validation_dynamic_cardinality": rounded_cardinality,
            "subset_id": int(best_subset_id),
            "modalities": best_test["modalities"],
            "validation_macc": best_validation["macc"],
            "validation_mf1": best_validation["mf1"],
            "test_macc": best_test["macc"],
            "test_mf1": best_test["mf1"],
            "dynamic_minus_fixed_macc_pp": 100.0 * (
                dynamic["macc"] - best_test["macc"]
            ),
        }
    parameter_count = {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )),
        "scmc": int(sum(parameter.numel() for parameter in model.scmc.parameters()))
        if model.scmc is not None else 0,
        "router": int(sum(parameter.numel() for parameter in router.parameters())),
    }
    failure_robustness = {}
    if args.eval_failures:
        for missing_id, missing_name in enumerate(MODALITY_NAMES):
            available = [True] * 5
            available[missing_id] = False
            static_missing = evaluate_masks(
                model, loaders["test"], device, False, None, args,
                forced_available=available,
            )
            dynamic_missing = evaluate_masks(
                model, loaders["test"], device, True, router, args,
                forced_available=available,
            )
            failure_robustness[f"missing_{missing_name}"] = {
                "static_available_streams": static_missing,
                "dynamic": dynamic_missing,
            }
    result = {
        "method": "COVA-MTL v6.3-ACSR-SCMC-SRCR-ValFixed",
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "run_dir": str(args.run_dir),
        "backbone_components": ["shared ResNet-18", "PS-MSBN", "ETB", "TSMF"],
        "subset_conditioned_modality_calibration": bool(args.use_scmc),
        "checkpoint": str(checkpoint),
        "router_checkpoint": str(args.router_ckpt),
        "calibration": str(router.loaded_calibration) if router.loaded_calibration else None,
        "selection": (
            "smallest availability-valid subset whose paired predicted mean four-"
            "task regret is no greater than the all-currently-available reference; "
            "paired calibrated upper risk breaks same-cardinality ties"
        ),
        "selection_has_no_cost_or_budget_term": True,
        "selection_reference": "all currently available modalities for the same sample",
        "sparse_route_exploration": (
            "next smallest self-referenced mean-admissible subset; primary-route fallback "
            "when no admissible alternative exists"
        ),
        "failure_protocol": (
            "known unavailability is given to ACSR before routing; sparse training "
            "also drops one routed stream stochastically for unexpected-failure robustness"
        ),
        "sparse_availability_dropout": float(args.sparse_availability_dropout),
        "unexpected_failure_dropout": float(args.failure_dropout),
        "modality_costs": dict(zip(MODALITY_NAMES, args.modality_costs)),
        "parameter_count": parameter_count,
        "dense": dense,
        "dynamic": dynamic,
        "exhaustive_nonempty_subsets": exhaustive_subsets,
        "exhaustive_subset_scope": (
            "test diagnostics only; no test result is used to choose the fixed subset"
            if exhaustive_subsets else None
        ),
        "fixed_subset_selection_protocol": (
            "round validation dynamic average cardinality, select the highest-"
            "mAcc fixed subset of that cardinality on validation, then evaluate "
            "the frozen subset once on test"
        ),
        "matched_cardinality_best_fixed_subset": matched_fixed,
        "single_stream_failure": failure_robustness,
        "macc_drop_pp": 100.0 * (dense["macc"] - dynamic["macc"]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with open(args.run_dir / "final_test.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def smoke_test(args, device: torch.device) -> None:
    args.pretrained = False
    # Keep the structural test light; production training still uses 16 frames.
    args.t_target = 4
    model = model_from_args(args, device).eval()
    router = AvailabilityConditionedSufficientRouter(
        args.embed, args.router_hidden
    ).to(device).eval()
    batch, steps, size = 2, 4, 64
    kp = torch.randn(batch, 3, steps, 136, 1, device=device)
    context = torch.rand(batch, 3, steps, size, size, device=device)
    body = torch.rand_like(context)
    face = torch.rand_like(context)
    with torch.no_grad():
        scout, tokens = model.encode_all(kp, context, body, face)
        full_mask = torch.ones(batch, 5, dtype=torch.bool, device=device)
        subset_mean, subset_scale = router(scout, full_mask)
        logits, state, _ = model.predict(scout, tokens, full_mask)
        if model.use_scmc:
            calibrated = model.scmc(tokens, scout, full_mask)
            assert torch.allclose(calibrated, tokens, atol=1e-6), (
                "SCMC must be identity-initialized before training"
            )
        route = route_from_raw(
            model, router, kp, context, body, face,
        )
        missing_gesture = full_mask.clone()
        missing_gesture[:, 4] = False
        failure_route = select_self_referenced_subset(
            router, scout, available=missing_gesture
        )
    assert scout.shape == (batch, 5, args.embed)
    assert tokens.shape == (batch, steps // 2, 5, args.embed)
    assert state.shape == (batch, args.d_model)
    assert all(logit.shape == (batch, classes) for logit, classes in zip(logits, TASK_CLASSES))
    assert route["mask"].shape == (batch, 5)
    assert subset_mean.shape == (batch, 31, 4)
    assert subset_scale.shape == (batch, 31, 4)
    assert (subset_scale > 0).all()
    assert route["reference_admissible"].all()
    assert (route["predicted_relative_mean_regret"] <= 1e-6).all()
    assert route["confidence_supported"].shape == (batch,)
    assert not failure_route["mask"][:, 4].any()
    print("smoke test passed")
    print("dense logits:", [tuple(x.shape) for x in logits])
    print("dynamic masks:", route["mask"].cpu().tolist())


# -----------------------------------------------------------------------------
# Seed-42 configuration-induced representation analysis (inference only)
# -----------------------------------------------------------------------------
REPRESENTATION_TASK_ABBREVIATIONS = ("DER", "DBR", "TCR", "VBR")
REPRESENTATION_MODEL_LABELS = {
    "full_scmc": "Full SCML",
    "no_subset_calibration": "w/o subset calibration",
}


def _is_complete_representation_run(path: Path) -> bool:
    return (
        (path / "sparse" / "bestmacc.pt").exists()
        and (path / "router" / "best.pt").exists()
        and (
            (path / "router" / "calibration_student.json").exists()
            or (path / "router" / "calibration_teacher.json").exists()
        )
    )


def _normalize_saved_run_dir(path: Path, seed: int = 42) -> Optional[Path]:
    path = path.expanduser().resolve()
    candidates = (path, path / f"seed_{seed}")
    for candidate in candidates:
        if _is_complete_representation_run(candidate):
            return candidate
        result_path = candidate / "final_test.json"
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as handle:
                    checkpoint = json.load(handle).get("checkpoint")
                if checkpoint:
                    inferred = Path(checkpoint).expanduser().resolve().parents[1]
                    if _is_complete_representation_run(inferred):
                        return inferred
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                pass
    return None


def resolve_internal_ablation_run(
    explicit: Optional[Path], variant: str, seed: int = 42
) -> Path:
    checked = []
    if explicit is not None:
        checked.append(explicit)
    roots = (
        Path(
            ""
            "RUNSAIDE_COVA_MTL_v6_3_SCML_InternalAblation"
        ),
        Path(
            ""
            "RUNS/AIDE_COVA_MTL_v6_3_SCML_InternalAblation"
        ),
        Path(
            ""
            "RUNSAIDE_COVA_MTL_v6_3_SCML_InternalAblation"
        ),
    )
    checked.extend(root / variant for root in roots)
    for candidate in checked:
        resolved = _normalize_saved_run_dir(Path(candidate), seed)
        if resolved is not None:
            return resolved
    message = "\n".join(f"  - {Path(path)}" for path in checked)
    raise FileNotFoundError(
        f"Cannot find the completed seed-{seed} '{variant}' run. Checked:\n"
        f"{message}\n"
        "The analysis never retrains a model. Supply the correct existing path "
        f"with --{'full-run-dir' if variant == 'full' else 'no-calibration-run-dir'}."
    )


def representation_args_from_saved_run(
    base_args, run_dir: Path, use_scmc: bool
):
    args = copy.deepcopy(base_args)
    config_path = run_dir / "config.json"
    restorable = {
        "data_root", "split_dir", "split_seed", "batch_size", "num_workers",
        "t_target", "stride", "face_pad_top", "face_pad_bottom",
        "face_pad_left", "face_pad_right", "embed", "d_model", "dropout",
        "preview_frames", "preview_size", "attention_heads", "train_base",
        "scmc_hidden", "scmc_strength", "router_hidden", "modality_costs",
        "require_calibration",
    }
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        for key in restorable:
            if key not in saved or saved[key] is None or not hasattr(args, key):
                continue
            value = saved[key]
            if key in {"data_root", "split_dir"}:
                value = Path(value).expanduser().resolve()
            setattr(args, key, value)
    args.seed = 42
    args.split_seed = 42
    args.multi_seed = False
    args.run_dir = run_dir.expanduser().resolve()
    args.use_scmc = bool(use_scmc)
    # Loading a complete checkpoint does not require ImageNet weight download.
    args.pretrained = False
    if isinstance(args.modality_costs, str):
        args.modality_costs = parse_float_list(args.modality_costs, 5, "modality-costs")
    return configure_run_paths(args)


def load_representation_variant(args, device: torch.device):
    missing = [
        path for path in (args.sparse_ckpt, args.router_ckpt)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing saved checkpoints: {missing}")
    model = model_from_args(args, device)
    load_checkpoint(args.sparse_ckpt, model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    router = load_router_from_args(args, device, prefer_student_calibration=True)
    return model, router


def _append_representation_condition(
    store: dict,
    name: str,
    features: torch.Tensor,
    logits: Sequence[torch.Tensor],
    mask: torch.Tensor,
) -> None:
    if name not in store:
        store[name] = {
            "features": [],
            "logits": [[] for _ in TASK_NAMES],
            "mask": [],
        }
    store[name]["features"].append(features.detach().float().cpu().numpy())
    for task_index, task_logits in enumerate(logits):
        store[name]["logits"][task_index].append(
            task_logits.detach().float().cpu().numpy()
        )
    store[name]["mask"].append(mask.detach().bool().cpu().numpy())


@torch.no_grad()
def collect_configuration_representations(
    args,
    model: COVAMTLSCMC,
    router: AvailabilityConditionedSufficientRouter,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 0,
) -> dict:
    """Collect paired full, selected-subset, and known-missing representations."""
    model.eval()
    router.eval()
    conditions = {}
    labels_store = []
    sample_count = 0
    progress = tqdm(loader, desc=f"representation {args.run_dir.name}", leave=False)
    for batch in progress:
        data = move_batch(batch, device)
        if max_samples > 0:
            remaining = max_samples - sample_count
            if remaining <= 0:
                break
            if data[0].size(0) > remaining:
                data = tuple(tensor[:remaining] for tensor in data)
        kp, context, body, face = data[:4]
        labels = torch.stack(labels_tuple(data), dim=1)
        batch_size = kp.size(0)
        full_available = torch.ones(
            batch_size, len(MODALITY_NAMES), dtype=torch.bool, device=device
        )

        with autocast("cuda", enabled=device.type == "cuda"):
            scout_tokens, all_tokens = model.encode_all(kp, context, body, face)
            full_logits, _, full_features = model.predict(
                scout_tokens, all_tokens, full_available
            )
            _append_representation_condition(
                conditions, "full", full_features, full_logits, full_available
            )

            dynamic_route = select_self_referenced_subset(
                router, scout_tokens, full_available
            )
            dynamic_logits, _, dynamic_features = model.predict(
                scout_tokens, all_tokens, dynamic_route["mask"]
            )
            _append_representation_condition(
                conditions,
                "dynamic",
                dynamic_features,
                dynamic_logits,
                dynamic_route["mask"],
            )

            for modality_index, modality_name in enumerate(MODALITY_NAMES):
                available = full_available.clone()
                available[:, modality_index] = False

                # Available-all under the single-missing configuration isolates
                # the representation shift caused by removing that modality.
                static_logits, _, static_features = model.predict(
                    scout_tokens, all_tokens, available
                )
                _append_representation_condition(
                    conditions,
                    f"missing_{modality_name}",
                    static_features,
                    static_logits,
                    available,
                )

                # The rerouted condition additionally measures SSR adaptation
                # after the same modality is declared unavailable.
                rerouted = select_self_referenced_subset(
                    router, scout_tokens, available
                )
                rerouted_logits, _, rerouted_features = model.predict(
                    scout_tokens, all_tokens, rerouted["mask"]
                )
                _append_representation_condition(
                    conditions,
                    f"rerouted_missing_{modality_name}",
                    rerouted_features,
                    rerouted_logits,
                    rerouted["mask"],
                )

        labels_store.append(labels.detach().cpu().numpy())
        sample_count += batch_size
        if max_samples > 0 and sample_count >= max_samples:
            break

    if sample_count == 0:
        raise RuntimeError("No test samples were collected.")
    packed = {"labels": np.concatenate(labels_store, axis=0)}
    packed_conditions = {}
    for name, value in conditions.items():
        packed_conditions[name] = {
            "features": np.concatenate(value["features"], axis=0),
            "logits": [np.concatenate(parts, axis=0) for parts in value["logits"]],
            "mask": np.concatenate(value["mask"], axis=0),
        }
    packed["conditions"] = packed_conditions
    packed["sample_count"] = int(sample_count)
    return packed


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _jensen_shannon_per_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.clip(_stable_softmax(a), 1e-12, 1.0)
    q = np.clip(_stable_softmax(b), 1e-12, 1.0)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        np.sum(p * (np.log(p) - np.log(midpoint)), axis=1)
        + np.sum(q * (np.log(q) - np.log(midpoint)), axis=1)
    )


def _cosine_distance_per_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - numerator / np.clip(denominator, 1e-12, None)


def representation_metric_rows(model_key: str, result: dict) -> Tuple[List[dict], dict]:
    labels = result["labels"]
    conditions = result["conditions"]
    full = conditions["full"]
    rows = []
    distributions = {}
    for condition_name, condition in conditions.items():
        if condition_name == "full":
            continue
        distributions[condition_name] = {}
        avg_modalities = float(condition["mask"].sum(axis=1).mean())
        for task_index, abbreviation in enumerate(REPRESENTATION_TASK_ABBREVIATIONS):
            full_feature = full["features"][:, task_index]
            condition_feature = condition["features"][:, task_index]
            cosine = _cosine_distance_per_sample(full_feature, condition_feature)
            js = _jensen_shannon_per_sample(
                full["logits"][task_index], condition["logits"][task_index]
            )
            full_prediction = full["logits"][task_index].argmax(1)
            condition_prediction = condition["logits"][task_index].argmax(1)
            ground_truth = labels[:, task_index]
            full_accuracy = float(np.mean(full_prediction == ground_truth))
            condition_accuracy = float(np.mean(condition_prediction == ground_truth))
            agreement = float(np.mean(full_prediction == condition_prediction))
            distributions[condition_name][abbreviation] = cosine
            rows.append({
                "model": model_key,
                "model_label": REPRESENTATION_MODEL_LABELS[model_key],
                "condition": condition_name,
                "task": abbreviation,
                "samples": int(cosine.size),
                "cosine_distance_mean": float(cosine.mean()),
                "cosine_distance_std": float(cosine.std(ddof=0)),
                "cosine_distance_median": float(np.median(cosine)),
                "cosine_distance_q25": float(np.quantile(cosine, 0.25)),
                "cosine_distance_q75": float(np.quantile(cosine, 0.75)),
                "js_divergence_mean": float(js.mean()),
                "js_divergence_std": float(js.std(ddof=0)),
                "prediction_agreement": agreement,
                "full_accuracy": full_accuracy,
                "condition_accuracy": condition_accuracy,
                "accuracy_delta_pp": 100.0 * (condition_accuracy - full_accuracy),
                "avg_modalities": avg_modalities,
            })
    return rows, distributions


def save_representation_arrays(path: Path, result: dict) -> None:
    arrays = {"labels": result["labels"]}
    for condition_name, condition in result["conditions"].items():
        arrays[f"{condition_name}__features"] = condition["features"]
        arrays[f"{condition_name}__mask"] = condition["mask"].astype(np.uint8)
        for task_index, abbreviation in enumerate(REPRESENTATION_TASK_ABBREVIATIONS):
            arrays[f"{condition_name}__logits_{abbreviation}"] = condition["logits"][task_index]
    np.savez_compressed(path, **arrays)


def _save_figure(figure, output_path: Path, dpi: int) -> None:
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def render_feature_drift_violin(
    distributions_by_model: dict, output_path: Path, dpi: int
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required for this analysis") from error

    condition_names = ["dynamic"] + [f"missing_{name}" for name in MODALITY_NAMES]
    condition_labels = ["SSR\nsubset", "-Face", "-Body", "-Scene", "-Posture", "-Gesture"]
    model_keys = ("full_scmc", "no_subset_calibration")
    colors = {"full_scmc": "#1769AA", "no_subset_calibration": "#E76F51"}
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharex=True)
    for task_index, (axis, abbreviation) in enumerate(
        zip(axes.flat, REPRESENTATION_TASK_ABBREVIATIONS)
    ):
        centers = np.arange(len(condition_names), dtype=np.float64)
        for model_offset, model_key in enumerate(model_keys):
            offset = -0.17 if model_offset == 0 else 0.17
            values = [
                distributions_by_model[model_key][condition][abbreviation]
                for condition in condition_names
            ]
            violin = axis.violinplot(
                values,
                positions=centers + offset,
                widths=0.30,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors[model_key])
                body.set_edgecolor(colors[model_key])
                body.set_alpha(0.55)
            violin["cmedians"].set_color(colors[model_key])
            violin["cmedians"].set_linewidth(1.4)
        axis.set_title(abbreviation, fontweight="bold")
        axis.set_ylabel("Paired cosine distance")
        axis.set_xticks(centers, condition_labels)
        axis.grid(axis="y", alpha=0.25)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)
    figure.legend(
        handles=[
            Patch(facecolor=colors[key], alpha=0.65, label=REPRESENTATION_MODEL_LABELS[key])
            for key in model_keys
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Configuration-induced task-feature drift (seed 42)",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(figure, output_path, dpi)
    plt.close(figure)


def _mean_metric(frame: pd.DataFrame, model: str, condition: str, metric: str) -> float:
    selected = frame[(frame["model"] == model) & (frame["condition"] == condition)]
    if selected.empty:
        return float("nan")
    return float(selected[metric].mean())


def render_representation_summary(
    metric_frame: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required for this analysis") from error

    model_keys = ("full_scmc", "no_subset_calibration")
    colors = ("#1769AA", "#E76F51")
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    width = 0.36

    axis = axes[0, 0]
    x = np.arange(4)
    for index, (model_key, color) in enumerate(zip(model_keys, colors)):
        values = [
            float(metric_frame[
                (metric_frame["model"] == model_key)
                & (metric_frame["condition"] == "dynamic")
                & (metric_frame["task"] == task)
            ]["cosine_distance_mean"].iloc[0])
            for task in REPRESENTATION_TASK_ABBREVIATIONS
        ]
        axis.bar(x + (index - 0.5) * width, values, width, color=color,
                 label=REPRESENTATION_MODEL_LABELS[model_key])
    axis.set_xticks(x, REPRESENTATION_TASK_ABBREVIATIONS)
    axis.set_ylabel("Mean cosine distance")
    axis.set_title("(a) Full vs SSR-selected subset")

    missing_conditions = [f"missing_{name}" for name in MODALITY_NAMES]
    missing_labels = ["Face", "Body", "Scene", "Posture", "Gesture"]
    axis = axes[0, 1]
    x = np.arange(5)
    for index, (model_key, color) in enumerate(zip(model_keys, colors)):
        values = [
            _mean_metric(metric_frame, model_key, condition, "cosine_distance_mean")
            for condition in missing_conditions
        ]
        axis.bar(x + (index - 0.5) * width, values, width, color=color)
    axis.set_xticks(x, missing_labels, rotation=20)
    axis.set_ylabel("Mean cosine distance")
    axis.set_title("(b) Full vs single-missing available-all")

    axis = axes[1, 0]
    for index, (model_key, color) in enumerate(zip(model_keys, colors)):
        values = [
            100.0 * _mean_metric(metric_frame, model_key, condition, "prediction_agreement")
            for condition in missing_conditions
        ]
        axis.bar(x + (index - 0.5) * width, values, width, color=color)
    axis.set_xticks(x, missing_labels, rotation=20)
    axis.set_ylabel("Prediction agreement (%)")
    axis.set_ylim(0, 101)
    axis.set_title("(c) Full-prediction stability")

    axis = axes[1, 1]
    rerouted_conditions = [f"rerouted_missing_{name}" for name in MODALITY_NAMES]
    for index, (model_key, color) in enumerate(zip(model_keys, colors)):
        values = [
            _mean_metric(metric_frame, model_key, condition, "accuracy_delta_pp")
            for condition in rerouted_conditions
        ]
        axis.bar(x + (index - 0.5) * width, values, width, color=color)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, missing_labels, rotation=20)
    axis.set_ylabel("Mean task-accuracy change (%p)")
    axis.set_title("(d) Known-missing SSR re-routing")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=2, frameon=False
    )
    figure.suptitle(
        "Representation and prediction stability across configurations (seed 42)",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(figure, output_path, dpi)
    plt.close(figure)


def _fit_paired_tsne(
    reference: np.ndarray,
    shifted: np.ndarray,
    perplexity: float,
    iterations: int,
    random_state: int,
) -> Tuple[np.ndarray, float]:
    """Jointly embed paired configurations using deterministic, cosine-aligned t-SNE."""
    combined = np.concatenate((reference, shifted), axis=0).astype(np.float64, copy=False)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.clip(norms, 1e-12, None)
    total_points = int(combined.shape[0])
    if total_points < 4:
        raise ValueError("Paired t-SNE needs at least two samples per configuration.")
    # sklearn requires perplexity < n_samples.  L2-normalized Euclidean geometry
    # is monotonic with cosine distance, which is the quantitative drift metric.
    effective_perplexity = min(float(perplexity), float(total_points - 1))
    common_kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "random",
        "random_state": int(random_state),
        "metric": "euclidean",
        "method": "barnes_hut",
    }
    try:
        estimator = TSNE(max_iter=int(iterations), **common_kwargs)
    except TypeError:  # scikit-learn < 1.5 uses n_iter.
        estimator = TSNE(n_iter=int(iterations), **common_kwargs)
    return estimator.fit_transform(combined), effective_perplexity


def render_tsne_pair_figure(
    results_by_model: dict,
    condition_name: str,
    output_path: Path,
    dpi: int,
    max_samples: int,
    perplexity: float,
    iterations: int,
    random_state: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required for this analysis") from error

    model_keys = ("full_scmc", "no_subset_calibration")
    figure, axes = plt.subplots(2, 4, figsize=(15.2, 7.0))
    for model_index, model_key in enumerate(model_keys):
        result = results_by_model[model_key]
        full_features = result["conditions"]["full"]["features"]
        condition_features = result["conditions"][condition_name]["features"]
        sample_count = min(full_features.shape[0], max_samples)
        for task_index, abbreviation in enumerate(REPRESENTATION_TASK_ABBREVIATIONS):
            axis = axes[model_index, task_index]
            reference = full_features[:sample_count, task_index]
            shifted = condition_features[:sample_count, task_index]
            projected, effective_perplexity = _fit_paired_tsne(
                reference,
                shifted,
                perplexity=perplexity,
                iterations=iterations,
                random_state=random_state,
            )
            reference_2d = projected[:sample_count]
            shifted_2d = projected[sample_count:]
            for sample_index in range(min(sample_count, 60)):
                axis.plot(
                    [reference_2d[sample_index, 0], shifted_2d[sample_index, 0]],
                    [reference_2d[sample_index, 1], shifted_2d[sample_index, 1]],
                    color="#808080", alpha=0.14, linewidth=0.45, zorder=1,
                )
            axis.scatter(
                reference_2d[:, 0], reference_2d[:, 1], s=12,
                c="#1769AA", alpha=0.58, marker="o", edgecolors="none", zorder=2,
            )
            axis.scatter(
                shifted_2d[:, 0], shifted_2d[:, 1], s=13,
                c="#E76F51", alpha=0.58, marker="^", edgecolors="none", zorder=3,
            )
            axis.set_title(
                f"{abbreviation} (perplexity={effective_perplexity:g})",
                fontweight="bold",
            )
            if task_index == 0:
                axis.set_ylabel(REPRESENTATION_MODEL_LABELS[model_key])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine_name in ("top", "right", "bottom", "left"):
                axis.spines[spine_name].set_visible(False)
    condition_label = condition_name.replace("missing_", "missing ").replace("_", " ")
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#1769AA",
                   markersize=7, label="Full configuration"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#E76F51",
                   markersize=7, label=condition_label.title()),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False,
    )
    figure.suptitle(
        f"Paired task-feature t-SNE: Full vs {condition_label} (seed 42)",
        y=0.995, fontsize=14, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(figure, output_path, dpi)
    plt.close(figure)


def run_configuration_representation_analysis(args, device: torch.device) -> dict:
    """Use completed seed-42 checkpoints; never train or update model weights."""
    seed_everything(42)
    full_run_dir = resolve_internal_ablation_run(args.full_run_dir, "full", 42)
    no_calibration_run_dir = resolve_internal_ablation_run(
        args.no_calibration_run_dir, "no_subset_calibration", 42
    )
    full_args = representation_args_from_saved_run(args, full_run_dir, True)
    no_calibration_args = representation_args_from_saved_run(
        args, no_calibration_run_dir, False
    )
    if full_args.split_dir.resolve() != no_calibration_args.split_dir.resolve():
        raise ValueError(
            "Full and w/o calibration runs use different split directories: "
            f"{full_args.split_dir} vs {no_calibration_args.split_dir}"
        )
    matched_settings = (
        "t_target", "stride", "face_pad_top", "face_pad_bottom",
        "face_pad_left", "face_pad_right", "embed", "d_model",
        "preview_frames", "preview_size", "attention_heads",
    )
    mismatched = {
        key: (getattr(full_args, key), getattr(no_calibration_args, key))
        for key in matched_settings
        if getattr(full_args, key) != getattr(no_calibration_args, key)
    }
    if mismatched:
        raise ValueError(
            "Full and w/o calibration checkpoints are not protocol-matched: "
            f"{mismatched}"
        )

    output_dir = (
        args.analysis_output_dir.expanduser().resolve()
        if args.analysis_output_dir is not None
        else full_run_dir.parent / "configuration_representation_seed42"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(full_args, cache_order=True)
    test_loader = loaders["test"]

    results_by_model = {}
    variant_specs = (
        ("full_scmc", full_args),
        ("no_subset_calibration", no_calibration_args),
    )
    for model_key, variant_args in variant_specs:
        print("\n" + "=" * 88)
        print(f"representation analysis: {REPRESENTATION_MODEL_LABELS[model_key]}")
        print(f"run_dir: {variant_args.run_dir}")
        print("=" * 88)
        model, router = load_representation_variant(variant_args, device)
        results_by_model[model_key] = collect_configuration_representations(
            variant_args,
            model,
            router,
            test_loader,
            device,
            max_samples=args.analysis_max_samples,
        )
        if args.save_feature_arrays:
            save_representation_arrays(
                output_dir / f"features_{model_key}_seed42.npz",
                results_by_model[model_key],
            )
        del model, router
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    distributions_by_model = {}
    for model_key, result in results_by_model.items():
        model_rows, distributions = representation_metric_rows(model_key, result)
        rows.extend(model_rows)
        distributions_by_model[model_key] = distributions
    metric_frame = pd.DataFrame(rows)
    csv_path = output_dir / "configuration_representation_metrics_seed42.csv"
    metric_frame.to_csv(csv_path, index=False)

    violin_path = output_dir / "feature_drift_violin_seed42.png"
    summary_figure_path = output_dir / "representation_stability_summary_seed42.png"
    tsne_dynamic_path = output_dir / "tsne_full_vs_dynamic_seed42.png"
    tsne_scene_path = output_dir / "tsne_full_vs_missing_scene_seed42.png"
    render_feature_drift_violin(distributions_by_model, violin_path, args.analysis_dpi)
    render_representation_summary(metric_frame, summary_figure_path, args.analysis_dpi)
    render_tsne_pair_figure(
        results_by_model,
        "dynamic",
        tsne_dynamic_path,
        args.analysis_dpi,
        args.tsne_samples,
        args.tsne_perplexity,
        args.tsne_iterations,
        args.tsne_seed,
    )
    render_tsne_pair_figure(
        results_by_model,
        "missing_scene",
        tsne_scene_path,
        args.analysis_dpi,
        args.tsne_samples,
        args.tsne_perplexity,
        args.tsne_iterations,
        args.tsne_seed,
    )

    payload = {
        "experiment": "COVA-MTL seed-42 configuration-induced representation analysis",
        "training_was_run": False,
        "weights_were_updated": False,
        "seed": 42,
        "full_run_dir": str(full_run_dir),
        "no_subset_calibration_run_dir": str(no_calibration_run_dir),
        "split_dir": str(full_args.split_dir),
        "sample_count": {
            key: int(value["sample_count"]) for key, value in results_by_model.items()
        },
        "comparison_protocol": (
            "Models are independently trained. Distances are computed only within each "
            "model between paired Full and changed-configuration features; latent "
            "coordinates are never directly compared across models."
        ),
        "tsne_protocol": {
            "role": (
                "qualitative visualization only; global inter-cluster distances and "
                "apparent cluster sizes are not interpreted as quantitative evidence"
            ),
            "joint_embedding": "paired Full and changed-configuration samples",
            "input_preprocessing": "per-vector L2 normalization",
            "sample_limit_per_configuration": int(args.tsne_samples),
            "requested_perplexity": float(args.tsne_perplexity),
            "iterations": int(args.tsne_iterations),
            "random_state": int(args.tsne_seed),
        },
        "conditions": [
            "dynamic SSR-selected subset",
            "single-missing available-all for each modality",
            "single-missing SSR-rerouted subset for each modality",
        ],
        "metrics": [
            "paired cosine distance", "Jensen-Shannon divergence",
            "Full-prediction agreement", "task accuracy change",
        ],
        "artifacts": {
            "metrics_csv": str(csv_path),
            "feature_drift_violin_png": str(violin_path),
            "feature_drift_violin_pdf": str(violin_path.with_suffix(".pdf")),
            "summary_png": str(summary_figure_path),
            "summary_pdf": str(summary_figure_path.with_suffix(".pdf")),
            "tsne_dynamic_png": str(tsne_dynamic_path),
            "tsne_dynamic_pdf": str(tsne_dynamic_path.with_suffix(".pdf")),
            "tsne_missing_scene_png": str(tsne_scene_path),
            "tsne_missing_scene_pdf": str(tsne_scene_path.with_suffix(".pdf")),
        },
        "rows": rows,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.save_feature_arrays:
        payload["artifacts"].update({
            "full_feature_arrays": str(output_dir / "features_full_scmc_seed42.npz"),
            "no_calibration_feature_arrays": str(
                output_dir / "features_no_subset_calibration_seed42.npz"
            ),
        })
    json_path = output_dir / "configuration_representation_summary_seed42.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print("\nNo training was run. Seed-42 checkpoints were evaluated read-only.")
    print(f"representation metrics: {csv_path}")
    print(f"summary metadata: {json_path}")
    print(f"figures: {output_dir}")
    return payload


# -----------------------------------------------------------------------------
# Seed-42 modality-wise token analysis (inference only)
# -----------------------------------------------------------------------------
MODALITY_DISPLAY_NAMES = ("Face", "Body", "Scene", "Posture", "Gesture")
MODALITY_COLORS = ("#D9553F", "#62B6CB", "#3A9D8F", "#334E8A", "#F1A085")
MODALITY_MARKERS = ("o", "s", "^", "D", "P")


def load_modality_analysis_model(args, device: torch.device) -> COVAMTLSCMC:
    """Load a completed sparse Student without constructing or updating a router."""
    if not args.sparse_ckpt.exists():
        raise FileNotFoundError(f"Missing sparse Student checkpoint: {args.sparse_ckpt}")
    model = model_from_args(args, device)
    load_checkpoint(args.sparse_ckpt, model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def collect_modality_token_features(
    args,
    model: COVAMTLSCMC,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 0,
) -> dict:
    """Collect one temporally pooled encoder token per clip and modality."""
    model.eval()
    raw_store, output_store, label_store = [], [], []
    sample_count = 0
    progress = tqdm(loader, desc=f"modality tokens {args.run_dir.name}", leave=False)
    for batch in progress:
        data = move_batch(batch, device)
        if max_samples > 0:
            remaining = max_samples - sample_count
            if remaining <= 0:
                break
            if data[0].size(0) > remaining:
                data = tuple(tensor[:remaining] for tensor in data)
        kp, context, body, face = data[:4]
        labels = torch.stack(labels_tuple(data), dim=1)
        scout_tokens, raw_tokens = model.encode_all(kp, context, body, face)
        full_mask = torch.ones(
            raw_tokens.size(0), len(MODALITY_NAMES),
            dtype=torch.bool, device=raw_tokens.device,
        )
        if model.use_scmc:
            output_tokens = model.scmc(raw_tokens, scout_tokens, full_mask)
        else:
            output_tokens = raw_tokens

        # [B,T,M,E] -> [B,M,E].  Temporal averaging matches the clip-level
        # visualization goal while keeping the five modality identities intact.
        raw_store.append(raw_tokens.mean(1).detach().float().cpu().numpy())
        output_store.append(output_tokens.mean(1).detach().float().cpu().numpy())
        label_store.append(labels.detach().cpu().numpy())
        sample_count += int(raw_tokens.size(0))

    if not raw_store:
        raise RuntimeError("No test samples were collected for modality-wise analysis.")
    return {
        "raw": np.concatenate(raw_store, axis=0),
        "output": np.concatenate(output_store, axis=0),
        "task_labels": np.concatenate(label_store, axis=0),
        "sample_count": int(sample_count),
    }


def _l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def _flatten_modality_tokens(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.ndim != 3 or features.shape[1] != len(MODALITY_NAMES):
        raise ValueError(
            "Expected modality tokens with shape [samples, 5, embedding], "
            f"got {features.shape}."
        )
    flat = features.reshape(-1, features.shape[-1])
    modality_ids = np.tile(np.arange(len(MODALITY_NAMES)), features.shape[0])
    return _l2_normalize_rows(flat), modality_ids


def modality_geometry_rows(variant: str, token_space: str, features: np.ndarray) -> List[dict]:
    """Compute cluster diagnostics in the original embedding space, not t-SNE."""
    flat, modality_ids = _flatten_modality_tokens(features)
    centroids = []
    intra_by_modality = []
    for modality_id in range(len(MODALITY_NAMES)):
        group = flat[modality_ids == modality_id]
        centroid = _l2_normalize_rows(group.mean(0, keepdims=True))[0]
        centroids.append(centroid)
        intra_by_modality.append(float(np.mean(1.0 - group @ centroid)))
    centroids = np.stack(centroids, axis=0)
    centroid_cosine_distance = 1.0 - np.clip(centroids @ centroids.T, -1.0, 1.0)
    upper = centroid_cosine_distance[np.triu_indices(len(MODALITY_NAMES), k=1)]
    mean_intra = float(np.mean(intra_by_modality))
    mean_inter = float(np.mean(upper))
    rows = [{
        "variant": variant,
        "token_space": token_space,
        "scope": "overall",
        "modality": "all",
        "sample_count": int(features.shape[0]),
        "point_count": int(flat.shape[0]),
        "silhouette_cosine": float(
            silhouette_score(flat, modality_ids, metric="cosine")
        ),
        "davies_bouldin_l2_normalized": float(
            davies_bouldin_score(flat, modality_ids)
        ),
        "mean_intra_modality_cosine_distance": mean_intra,
        "mean_inter_centroid_cosine_distance": mean_inter,
        "inter_to_intra_ratio": mean_inter / max(mean_intra, 1e-12),
    }]
    for modality_id, modality_name in enumerate(MODALITY_NAMES):
        other_distances = np.delete(centroid_cosine_distance[modality_id], modality_id)
        rows.append({
            "variant": variant,
            "token_space": token_space,
            "scope": "per_modality",
            "modality": modality_name,
            "sample_count": int(features.shape[0]),
            "point_count": int(features.shape[0]),
            "silhouette_cosine": np.nan,
            "davies_bouldin_l2_normalized": np.nan,
            "mean_intra_modality_cosine_distance": intra_by_modality[modality_id],
            "mean_inter_centroid_cosine_distance": float(np.mean(other_distances)),
            "nearest_other_centroid_cosine_distance": float(np.min(other_distances)),
        })
    return rows


def fit_modality_tsne(
    features: np.ndarray,
    clip_indices: np.ndarray,
    perplexity: float,
    iterations: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    selected = features[clip_indices]
    flat, modality_ids = _flatten_modality_tokens(selected)
    if flat.shape[0] < 6:
        raise ValueError("Modality-wise t-SNE needs at least two clips.")
    effective_perplexity = min(float(perplexity), float(flat.shape[0] - 1))
    common_kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "random",
        "random_state": int(random_state),
        "metric": "euclidean",
        "method": "barnes_hut",
    }
    try:
        estimator = TSNE(max_iter=int(iterations), **common_kwargs)
    except TypeError:  # scikit-learn < 1.5
        estimator = TSNE(n_iter=int(iterations), **common_kwargs)
    return estimator.fit_transform(flat), modality_ids, effective_perplexity


def render_modality_tsne_comparison(
    panel_specs: Sequence[Tuple[str, np.ndarray]],
    output_path: Path,
    dpi: int,
    max_samples: int,
    perplexity: float,
    iterations: int,
    random_state: int,
) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required for this analysis") from error

    common_sample_count = min(spec[1].shape[0] for spec in panel_specs)
    display_count = min(int(max_samples), common_sample_count)
    rng = np.random.default_rng(int(random_state))
    clip_indices = np.sort(
        rng.choice(common_sample_count, size=display_count, replace=False)
    )
    figure, axes = plt.subplots(1, len(panel_specs), figsize=(13.8, 5.8), squeeze=False)
    metadata = {}
    for panel_index, (panel_title, features) in enumerate(panel_specs):
        axis = axes[0, panel_index]
        projected, modality_ids, effective_perplexity = fit_modality_tsne(
            features,
            clip_indices,
            perplexity=perplexity,
            iterations=iterations,
            random_state=random_state,
        )
        for modality_id, modality_name in enumerate(MODALITY_DISPLAY_NAMES):
            selected = modality_ids == modality_id
            axis.scatter(
                projected[selected, 0], projected[selected, 1],
                s=16, alpha=0.72, color=MODALITY_COLORS[modality_id],
                marker=MODALITY_MARKERS[modality_id], edgecolors="none",
                rasterized=True,
            )
        axis.set_title(panel_title, fontsize=13, fontweight="bold")
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.grid(alpha=0.16, linewidth=0.6)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)
        metadata[panel_title] = {
            "displayed_clips": int(display_count),
            "displayed_points": int(display_count * len(MODALITY_NAMES)),
            "effective_perplexity": float(effective_perplexity),
        }
    legend_handles = [
        Line2D(
            [0], [0], linestyle="none", marker=MODALITY_MARKERS[index],
            markerfacecolor=MODALITY_COLORS[index], markeredgecolor="none",
            markersize=7, label=name,
        )
        for index, name in enumerate(MODALITY_DISPLAY_NAMES)
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=5, frameon=True, fontsize=10,
    )
    figure.suptitle(
        "Modality-wise token distributions on AIDE (seed 42)",
        y=0.985, fontsize=15, fontweight="bold",
    )
    figure.text(
        0.5, 0.095,
        "Each panel is embedded independently; t-SNE axes and inter-cluster gaps are qualitative.",
        ha="center", va="center", fontsize=8.5, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.15, 1, 0.93))
    _save_figure(figure, output_path, dpi)
    plt.close(figure)
    return metadata


def save_modality_arrays(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        raw=result["raw"].astype(np.float32),
        output=result["output"].astype(np.float32),
        task_labels=result["task_labels"].astype(np.int64),
        modality_names=np.asarray(MODALITY_NAMES),
    )


def run_modalitywise_tsne_analysis(args, device: torch.device) -> dict:
    """Read completed seed-42 Students and generate modality-wise t-SNE only."""
    seed_everything(42)
    full_run_dir = resolve_internal_ablation_run(args.full_run_dir, "full", 42)
    no_calibration_run_dir = resolve_internal_ablation_run(
        args.no_calibration_run_dir, "no_subset_calibration", 42
    )
    full_args = representation_args_from_saved_run(args, full_run_dir, True)
    no_calibration_args = representation_args_from_saved_run(
        args, no_calibration_run_dir, False
    )
    matched_settings = (
        "split_dir", "t_target", "stride", "face_pad_top", "face_pad_bottom",
        "face_pad_left", "face_pad_right", "embed", "preview_frames",
        "preview_size",
    )
    mismatched = {
        key: (getattr(full_args, key), getattr(no_calibration_args, key))
        for key in matched_settings
        if getattr(full_args, key) != getattr(no_calibration_args, key)
    }
    if mismatched:
        raise ValueError(
            "Full and w/o-calibration checkpoints are not protocol-matched: "
            f"{mismatched}"
        )

    output_dir = (
        args.modality_output_dir.expanduser().resolve()
        if args.modality_output_dir is not None
        else full_run_dir.parent / "modalitywise_tsne_seed42"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(full_args, cache_order=True)
    try:
        results = {}
        for variant_key, variant_args in (
            ("no_subset_calibration", no_calibration_args),
            ("full_scmc", full_args),
        ):
            print("\n" + "=" * 88)
            print(f"modality-wise analysis: {REPRESENTATION_MODEL_LABELS[variant_key]}")
            print(f"checkpoint: {variant_args.sparse_ckpt}")
            print("=" * 88)
            model = load_modality_analysis_model(variant_args, device)
            results[variant_key] = collect_modality_token_features(
                variant_args,
                model,
                loaders["test"],
                device,
                max_samples=args.analysis_max_samples,
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        shutdown_loaders(loaders)

    metric_rows = []
    metric_rows.extend(modality_geometry_rows(
        "w/o subset calibration", "fusion-input tokens",
        results["no_subset_calibration"]["output"],
    ))
    metric_rows.extend(modality_geometry_rows(
        "Full SCML", "before SCMC",
        results["full_scmc"]["raw"],
    ))
    metric_rows.extend(modality_geometry_rows(
        "Full SCML", "after SCMC",
        results["full_scmc"]["output"],
    ))
    metric_frame = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "modalitywise_feature_metrics_seed42.csv"
    metric_frame.to_csv(metrics_path, index=False)

    ablation_figure_path = (
        output_dir / "modalitywise_tsne_full_vs_no_calibration_seed42.png"
    )
    before_after_figure_path = (
        output_dir / "modalitywise_tsne_full_before_after_scmc_seed42.png"
    )
    ablation_tsne = render_modality_tsne_comparison(
        (
            ("(a) w/o subset calibration", results["no_subset_calibration"]["output"]),
            ("(b) Full SCML", results["full_scmc"]["output"]),
        ),
        ablation_figure_path,
        dpi=args.analysis_dpi,
        max_samples=args.tsne_samples,
        perplexity=args.tsne_perplexity,
        iterations=args.tsne_iterations,
        random_state=args.tsne_seed,
    )
    before_after_tsne = render_modality_tsne_comparison(
        (
            ("(a) Full SCML before SCMC", results["full_scmc"]["raw"]),
            ("(b) Full SCML after SCMC", results["full_scmc"]["output"]),
        ),
        before_after_figure_path,
        dpi=args.analysis_dpi,
        max_samples=args.tsne_samples,
        perplexity=args.tsne_perplexity,
        iterations=args.tsne_iterations,
        random_state=args.tsne_seed,
    )

    artifacts = {
        "metrics_csv": str(metrics_path),
        "ablation_tsne_png": str(ablation_figure_path),
        "ablation_tsne_pdf": str(ablation_figure_path.with_suffix(".pdf")),
        "full_before_after_tsne_png": str(before_after_figure_path),
        "full_before_after_tsne_pdf": str(before_after_figure_path.with_suffix(".pdf")),
    }
    if args.save_feature_arrays:
        full_array_path = output_dir / "modality_tokens_full_scmc_seed42.npz"
        no_cal_array_path = output_dir / "modality_tokens_no_calibration_seed42.npz"
        save_modality_arrays(full_array_path, results["full_scmc"])
        save_modality_arrays(no_cal_array_path, results["no_subset_calibration"])
        artifacts.update({
            "full_token_arrays": str(full_array_path),
            "no_calibration_token_arrays": str(no_cal_array_path),
        })

    payload = {
        "experiment": "COVA-MTL seed-42 modality-wise token distribution analysis",
        "training_was_run": False,
        "weights_were_updated": False,
        "seed": 42,
        "full_run_dir": str(full_run_dir),
        "no_subset_calibration_run_dir": str(no_calibration_run_dir),
        "split_dir": str(full_args.split_dir),
        "feature_definition": (
            "temporally averaged modality encoder tokens immediately before "
            "mask-aware task-specific fusion; Full output applies SCMC under the "
            "all-available mask"
        ),
        "sample_count": {
            key: int(value["sample_count"]) for key, value in results.items()
        },
        "tsne_protocol": {
            "qualitative_only": True,
            "each_panel_embedded_independently": True,
            "input_preprocessing": "per-token L2 normalization",
            "requested_samples_per_modality": int(args.tsne_samples),
            "requested_perplexity": float(args.tsne_perplexity),
            "iterations": int(args.tsne_iterations),
            "random_state": int(args.tsne_seed),
            "ablation_panels": ablation_tsne,
            "full_before_after_panels": before_after_tsne,
        },
        "interpretation_scope": (
            "Modality separation indicates retained modality identity, not recognition "
            "accuracy. Silhouette and centroid metrics are computed in the original "
            "embedding space and are the quantitative companions to t-SNE."
        ),
        "artifacts": artifacts,
        "metric_rows": metric_rows,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = output_dir / "modalitywise_tsne_summary_seed42.json"
    write_json(summary_path, payload)
    print("\nNo training was run. Seed-42 sparse Students were read only.")
    print(f"modality metrics: {metrics_path}")
    print(f"summary metadata: {summary_path}")
    print(f"figures: {output_dir}")
    return payload


# -----------------------------------------------------------------------------
# Plain shared-encoder baseline and 2x2 qualitative comparison (seed 42)
# -----------------------------------------------------------------------------
class PlainSharedRecognitionBaseline(nn.Module):
    """Minimal dense MTL baseline with no COVA/SCML routing components.

    Face, body, and scene frames share one standard ResNet-18 including BN.
    Posture and gesture cannot be passed to a 2-D CNN, so their coordinates are
    zero-padded to one common input width and processed by one shared MLP.
    Temporal aggregation and modality fusion are both non-parametric means.
    """

    def __init__(self, embed: int = 128, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.visual_encoder = backbone
        self.visual_projection = nn.Linear(512, embed)
        self.vector_encoder = nn.Sequential(
            nn.Linear(3 * 42, 256), nn.ReLU(inplace=True),
            nn.Linear(256, embed),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(embed, classes) for classes in TASK_CLASSES]
        )
        self.embed = int(embed)

    def encode_modalities(self, kp, context, body, face) -> torch.Tensor:
        batch, _, steps = context.shape[:3]
        # [B,3 modalities,C,T,H,W] -> one combined standard-BN batch.
        visual = torch.stack((face, body, context), dim=1)
        visual = visual.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * 3 * steps, 3, context.size(-2), context.size(-1)
        )
        visual = self.visual_projection(self.visual_encoder(visual))
        visual = visual.reshape(batch, 3, steps, self.embed).mean(2)

        posture = kp[:, :, :, :26, :].squeeze(-1).permute(0, 2, 1, 3).reshape(
            batch, steps, 3 * 26
        )
        posture = F.pad(posture, (0, 3 * (42 - 26)))
        gesture = kp[:, :, :, 94:136, :].squeeze(-1).permute(0, 2, 1, 3).reshape(
            batch, steps, 3 * 42
        )
        vectors = torch.stack((posture, gesture), dim=1).reshape(
            batch * 2 * steps, 3 * 42
        )
        vectors = self.vector_encoder(vectors).reshape(
            batch, 2, steps, self.embed
        ).mean(2)
        return torch.cat((visual, vectors), dim=1)

    def encode_selected_modalities(
        self, kp, context, body, face, mask: torch.Tensor
    ) -> torch.Tensor:
        """Execute only the sample-wise active branches of a dense-trained model.

        This method is deliberately different from encoding all five streams and
        hiding points only at plot time.  For every modality, only clips whose
        COVA-MTL mask activates that modality are sent through the shared encoder;
        inactive token slots remain zero and are excluded from fusion and t-SNE.
        The model is still the same dense-only checkpoint used in panel (c).
        """
        mask = mask.bool()
        batch, _, steps = context.shape[:3]
        if mask.shape != (batch, len(MODALITY_NAMES)):
            raise ValueError(
                f"Expected mask [{batch},{len(MODALITY_NAMES)}], got {tuple(mask.shape)}"
            )
        if (~mask).all(1).any():
            raise ValueError("Every clip must retain at least one modality.")

        tokens = context.new_zeros(batch, len(MODALITY_NAMES), self.embed)
        visual_streams = (face, body, context)
        for modality_id, stream in enumerate(visual_streams):
            active = torch.nonzero(mask[:, modality_id], as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            selected = stream.index_select(0, active)
            selected = selected.permute(0, 2, 1, 3, 4).reshape(
                active.numel() * steps, 3, stream.size(-2), stream.size(-1)
            )
            encoded = self.visual_projection(self.visual_encoder(selected))
            encoded = encoded.reshape(active.numel(), steps, self.embed).mean(1)
            tokens[active, modality_id] = encoded

        posture = kp[:, :, :, :26, :].squeeze(-1).permute(0, 2, 1, 3).reshape(
            batch, steps, 3 * 26
        )
        posture = F.pad(posture, (0, 3 * (42 - 26)))
        gesture = kp[:, :, :, 94:136, :].squeeze(-1).permute(0, 2, 1, 3).reshape(
            batch, steps, 3 * 42
        )
        for modality_id, stream in ((3, posture), (4, gesture)):
            active = torch.nonzero(mask[:, modality_id], as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            selected = stream.index_select(0, active).reshape(
                active.numel() * steps, 3 * 42
            )
            encoded = self.vector_encoder(selected).reshape(
                active.numel(), steps, self.embed
            ).mean(1)
            tokens[active, modality_id] = encoded
        return tokens

    def predict(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        if mask is None:
            mask = torch.ones(
                tokens.size(0), tokens.size(1), dtype=torch.bool,
                device=tokens.device,
            )
        if (~mask.bool()).all(1).any():
            raise ValueError("Plain baseline requires at least one available modality.")
        value = mask.to(tokens.dtype)
        fused = (tokens * value[:, :, None]).sum(1) / value.sum(
            1, keepdim=True
        ).clamp_min(1.0)
        logits = tuple(head(fused) for head in self.heads)
        return logits, fused

    def forward(self, kp, context, body, face):
        return self.predict(self.encode_modalities(kp, context, body, face))[0]


def plain_baseline_paths(args) -> dict:
    root = args.plain_run_dir
    return {
        "root": root,
        "last": root / "checkpoints" / "last.pt",
        "best": root / "checkpoints" / "bestmacc.pt",
        "config": root / "plain_baseline_config.json",
        "history": root / "plain_baseline_history.csv",
    }


@torch.no_grad()
def evaluate_plain_baseline(
    model: PlainSharedRecognitionBaseline,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> dict:
    model.eval()
    store = new_prediction_store()
    for batch in tqdm(loader, desc=desc, leave=False):
        data = move_batch(batch, device)
        tokens = model.encode_modalities(*data[:4])
        logits, _ = model.predict(tokens)
        update_predictions(store, logits, labels_tuple(data))
    return finish_metrics(store)


def train_plain_baseline_once(args, loaders, device: torch.device) -> Path:
    """Train one seed-42 dense-only baseline, then reuse its best checkpoint."""
    paths = plain_baseline_paths(args)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["best"].parent.mkdir(parents=True, exist_ok=True)
    if paths["best"].exists() and not args.force_retrain_plain:
        print(f"Plain baseline already exists; skip training: {paths['best']}")
        return paths["best"]

    seed_everything(42)
    model = PlainSharedRecognitionBaseline(
        embed=args.embed, pretrained=args.pretrained
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.plain_lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_macc, history = 1, -1.0, []
    if args.resume and paths["last"].exists() and not args.force_retrain_plain:
        checkpoint = load_checkpoint(paths["last"], model, optimizer, scaler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macc = float(checkpoint.get("best", {}).get("macc", -1.0))
        if paths["history"].exists():
            history = pd.read_csv(paths["history"]).to_dict("records")

    write_json(paths["config"], {
        "experiment": "plain shared ResNet-18 dense recognition baseline",
        "seed": 42,
        "split_seed": int(args.split_seed),
        "data_root": str(args.data_root),
        "split_dir": str(args.split_dir),
        "visual_encoder": "one shared standard ResNet-18 for face/body/scene",
        "vector_encoder": "one shared two-layer MLP for padded posture/gesture",
        "temporal_reducer": "mean",
        "modality_fusion": "mean",
        "removed": [
            "PS-MSBN", "stage attention", "learned temporal reducer", "SCMC",
            "mask-aware task attention", "scout", "ACSR/SRCR", "subset training",
            "availability dropout", "knowledge distillation",
        ],
        "epochs": int(args.plain_epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.plain_lr),
        "weight_decay": float(args.weight_decay),
    })

    for epoch in range(start_epoch, args.plain_epochs + 1):
        model.train()
        running_loss, batches = 0.0, 0
        for batch in tqdm(
            loaders["train"], desc=f"plain baseline {epoch}", leave=False
        ):
            data = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                tokens = model.encode_modalities(*data[:4])
                logits, _ = model.predict(tokens)
                loss = multitask_ce(logits, labels_tuple(data))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            batches += 1

        validation = evaluate_plain_baseline(
            model, loaders["val"], device, f"plain val {epoch}"
        )
        row = {
            "epoch": int(epoch),
            "train_loss": running_loss / max(batches, 1),
            "val_macc": float(validation["macc"]),
            "val_mf1": float(validation["mf1"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(paths["history"], index=False)
        best_payload = {"macc": max(best_macc, validation["macc"])}
        save_checkpoint(paths["last"], epoch, model, optimizer, scaler, best_payload)
        if validation["macc"] > best_macc:
            best_macc = float(validation["macc"])
            save_checkpoint(
                paths["best"], epoch, model, optimizer, scaler,
                {"macc": best_macc, "mf1": float(validation["mf1"])},
            )
        print(
            f"plain epoch {epoch:03d}: loss={row['train_loss']:.4f} "
            f"val_mAcc={validation['macc']:.4f} val_mF1={validation['mf1']:.4f}"
        )
    if not paths["best"].exists():
        raise RuntimeError("Plain baseline training finished without a best checkpoint.")
    del model, optimizer, scaler
    release_seed_resources()
    return paths["best"]


def load_plain_baseline(args, device: torch.device) -> PlainSharedRecognitionBaseline:
    checkpoint_path = plain_baseline_paths(args)["best"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Plain baseline checkpoint not found: {checkpoint_path}")
    model = PlainSharedRecognitionBaseline(
        embed=args.embed, pretrained=False
    ).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def resolve_cova_seed42_run(explicit: Optional[Path]) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        Path(
            ""
            "RUNS/AIDE_COVA_MTL_v6_3_MultiSeed5/seed_42"
        ),
        Path(
            ""
            "RUNSAIDE_COVA_MTL_v6_3_SCML_InternalAblation/full"
        ),
        Path(
            ""
            "RUNS/AIDE_COVA_MTL_v6_3_SCML_InternalAblation/full"
        ),
    ])
    for candidate in candidates:
        resolved = _normalize_saved_run_dir(Path(candidate), 42)
        if resolved is not None:
            return resolved
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Cannot find the completed Full COVA-MTL seed-42 run. Checked:\n"
        f"{checked}\nPass the exact directory with --cova-run-dir."
    )


def flatten_masked_modality_features(
    features: np.ndarray, masks: np.ndarray, clip_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if clip_indices is not None:
        features = features[clip_indices]
        masks = masks[clip_indices]
    if features.ndim != 3 or features.shape[1] != len(MODALITY_NAMES):
        raise ValueError(f"Expected [N,5,E] features, got {features.shape}")
    masks = masks.astype(bool)
    flat_features, modality_ids = [], []
    for modality_id in range(len(MODALITY_NAMES)):
        selected = features[masks[:, modality_id], modality_id]
        if selected.size == 0:
            continue
        flat_features.append(selected)
        modality_ids.append(
            np.full(selected.shape[0], modality_id, dtype=np.int64)
        )
    if not flat_features:
        raise ValueError("The selected mask contains no modality tokens.")
    return (
        _l2_normalize_rows(np.concatenate(flat_features, axis=0)),
        np.concatenate(modality_ids, axis=0),
    )


def masked_modality_geometry_rows(
    panel: str, features: np.ndarray, masks: np.ndarray
) -> List[dict]:
    flat, modality_ids = flatten_masked_modality_features(features, masks)
    active_labels = np.unique(modality_ids)
    centroids, intra, counts = {}, {}, {}
    for modality_id in active_labels:
        group = flat[modality_ids == modality_id]
        centroid = _l2_normalize_rows(group.mean(0, keepdims=True))[0]
        centroids[int(modality_id)] = centroid
        intra[int(modality_id)] = float(np.mean(1.0 - group @ centroid))
        counts[int(modality_id)] = int(group.shape[0])
    centroid_array = np.stack([centroids[int(i)] for i in active_labels])
    centroid_distance = 1.0 - np.clip(
        centroid_array @ centroid_array.T, -1.0, 1.0
    )
    upper = centroid_distance[np.triu_indices(len(active_labels), k=1)]
    valid_for_silhouette = (
        len(active_labels) >= 2
        and all(counts[int(label)] >= 2 for label in active_labels)
        and flat.shape[0] > len(active_labels)
    )
    rows = [{
        "panel": panel,
        "scope": "overall",
        "modality": "all",
        "point_count": int(flat.shape[0]),
        "active_modality_count": int(len(active_labels)),
        "silhouette_cosine": float(
            silhouette_score(flat, modality_ids, metric="cosine")
        ) if valid_for_silhouette else np.nan,
        "davies_bouldin_l2_normalized": float(
            davies_bouldin_score(flat, modality_ids)
        ) if valid_for_silhouette else np.nan,
        "mean_intra_modality_cosine_distance": float(np.mean(list(intra.values()))),
        "mean_inter_centroid_cosine_distance": float(np.mean(upper))
        if upper.size else np.nan,
    }]
    for modality_id in range(len(MODALITY_NAMES)):
        rows.append({
            "panel": panel,
            "scope": "per_modality",
            "modality": MODALITY_NAMES[modality_id],
            "point_count": counts.get(modality_id, 0),
            "selection_rate": float(masks[:, modality_id].mean()),
            "mean_intra_modality_cosine_distance": intra.get(modality_id, np.nan),
        })
    return rows


def fit_masked_modality_tsne(
    features: np.ndarray,
    masks: np.ndarray,
    clip_indices: np.ndarray,
    perplexity: float,
    iterations: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    flat, modality_ids = flatten_masked_modality_features(
        features, masks, clip_indices
    )
    if flat.shape[0] < 6:
        raise ValueError("t-SNE requires at least six active modality points.")
    effective_perplexity = min(float(perplexity), float(flat.shape[0] - 1))
    kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "random",
        "random_state": int(random_state),
        "metric": "euclidean",
        "method": "barnes_hut",
    }
    try:
        estimator = TSNE(max_iter=int(iterations), **kwargs)
    except TypeError:
        estimator = TSNE(n_iter=int(iterations), **kwargs)
    return estimator.fit_transform(flat), modality_ids, effective_perplexity


def render_four_panel_modality_tsne(
    panels: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    args,
    individual_dir: Optional[Path] = None,
    coordinates_path: Optional[Path] = None,
) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required") from error

    common_count = min(features.shape[0] for _, features, _ in panels)
    display_count = min(args.tsne_samples, common_count)
    rng = np.random.default_rng(args.tsne_seed)
    clip_indices = np.sort(
        rng.choice(common_count, size=display_count, replace=False)
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 11.0))
    metadata = {}
    coordinate_rows = []
    panel_file_stems = (
        "panel_a_cova_all_seed42",
        "panel_b_cova_selected_seed42",
        "panel_c_plain_all_seed42",
        "panel_d_plain_selected_inference_seed42",
    )
    if individual_dir is not None:
        individual_dir.mkdir(parents=True, exist_ok=True)
    for panel_index, (axis, (title, features, masks)) in enumerate(
        zip(axes.flat, panels)
    ):
        projected, modality_ids, effective_perplexity = fit_masked_modality_tsne(
            features, masks, clip_indices, args.tsne_perplexity,
            args.tsne_iterations, args.tsne_seed,
        )
        counts = {}
        for modality_id, display_name in enumerate(MODALITY_DISPLAY_NAMES):
            selected = modality_ids == modality_id
            counts[MODALITY_NAMES[modality_id]] = int(selected.sum())
            if not selected.any():
                continue
            axis.scatter(
                projected[selected, 0], projected[selected, 1],
                s=15, alpha=0.72, color=MODALITY_COLORS[modality_id],
                marker=MODALITY_MARKERS[modality_id], edgecolors="none",
                rasterized=True,
            )
            for point_x, point_y in projected[selected]:
                coordinate_rows.append({
                    "panel": chr(ord("a") + panel_index),
                    "panel_title": title,
                    "modality": MODALITY_NAMES[modality_id],
                    "tsne_1": float(point_x),
                    "tsne_2": float(point_y),
                })
        axis.set_title(title, fontsize=12.5, fontweight="bold")
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.grid(alpha=0.15, linewidth=0.6)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        metadata[title] = {
            "displayed_clips": int(display_count),
            "displayed_points": int(len(modality_ids)),
            "effective_perplexity": float(effective_perplexity),
            "points_by_modality": counts,
        }

        # Export the exact same embedding as a clean standalone panel for PPT.
        if individual_dir is not None:
            single_figure, single_axis = plt.subplots(figsize=(6.6, 5.5))
            for modality_id, display_name in enumerate(MODALITY_DISPLAY_NAMES):
                selected = modality_ids == modality_id
                if not selected.any():
                    continue
                single_axis.scatter(
                    projected[selected, 0], projected[selected, 1],
                    s=20, alpha=0.76, color=MODALITY_COLORS[modality_id],
                    marker=MODALITY_MARKERS[modality_id], edgecolors="none",
                    rasterized=True,
                )
            single_axis.set_title(title, fontsize=13, fontweight="bold")
            single_axis.set_xlabel("t-SNE 1")
            single_axis.set_ylabel("t-SNE 2")
            single_axis.grid(alpha=0.15, linewidth=0.6)
            for spine in ("top", "right"):
                single_axis.spines[spine].set_visible(False)
            single_figure.legend(
                handles=[
                    Line2D(
                        [0], [0], linestyle="none",
                        marker=MODALITY_MARKERS[index],
                        markerfacecolor=MODALITY_COLORS[index],
                        markeredgecolor="none", markersize=7, label=name,
                    )
                    for index, name in enumerate(MODALITY_DISPLAY_NAMES)
                    if counts[MODALITY_NAMES[index]] > 0
                ],
                loc="lower center", bbox_to_anchor=(0.5, 0.005),
                ncol=5, frameon=True, fontsize=8.5,
            )
            single_figure.tight_layout(rect=(0, 0.10, 1, 1))
            _save_figure(
                single_figure,
                individual_dir / f"{panel_file_stems[panel_index]}.png",
                args.analysis_dpi,
            )
            plt.close(single_figure)
    handles = [
        Line2D(
            [0], [0], linestyle="none", marker=MODALITY_MARKERS[index],
            markerfacecolor=MODALITY_COLORS[index], markeredgecolor="none",
            markersize=7, label=name,
        )
        for index, name in enumerate(MODALITY_DISPLAY_NAMES)
    ]
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=5, frameon=True,
    )
    figure.suptitle(
        "Modality-wise feature distributions: COVA-MTL vs plain shared baseline",
        y=0.985, fontsize=15, fontweight="bold",
    )
    figure.text(
        0.5, 0.065,
        "Panels are embedded independently. Only active tokens are plotted in (b) and (d); "
        "both use the identical COVA-MTL masks.",
        ha="center", fontsize=8.8, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    _save_figure(figure, output_path, args.analysis_dpi)
    plt.close(figure)
    if coordinates_path is not None:
        pd.DataFrame(coordinate_rows).to_csv(coordinates_path, index=False)
    return metadata


TASK_DISPLAY_NAMES = ("DER", "DBR", "TCR", "VBR")
TASK_LABEL_NAMES = (
    EMOTION_LABEL,
    DRIVER_BEHAVIOR_LABEL,
    SCENE_CENTRIC_CONTEXT_LABEL,
    VEHICLE_BASED_CONTEXT_LABEL,
)


def _preview_image(stream: torch.Tensor, sample_index: int) -> np.ndarray:
    middle = stream.size(2) // 2
    image = stream[sample_index, :, middle].detach().float().cpu()
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def _preview_frames(
    stream: torch.Tensor, sample_index: int, frame_count: int = 4
) -> List[np.ndarray]:
    steps = int(stream.size(2))
    indices = np.linspace(0, max(steps - 1, 0), num=min(frame_count, steps))
    indices = np.unique(np.rint(indices).astype(np.int64))
    frames = []
    for frame_index in indices:
        image = stream[sample_index, :, int(frame_index)].detach().float().cpu()
        frames.append(np.clip(image.permute(1, 2, 0).numpy(), 0.0, 1.0))
    return frames


def _preview_keypoint_frames(
    kp: torch.Tensor,
    sample_index: int,
    start: int,
    end: int,
    frame_count: int = 4,
) -> dict:
    """Return raw [x,y,confidence] points for matching representative frames."""
    steps = int(kp.size(2))
    indices = np.linspace(0, max(steps - 1, 0), num=min(frame_count, steps))
    indices = np.unique(np.rint(indices).astype(np.int64))

    def extract(frame_index: int) -> np.ndarray:
        points = kp[
            sample_index, :, int(frame_index), start:end, 0
        ].detach().float().cpu().permute(1, 0).numpy()
        return points.astype(np.float32, copy=False)

    return {
        "representative": extract(steps // 2),
        "frames": [extract(int(frame_index)) for frame_index in indices],
        "frame_indices": indices.astype(int).tolist(),
    }


def _case_visual_payload(kp, context, body, face, sample_index: int) -> dict:
    """Collect the five modality assets from one clip using matching frames."""
    return {
        "images": {
            "face": _preview_image(face, sample_index),
            "body": _preview_image(body, sample_index),
            "scene": _preview_image(context, sample_index),
        },
        "frames": {
            "face": _preview_frames(face, sample_index),
            "body": _preview_frames(body, sample_index),
            "scene": _preview_frames(context, sample_index),
        },
        "keypoint_assets": {
            "posture": _preview_keypoint_frames(kp, sample_index, 0, 26),
            "gesture": _preview_keypoint_frames(kp, sample_index, 94, 136),
        },
    }


@torch.no_grad()
def collect_plain_cova_qualitative_data(
    args,
    cova_model: COVAMTLSCMC,
    router: AvailabilityConditionedSufficientRouter,
    plain_model: PlainSharedRecognitionBaseline,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    cova_model.eval()
    router.eval()
    plain_model.eval()
    cova_full_features, cova_selected_features = [], []
    plain_features, plain_selected_features = [], []
    selected_masks, labels_store = [], []
    cova_prediction_store, plain_selected_prediction_store = [], []
    best_cases = {}
    best_selection_cases = {}
    cova_dense_store = new_prediction_store()
    cova_dynamic_store = new_prediction_store()
    plain_dense_store = new_prediction_store()
    plain_selected_store = new_prediction_store()
    processed = 0
    dataset_frame = getattr(loader.dataset, "df", None)

    for batch in tqdm(loader, desc="COVA vs plain qualitative", leave=False):
        data = move_batch(batch, device)
        if args.analysis_max_samples > 0:
            remaining = args.analysis_max_samples - processed
            if remaining <= 0:
                break
            if data[0].size(0) > remaining:
                data = tuple(tensor[:remaining] for tensor in data)
        kp, context, body, face = data[:4]
        labels = labels_tuple(data)
        label_matrix = torch.stack(labels, dim=1)
        batch_size = kp.size(0)
        full_mask = torch.ones(
            batch_size, len(MODALITY_NAMES), dtype=torch.bool, device=device
        )

        scout, cova_raw = cova_model.encode_all(kp, context, body, face)
        if cova_model.use_scmc:
            cova_full = cova_model.scmc(cova_raw, scout, full_mask)
        else:
            cova_full = cova_raw
        cova_dense_logits, _, _ = cova_model.predict(
            scout, cova_raw, full_mask
        )
        route = select_self_referenced_subset(router, scout, full_mask)
        selected_mask = route["mask"]
        cova_selected_raw = cova_model.encode_selected(
            selected_mask, kp, context, body, face
        )
        cova_dynamic_logits, _, _ = cova_model.predict(
            scout, cova_selected_raw, selected_mask
        )
        if cova_model.use_scmc:
            cova_selected = cova_model.scmc(
                cova_selected_raw, scout, selected_mask
            )
        else:
            cova_selected = (
                cova_selected_raw
                * selected_mask[:, None, :, None].to(cova_selected_raw.dtype)
            )

        plain_tokens = plain_model.encode_modalities(kp, context, body, face)
        plain_dense_logits, _ = plain_model.predict(plain_tokens, full_mask)
        # Panel (d) is a genuine partial-modality inference pass.  The plain
        # checkpoint was trained densely, but only branches selected by the
        # COVA mask are executed here.
        plain_selected_tokens = plain_model.encode_selected_modalities(
            kp, context, body, face, selected_mask
        )
        plain_selected_logits, _ = plain_model.predict(
            plain_selected_tokens, selected_mask
        )
        cova_selected_prediction = torch.stack(
            [logit.argmax(1) for logit in cova_dynamic_logits], dim=1
        )
        plain_selected_prediction = torch.stack(
            [logit.argmax(1) for logit in plain_selected_logits], dim=1
        )
        cova_selected_confidence = torch.stack(
            [torch.softmax(logit.float(), dim=1).amax(1)
             for logit in cova_dynamic_logits], dim=1
        ).mean(1)
        update_predictions(cova_dense_store, cova_dense_logits, labels)
        update_predictions(cova_dynamic_store, cova_dynamic_logits, labels)
        update_predictions(plain_dense_store, plain_dense_logits, labels)
        update_predictions(plain_selected_store, plain_selected_logits, labels)

        cova_full_features.append(cova_full.mean(1).float().cpu().numpy())
        cova_selected_features.append(
            cova_selected.mean(1).float().cpu().numpy()
        )
        plain_features.append(plain_tokens.float().cpu().numpy())
        plain_selected_features.append(
            plain_selected_tokens.float().cpu().numpy()
        )
        selected_masks.append(selected_mask.cpu().numpy())
        labels_store.append(label_matrix.cpu().numpy())
        cova_prediction_store.append(cova_selected_prediction.cpu().numpy())
        plain_selected_prediction_store.append(
            plain_selected_prediction.cpu().numpy()
        )

        # Find one strong normal-routing example per target cardinality:
        # the dense-trained Plain model is wrong after actual partial inference,
        # while COVA-MTL predicts all four tasks correctly with the same mask.
        for local_index in range(batch_size):
            cardinality = int(selected_mask[local_index].sum().item())
            if cardinality not in args.target_selected_cardinalities:
                continue
            ground_truth = label_matrix[local_index]
            cova_wrong = int(
                (cova_selected_prediction[local_index] != ground_truth).sum().item()
            )
            plain_wrong = int(
                (plain_selected_prediction[local_index] != ground_truth).sum().item()
            )
            if cova_wrong != 0 or plain_wrong == 0:
                continue
            global_index = processed + local_index
            priority = (
                plain_wrong,
                float(cova_selected_confidence[local_index].item()),
                -global_index,
            )
            previous = best_selection_cases.get(cardinality)
            if previous is not None and priority <= previous["priority"]:
                continue
            source_path = None
            if dataset_frame is not None and global_index < len(dataset_frame):
                source_path = str(dataset_frame.iloc[global_index, 0])
            best_selection_cases[cardinality] = {
                "case_type": "normal_selection",
                "priority": priority,
                "cardinality": int(cardinality),
                "sample_index": int(global_index),
                "source_path": source_path,
                "ground_truth": ground_truth.detach().cpu().tolist(),
                "plain_prediction": plain_selected_prediction[
                    local_index
                ].cpu().tolist(),
                "cova_prediction": cova_selected_prediction[
                    local_index
                ].cpu().tolist(),
                "cova_selected_mask": selected_mask[
                    local_index
                ].cpu().tolist(),
                "plain_wrong_tasks": int(plain_wrong),
                "cova_wrong_tasks": int(cova_wrong),
                "cova_mean_confidence": float(
                    cova_selected_confidence[local_index].item()
                ),
                **_case_visual_payload(
                    kp, context, body, face, local_index
                ),
            }

        # Search paired known-missing cases. The plain baseline receives every
        # remaining modality; COVA-MTL re-routes under the same availability.
        for missing_id in (0, 1, 2):
            available = full_mask.clone()
            available[:, missing_id] = False
            missing_route = select_self_referenced_subset(
                router, scout, available
            )
            cova_missing_raw = cova_model.encode_selected(
                missing_route["mask"], kp, context, body, face
            )
            cova_missing_logits, _, _ = cova_model.predict(
                scout, cova_missing_raw, missing_route["mask"]
            )
            plain_missing_tokens = plain_model.encode_selected_modalities(
                kp, context, body, face, available
            )
            plain_missing_logits, _ = plain_model.predict(
                plain_missing_tokens, available
            )
            cova_prediction = torch.stack(
                [logit.argmax(1) for logit in cova_missing_logits], dim=1
            )
            plain_prediction = torch.stack(
                [logit.argmax(1) for logit in plain_missing_logits], dim=1
            )
            for local_index in range(batch_size):
                ground_truth = label_matrix[local_index]
                cova_wrong = int(
                    (cova_prediction[local_index] != ground_truth).sum().item()
                )
                plain_wrong = int(
                    (plain_prediction[local_index] != ground_truth).sum().item()
                )
                # The requested case must be unambiguous: the plain baseline
                # misses at least one task while COVA gets all four tasks right.
                if plain_wrong == 0 or cova_wrong != 0:
                    continue
                priority = (plain_wrong,)
                previous = best_cases.get(missing_id)
                if previous is not None and priority <= previous["priority"]:
                    continue
                global_index = processed + local_index
                source_path = None
                if dataset_frame is not None and global_index < len(dataset_frame):
                    source_path = str(dataset_frame.iloc[global_index, 0])
                best_cases[missing_id] = {
                    "priority": priority,
                    "sample_index": int(global_index),
                    "source_path": source_path,
                    "missing_id": int(missing_id),
                    "missing_modality": MODALITY_NAMES[missing_id],
                    "ground_truth": ground_truth.detach().cpu().tolist(),
                    "plain_prediction": plain_prediction[local_index].cpu().tolist(),
                    "cova_prediction": cova_prediction[local_index].cpu().tolist(),
                    "cova_selected_mask": missing_route["mask"][local_index].cpu().tolist(),
                    "plain_wrong_tasks": int(plain_wrong),
                    "cova_wrong_tasks": int(cova_wrong),
                    **_case_visual_payload(
                        kp, context, body, face, local_index
                    ),
                }
        processed += batch_size

    if not cova_full_features:
        raise RuntimeError("No samples were collected from the test split.")
    result = {
        "cova_full": np.concatenate(cova_full_features, axis=0),
        "cova_selected": np.concatenate(cova_selected_features, axis=0),
        "plain_full": np.concatenate(plain_features, axis=0),
        "plain_selected_inference": np.concatenate(
            plain_selected_features, axis=0
        ),
        "selected_mask": np.concatenate(selected_masks, axis=0).astype(bool),
        "labels": np.concatenate(labels_store, axis=0),
        "cova_dynamic_prediction": np.concatenate(
            cova_prediction_store, axis=0
        ),
        "plain_selected_prediction": np.concatenate(
            plain_selected_prediction_store, axis=0
        ),
        "cases": [best_cases[key] for key in sorted(best_cases)],
        "selection_cases": [
            best_selection_cases[key] for key in sorted(best_selection_cases)
        ],
        "metrics": {
            "cova_dense": finish_metrics(cova_dense_store),
            "cova_dynamic": finish_metrics(cova_dynamic_store),
            "plain_dense": finish_metrics(plain_dense_store),
            "plain_selected_inference": finish_metrics(plain_selected_store),
        },
        "sample_count": int(processed),
    }
    return result


def _prediction_label(task_index: int, class_index: int) -> str:
    labels = TASK_LABEL_NAMES[task_index]
    if 0 <= int(class_index) < len(labels):
        return labels[int(class_index)]
    return str(class_index)


def render_missing_modality_cases(
    cases: Sequence[dict], output_path: Path, dpi: int
) -> None:
    if not cases:
        raise RuntimeError(
            "No test case was found where COVA improves over the plain baseline "
            "under a single visual-modality failure."
        )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required") from error

    figure, axes = plt.subplots(
        len(cases), 5, figsize=(16.5, 4.0 * len(cases)),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.55, 1.85]},
        squeeze=False,
    )
    stream_names = ("face", "body", "scene")
    for row_index, case in enumerate(cases):
        for column_index, stream_name in enumerate(stream_names):
            axis = axes[row_index, column_index]
            axis.imshow(case["images"][stream_name])
            is_missing = stream_name == case["missing_modality"]
            axis.set_title(
                stream_name.title() + (" (missing)" if is_missing else ""),
                color="#B22222" if is_missing else "black",
                fontweight="bold",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if is_missing:
                axis.plot([0, 1], [0, 1], transform=axis.transAxes,
                          color="#B22222", linewidth=4.0, alpha=0.85)
                axis.plot([0, 1], [1, 0], transform=axis.transAxes,
                          color="#B22222", linewidth=4.0, alpha=0.85)

        ground_truth = case["ground_truth"]
        for column_index, (method, prediction) in enumerate((
            ("Plain shared baseline", case["plain_prediction"]),
            ("COVA-MTL (known-missing re-routing)", case["cova_prediction"]),
        ), start=3):
            axis = axes[row_index, column_index]
            axis.axis("off")
            axis.set_title(method, fontweight="bold")
            y = 0.88
            if column_index == 4:
                selected_names = [
                    MODALITY_DISPLAY_NAMES[index]
                    for index, active in enumerate(case["cova_selected_mask"])
                    if active
                ]
                axis.text(
                    0.02, y, "Selected: " + "+".join(selected_names),
                    fontsize=9.5, fontweight="bold", va="top",
                )
                y -= 0.15
            for task_index, task_name in enumerate(TASK_DISPLAY_NAMES):
                correct = int(prediction[task_index]) == int(ground_truth[task_index])
                pred_name = _prediction_label(task_index, prediction[task_index])
                gt_name = _prediction_label(task_index, ground_truth[task_index])
                axis.text(
                    0.02, y,
                    f"{task_name}: {pred_name}\nGT: {gt_name}",
                    fontsize=9.0, va="top",
                    color="#16803A" if correct else "#B22222",
                    fontweight="bold" if not correct else "normal",
                )
                y -= 0.20
        axes[row_index, 0].set_ylabel(
            f"Sample {case['sample_index']}\nMissing {case['missing_modality'].title()}",
            fontsize=10, fontweight="bold",
        )

    figure.suptitle(
        "Paired missing-modality cases: plain baseline failures vs COVA-MTL recovery",
        fontsize=15, fontweight="bold", y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(figure, output_path, dpi)
    plt.close(figure)


def _valid_keypoints(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 3), dtype=np.float32)
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    if points.shape[1] >= 3:
        confident = points[:, 2] > 0.01
        valid = finite & confident
        if not valid.any():
            valid = finite & ((points[:, 0] != 0) | (points[:, 1] != 0))
    else:
        valid = finite
    return points[valid]


def _shared_pose_bounds(keypoint_assets: dict) -> Tuple[float, float, float, float]:
    groups = []
    for modality in ("posture", "gesture"):
        payload = keypoint_assets[modality]
        groups.append(_valid_keypoints(payload["representative"]))
        groups.extend(_valid_keypoints(points) for points in payload["frames"])
    groups = [group for group in groups if group.size]
    if not groups:
        return 0.0, 1.0, 0.0, 1.0
    combined = np.concatenate(groups, axis=0)
    xmin, xmax = float(combined[:, 0].min()), float(combined[:, 0].max())
    ymin, ymax = float(combined[:, 1].min()), float(combined[:, 1].max())
    if xmax - xmin < 1e-6:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymax - ymin < 1e-6:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    return xmin, xmax, ymin, ymax


# Halpe-136 connectivity.  The AIDE pose vector is ordered as
# 26 body + 68 face + 21 left-hand + 21 right-hand keypoints.  The renderer
# receives the body slice (0:26) or the concatenated hand slice (94:136), so
# the hand indices below are local to the 42-point Gesture array.
HALPE_POSTURE_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 18), (18, 19), (19, 11), (19, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),
)


def _halpe_hand_edges(offset: int) -> Tuple[Tuple[int, int], ...]:
    """Return the five four-segment fingers for one local 21-point hand."""
    return tuple(
        (offset + start, offset + end)
        for start, end in (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        )
    )


HALPE_GESTURE_EDGES = _halpe_hand_edges(0) + _halpe_hand_edges(21)


def _render_green_keypoints(
    points: np.ndarray,
    bounds: Tuple[float, float, float, float],
    canvas_size: int = 512,
    padding: int = 32,
    radius: int = 5,
    line_thickness: int = 2,
) -> np.ndarray:
    """Render Halpe body/hand skeletons in green on a black background.

    Limbs are drawn first with ``line_thickness=2`` and keypoints are then
    drawn with ``radius=5``.  Thus the line is substantially thinner than the
    10-pixel point diameter and cannot visually overpower the keypoints.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    raw = np.asarray(points, dtype=np.float32)
    valid = _valid_keypoints(raw)
    if not valid.size:
        return canvas

    finite = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1])
    if raw.shape[1] >= 3:
        valid_mask = finite & (raw[:, 2] > 0.01)
        if not valid_mask.any():
            valid_mask = finite & ((raw[:, 0] != 0) | (raw[:, 1] != 0))
    else:
        valid_mask = finite

    xmin, xmax, ymin, ymax = bounds
    usable = float(canvas_size - 2 * padding)
    scale = min(usable / (xmax - xmin), usable / (ymax - ymin))
    drawn_width = (xmax - xmin) * scale
    drawn_height = (ymax - ymin) * scale
    offset_x = (canvas_size - drawn_width) / 2.0
    offset_y = (canvas_size - drawn_height) / 2.0
    x_pixels = np.rint(offset_x + (valid[:, 0] - xmin) * scale).astype(int)
    y_pixels = np.rint(offset_y + (valid[:, 1] - ymin) * scale).astype(int)

    valid_indices = np.flatnonzero(valid_mask)
    pixel_by_index = {
        int(point_index): (
            int(np.clip(x_pixel, 0, canvas_size - 1)),
            int(np.clip(y_pixel, 0, canvas_size - 1)),
        )
        for point_index, x_pixel, y_pixel in zip(
            valid_indices, x_pixels, y_pixels
        )
    }

    if raw.shape[0] == 26:
        edges = HALPE_POSTURE_EDGES
    elif raw.shape[0] == 42:
        edges = HALPE_GESTURE_EDGES
    else:
        edges = ()

    # Lines are deliberately rendered before the dots and kept thinner than
    # their diameter.  cv2 is already required by the qualitative data loader.
    for start_index, end_index in edges:
        if start_index not in pixel_by_index or end_index not in pixel_by_index:
            continue
        if cv2 is not None:
            cv2.line(
                canvas,
                pixel_by_index[start_index],
                pixel_by_index[end_index],
                color=(0.0, 0.72, 0.0),
                thickness=max(1, int(line_thickness)),
                lineType=cv2.LINE_AA,
            )

    yy, xx = np.ogrid[:canvas_size, :canvas_size]
    for x_pixel, y_pixel in zip(x_pixels, y_pixels):
        x_pixel = int(np.clip(x_pixel, 0, canvas_size - 1))
        y_pixel = int(np.clip(y_pixel, 0, canvas_size - 1))
        circle = (xx - x_pixel) ** 2 + (yy - y_pixel) ** 2 <= radius ** 2
        canvas[circle, 1] = 1.0
    return canvas


def export_missing_case_ppt_assets(
    cases: Sequence[dict],
    output_dir: Path,
    manifest_filename: str = "missing_case_ppt_manifest_seed42.csv",
) -> dict:
    """Save clean, uncomposed case images and tables for manual PPT layout.

    The model's Scene tensor is a 2x2 mosaic ordered as Front, Left, Right,
    In-cabin.  Scene assets are split back into these four views and the mosaic
    itself is intentionally not written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    case_directories = []
    for case in cases:
        if case.get("case_type") == "normal_selection":
            case_dir = output_dir / (
                f"cardinality_{case['cardinality']}_"
                f"sample_{case['sample_index']:04d}"
            )
        else:
            case_dir = output_dir / (
                f"missing_{case['missing_modality']}_"
                f"sample_{case['sample_index']:04d}"
            )
        case_dir.mkdir(parents=True, exist_ok=True)
        case_directories.append(str(case_dir))
        for stream_name in ("face", "body"):
            for frame_index, image in enumerate(case["frames"][stream_name], 1):
                plt.imsave(
                    case_dir / f"{stream_name}_frame_{frame_index:02d}.png",
                    np.clip(image, 0.0, 1.0),
                )
            plt.imsave(
                case_dir / f"{stream_name}_representative.png",
                np.clip(case["images"][stream_name], 0.0, 1.0),
            )

        scene_view_names = ("front", "left", "right", "incar")
        for frame_index, mosaic in enumerate(case["frames"]["scene"], 1):
            height, width = mosaic.shape[:2]
            half_height, half_width = height // 2, width // 2
            scene_views = (
                mosaic[:half_height, :half_width],
                mosaic[:half_height, half_width:],
                mosaic[half_height:, :half_width],
                mosaic[half_height:, half_width:],
            )
            for view_name, view_image in zip(scene_view_names, scene_views):
                plt.imsave(
                    case_dir / f"scene_{view_name}_frame_{frame_index:02d}.png",
                    np.clip(view_image, 0.0, 1.0),
                )
        representative = case["images"]["scene"]
        height, width = representative.shape[:2]
        half_height, half_width = height // 2, width // 2
        representative_views = (
            representative[:half_height, :half_width],
            representative[:half_height, half_width:],
            representative[half_height:, :half_width],
            representative[half_height:, half_width:],
        )
        for view_name, view_image in zip(scene_view_names, representative_views):
            plt.imsave(
                case_dir / f"scene_{view_name}_representative.png",
                np.clip(view_image, 0.0, 1.0),
            )

        # Posture and Gesture use the same sample-specific coordinate bounds,
        # producing directly comparable black-background/green-point assets.
        pose_bounds = _shared_pose_bounds(case["keypoint_assets"])
        for modality in ("posture", "gesture"):
            payload = case["keypoint_assets"][modality]
            plt.imsave(
                case_dir / f"{modality}_representative.png",
                _render_green_keypoints(payload["representative"], pose_bounds),
            )
            for frame_number, points in enumerate(payload["frames"], 1):
                plt.imsave(
                    case_dir / f"{modality}_frame_{frame_number:02d}.png",
                    _render_green_keypoints(points, pose_bounds),
                )

        selected_names = [
            MODALITY_DISPLAY_NAMES[index]
            for index, active in enumerate(case["cova_selected_mask"])
            if active
        ]
        for task_index, task_name in enumerate(TASK_DISPLAY_NAMES):
            ground_truth = int(case["ground_truth"][task_index])
            plain_prediction = int(case["plain_prediction"][task_index])
            cova_prediction = int(case["cova_prediction"][task_index])
            manifest_rows.append({
                "case_directory": str(case_dir),
                "sample_index": int(case["sample_index"]),
                "source_path": case.get("source_path"),
                "case_type": case.get("case_type", "known_missing"),
                "cardinality": int(sum(case["cova_selected_mask"])),
                "missing_modality": case.get("missing_modality"),
                "cova_selected_modalities": "+".join(selected_names),
                "task": task_name,
                "ground_truth": _prediction_label(task_index, ground_truth),
                "plain_prediction": _prediction_label(
                    task_index, plain_prediction
                ),
                "plain_correct": plain_prediction == ground_truth,
                "cova_prediction": _prediction_label(task_index, cova_prediction),
                "cova_correct": cova_prediction == ground_truth,
            })
    manifest_path = output_dir / manifest_filename
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return {
        "asset_root": str(output_dir),
        "manifest_csv": str(manifest_path),
        "case_directories": case_directories,
    }


def _draw_prediction_card(axis, title: str, prediction, ground_truth, color: str):
    from matplotlib.patches import FancyBboxPatch

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    card = FancyBboxPatch(
        (0.025, 0.025), 0.95, 0.95,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=2.2, edgecolor=color, facecolor="#FAFAFA",
        transform=axis.transAxes,
    )
    axis.add_patch(card)
    axis.text(
        0.5, 0.925, title, ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=color,
    )
    axis.plot(
        [0.08, 0.92], [0.865, 0.865], color="#D8D8D8",
        linewidth=1.0, transform=axis.transAxes,
    )
    y_positions = (0.75, 0.56, 0.37, 0.18)
    for task_index, (task_name, y_value) in enumerate(
        zip(TASK_DISPLAY_NAMES, y_positions)
    ):
        target = int(ground_truth[task_index])
        pred = int(prediction[task_index])
        correct = pred == target
        status_color = "#198754" if correct else "#C62828"
        status_text = "CORRECT" if correct else "WRONG"
        axis.text(
            0.08, y_value + 0.045, task_name,
            fontsize=9.4, fontweight="bold", color="#333333",
            transform=axis.transAxes,
        )
        axis.text(
            0.92, y_value + 0.045, status_text,
            fontsize=8.2, fontweight="bold", color=status_color,
            ha="right", transform=axis.transAxes,
        )
        axis.text(
            0.08, y_value - 0.015,
            f"GT: {_prediction_label(task_index, target)}",
            fontsize=8.2, color="#666666", transform=axis.transAxes,
        )
        axis.text(
            0.08, y_value - 0.070,
            f"Pred: {_prediction_label(task_index, pred)}",
            fontsize=8.7, fontweight="bold", color=status_color,
            transform=axis.transAxes,
        )


def render_selection_recovery_figure(
    cases: Sequence[dict], output_path: Path, dpi: int, title: Optional[str] = None
) -> None:
    """Render publication-ready modality selection and prediction comparisons."""
    if not cases:
        raise RuntimeError("No normal selection recovery cases were found.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("matplotlib is required") from error

    ordered_cases = sorted(cases, key=lambda case: int(case["cardinality"]))
    figure, axes = plt.subplots(
        len(ordered_cases), 7,
        figsize=(20.5, 3.65 * len(ordered_cases)),
        gridspec_kw={
            "width_ratios": [1.05, 1.05, 1.05, 1.05, 1.05, 2.35, 2.35],
            "wspace": 0.16, "hspace": 0.36,
        },
        squeeze=False,
    )
    modality_titles = ("Face", "Body", "Scene", "Posture", "Gesture")
    for row_index, case in enumerate(ordered_cases):
        pose_bounds = _shared_pose_bounds(case["keypoint_assets"])
        modality_images = (
            case["images"]["face"],
            case["images"]["body"],
            case["images"]["scene"],
            _render_green_keypoints(
                case["keypoint_assets"]["posture"]["representative"],
                pose_bounds,
            ),
            _render_green_keypoints(
                case["keypoint_assets"]["gesture"]["representative"],
                pose_bounds,
            ),
        )
        selected_mask = np.asarray(case["cova_selected_mask"], dtype=bool)
        selected_names = [
            MODALITY_DISPLAY_NAMES[index]
            for index, active in enumerate(selected_mask) if active
        ]
        for modality_index, (modality_title, image) in enumerate(
            zip(modality_titles, modality_images)
        ):
            axis = axes[row_index, modality_index]
            axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_xticks([])
            axis.set_yticks([])
            active = bool(selected_mask[modality_index])
            border_color = "#198754" if active else "#C62828"
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.4)
                spine.set_edgecolor(border_color)
            axis.set_title(
                modality_title,
                fontsize=10.5, fontweight="bold",
                color="#1E1E1E" if active else "#777777",
                pad=7,
            )
            if not active:
                axis.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=axis.transAxes,
                    facecolor="black", alpha=0.38, zorder=3,
                ))
                axis.plot(
                    [0.10, 0.90], [0.10, 0.90], transform=axis.transAxes,
                    color="#E53935", linewidth=5.0, zorder=4,
                    solid_capstyle="round",
                )
                axis.plot(
                    [0.10, 0.90], [0.90, 0.10], transform=axis.transAxes,
                    color="#E53935", linewidth=5.0, zorder=4,
                    solid_capstyle="round",
                )
            else:
                axis.text(
                    0.06, 0.94, "SELECTED", transform=axis.transAxes,
                    ha="left", va="top", fontsize=7.2, fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.22", facecolor="#198754",
                        edgecolor="none", alpha=0.94,
                    ),
                )
        axes[row_index, 0].text(
            0.0, 1.24,
            f"Example {chr(ord('A') + row_index)}  |  "
            f"{case['cardinality']}/5 selected: {' + '.join(selected_names)}",
            transform=axes[row_index, 0].transAxes,
            fontsize=11.5, fontweight="bold", color="#222222",
            ha="left", va="bottom", clip_on=False,
        )
        _draw_prediction_card(
            axes[row_index, 5], "Plain shared baseline",
            case["plain_prediction"], case["ground_truth"], "#C62828",
        )
        _draw_prediction_card(
            axes[row_index, 6], "COVA-MTL",
            case["cova_prediction"], case["ground_truth"], "#198754",
        )

    figure.suptitle(
        title or "Sample-wise modality selection and prediction recovery",
        fontsize=16, fontweight="bold", y=0.995,
    )
    figure.text(
        0.5, 0.008,
        "Red crosses denote modalities excluded before the recognition encoder. "
        "Both methods receive the identical selected subset.",
        ha="center", fontsize=9.2, color="#555555",
    )
    figure.subplots_adjust(
        left=0.015, right=0.99, top=0.94, bottom=0.05,
        wspace=0.16, hspace=0.42,
    )
    _save_figure(figure, output_path, dpi)
    plt.close(figure)


def choose_dominant_exact_cardinality_route(
    selected_masks: np.ndarray, target_cardinality: int
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Choose one real, most-frequent route with exactly K active modalities.

    Combining different K-modality masks would make their union display more
    than K modality classes.  Therefore the t-SNE cohort is restricted to clips
    sharing one identical mask.  Ties are resolved deterministically by the
    binary mask ordering returned by ``np.unique``.
    """
    selected_masks = np.asarray(selected_masks, dtype=bool)
    if selected_masks.ndim != 2 or selected_masks.shape[1] != len(MODALITY_NAMES):
        raise ValueError(
            f"Expected selected masks [N,{len(MODALITY_NAMES)}], "
            f"got {selected_masks.shape}"
        )
    unique_masks, inverse, counts = np.unique(
        selected_masks, axis=0, return_inverse=True, return_counts=True
    )
    rows = []
    for route_id, (mask, count) in enumerate(zip(unique_masks, counts)):
        names = [
            MODALITY_DISPLAY_NAMES[index]
            for index, active in enumerate(mask)
            if active
        ]
        rows.append({
            "route_id": int(route_id),
            "mask_bits": "".join("1" if active else "0" for active in mask),
            "modalities": "+".join(names),
            "cardinality": int(mask.sum()),
            "sample_count": int(count),
            "sample_fraction": float(count / selected_masks.shape[0]),
        })
    histogram = pd.DataFrame(rows).sort_values(
        ["cardinality", "sample_count", "mask_bits"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    candidates = np.flatnonzero(unique_masks.sum(1) == int(target_cardinality))
    if candidates.size == 0:
        available = histogram.groupby("cardinality")["sample_count"].sum().to_dict()
        raise RuntimeError(
            f"No seed-42 test sample selected exactly {target_cardinality} "
            f"modalities. Cardinality counts: {available}"
        )
    best_route_id = int(candidates[np.argmax(counts[candidates])])
    cohort_indices = np.flatnonzero(inverse == best_route_id)
    if cohort_indices.size < 2:
        names = "+".join(
            MODALITY_DISPLAY_NAMES[index]
            for index, active in enumerate(unique_masks[best_route_id])
            if active
        )
        raise RuntimeError(
            f"The most frequent exact-{target_cardinality} route ({names}) has "
            f"only {cohort_indices.size} sample; at least 2 are required for t-SNE."
        )
    return cohort_indices, unique_masks[best_route_id], histogram


def run_plain_baseline_qualitative_pipeline(args, device: torch.device) -> dict:
    """Train the plain baseline once, then generate both requested figures."""
    seed_everything(42)
    cova_run_dir = resolve_cova_seed42_run(args.cova_run_dir)
    cova_args = representation_args_from_saved_run(args, cova_run_dir, True)
    # The baseline must use the exact data and split configuration stored with
    # the seed-42 COVA checkpoint.
    for key in (
        "data_root", "split_dir", "split_seed", "t_target", "stride",
        "face_pad_top", "face_pad_bottom", "face_pad_left", "face_pad_right",
        "embed", "batch_size", "num_workers",
    ):
        setattr(args, key, getattr(cova_args, key))

    output_dir = (
        args.qualitative_output_dir.expanduser().resolve()
        if args.qualitative_output_dir is not None
        else args.plain_run_dir / "qualitative_selection_recovery_seed42"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(cova_args, cache_order=True)
    try:
        train_plain_baseline_once(args, loaders, device)
        plain_model = load_plain_baseline(args, device)
        cova_model, router = load_representation_variant(cova_args, device)
        result = collect_plain_cova_qualitative_data(
            args, cova_model, router, plain_model,
            loaders["test"], device,
        )
    finally:
        shutdown_loaders(loaders)

    cardinality_summaries = {}
    route_histogram = None
    generated_cardinalities = []
    for target_cardinality in args.target_selected_cardinalities:
        cardinality_dir = output_dir / f"cardinality_{target_cardinality}"
        try:
            cohort_indices, selected_pattern, current_histogram = (
                choose_dominant_exact_cardinality_route(
                    result["selected_mask"], target_cardinality
                )
            )
        except RuntimeError as error:
            cardinality_summaries[str(target_cardinality)] = {
                "status": "skipped",
                "reason": str(error),
            }
            print(f"WARNING: cardinality {target_cardinality} skipped: {error}")
            continue
        if route_histogram is None:
            route_histogram = current_histogram
        cardinality_dir.mkdir(parents=True, exist_ok=True)
        generated_cardinalities.append(int(target_cardinality))
        selected_names = [
            MODALITY_DISPLAY_NAMES[index]
            for index, active in enumerate(selected_pattern)
            if active
        ]
        selected_pattern_name = "+".join(selected_names)
        print(
            f"Selected exact-{target_cardinality} route for t-SNE: "
            f"{selected_pattern_name} ({len(cohort_indices)} test clips)"
        )

        # Each cardinality uses one identical route and one shared clip cohort.
        # Thus panels (b)/(d) show exactly K modality classes, while panels
        # (a)/(c) use all five modalities on those same clips.
        cohort_selected_mask = result["selected_mask"][cohort_indices]
        full_mask = np.ones_like(cohort_selected_mask, dtype=bool)
        panels = (
            (
                "(a) COVA-MTL: all modalities",
                result["cova_full"][cohort_indices], full_mask,
            ),
            (
                f"(b) COVA-MTL: selected {selected_pattern_name}",
                result["cova_selected"][cohort_indices], cohort_selected_mask,
            ),
            (
                "(c) Plain shared baseline: all modalities",
                result["plain_full"][cohort_indices], full_mask,
            ),
            (
                f"(d) Plain partial inference: selected {selected_pattern_name}",
                result["plain_selected_inference"][cohort_indices],
                cohort_selected_mask,
            ),
        )
        metric_rows = []
        for title, features, masks in panels:
            metric_rows.extend(
                masked_modality_geometry_rows(title, features, masks)
            )
        metrics_path = (
            cardinality_dir / "four_panel_modality_geometry_seed42.csv"
        )
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

        distribution_path = (
            cardinality_dir / "four_panel_modality_tsne_seed42.png"
        )
        individual_panel_dir = cardinality_dir / "individual_tsne_panels"
        coordinates_path = (
            cardinality_dir / "four_panel_tsne_coordinates_seed42.csv"
        )
        tsne_metadata = render_four_panel_modality_tsne(
            panels, distribution_path, args,
            individual_dir=individual_panel_dir,
            coordinates_path=coordinates_path,
        )

        arrays_path = cardinality_dir / "plain_cova_modality_features_seed42.npz"
        np.savez_compressed(
            arrays_path,
            cohort_indices=cohort_indices.astype(np.int64),
            selected_pattern=selected_pattern.astype(np.uint8),
            selected_mask=cohort_selected_mask.astype(np.uint8),
            cova_full=result["cova_full"][cohort_indices].astype(np.float32),
            cova_selected=result["cova_selected"][cohort_indices].astype(
                np.float32
            ),
            plain_full=result["plain_full"][cohort_indices].astype(np.float32),
            plain_selected_inference=result["plain_selected_inference"][
                cohort_indices
            ].astype(np.float32),
            labels=result["labels"][cohort_indices].astype(np.int64),
            cova_dynamic_prediction=result["cova_dynamic_prediction"][
                cohort_indices
            ].astype(np.int64),
            plain_selected_prediction=result["plain_selected_prediction"][
                cohort_indices
            ].astype(np.int64),
            modality_names=np.asarray(MODALITY_NAMES),
        )
        cardinality_summaries[str(target_cardinality)] = {
            "status": "generated",
            "selected_pattern": selected_pattern.astype(int).tolist(),
            "selected_modalities": selected_names,
            "sample_count": int(len(cohort_indices)),
            "sample_indices": cohort_indices.astype(int).tolist(),
            "tsne": tsne_metadata,
            "metric_rows": metric_rows,
            "artifacts": {
                "distribution_png": str(distribution_path),
                "distribution_pdf": str(distribution_path.with_suffix(".pdf")),
                "individual_panel_dir": str(individual_panel_dir),
                "tsne_coordinates_csv": str(coordinates_path),
                "geometry_csv": str(metrics_path),
                "feature_arrays": str(arrays_path),
            },
        }

    if not generated_cardinalities:
        raise RuntimeError(
            "No requested cardinality produced a valid repeated routing pattern. "
            "See the warnings above; no t-SNE figure was fabricated."
        )
    route_histogram_path = output_dir / "routing_pattern_histogram_seed42.csv"
    route_histogram.to_csv(route_histogram_path, index=False)

    # Save a complete normal-routing audit so every selected example can be
    # traced to its mask, ground truth, and both predictions.
    audit_rows = []
    for sample_index in range(result["sample_count"]):
        mask = result["selected_mask"][sample_index]
        selected_names = [
            MODALITY_DISPLAY_NAMES[index]
            for index, active in enumerate(mask) if active
        ]
        row = {
            "sample_index": int(sample_index),
            "cardinality": int(mask.sum()),
            "selected_modalities": "+".join(selected_names),
        }
        plain_wrong = 0
        cova_wrong = 0
        for task_index, task_name in enumerate(TASK_DISPLAY_NAMES):
            target = int(result["labels"][sample_index, task_index])
            plain_pred = int(
                result["plain_selected_prediction"][sample_index, task_index]
            )
            cova_pred = int(
                result["cova_dynamic_prediction"][sample_index, task_index]
            )
            plain_wrong += int(plain_pred != target)
            cova_wrong += int(cova_pred != target)
            row[f"{task_name}_ground_truth"] = _prediction_label(
                task_index, target
            )
            row[f"{task_name}_plain_prediction"] = _prediction_label(
                task_index, plain_pred
            )
            row[f"{task_name}_plain_correct"] = plain_pred == target
            row[f"{task_name}_cova_prediction"] = _prediction_label(
                task_index, cova_pred
            )
            row[f"{task_name}_cova_correct"] = cova_pred == target
        row["plain_wrong_task_count"] = int(plain_wrong)
        row["cova_wrong_task_count"] = int(cova_wrong)
        row["qualifies_plain_wrong_cova_all_correct"] = (
            plain_wrong > 0 and cova_wrong == 0
        )
        audit_rows.append(row)
    prediction_audit_path = output_dir / "normal_selection_prediction_audit_seed42.csv"
    pd.DataFrame(audit_rows).to_csv(prediction_audit_path, index=False)

    selection_case_payload = []
    for case in result["selection_cases"]:
        selection_case_payload.append({
            key: value for key, value in case.items()
            if key not in {
                "images", "frames", "keypoint_assets", "priority"
            }
        })
    selection_metadata_path = (
        output_dir / "selection_recovery_case_metadata_seed42.json"
    )
    found_selection_cardinalities = sorted(
        int(case["cardinality"]) for case in result["selection_cases"]
    )
    missing_selection_cardinalities = [
        int(value) for value in args.target_selected_cardinalities
        if int(value) not in found_selection_cardinalities
    ]
    write_json(selection_metadata_path, {
        "strict_selection_rule": (
            "For each requested exact cardinality, select the real test clip "
            "with the most Plain partial-inference task errors among clips where "
            "COVA-MTL predicts all four tasks correctly under the identical mask."
        ),
        "found_cardinalities": found_selection_cardinalities,
        "missing_cardinalities": missing_selection_cardinalities,
        "cases": selection_case_payload,
    })
    selection_case_artifacts = {}
    if result["selection_cases"]:
        selection_asset_info = export_missing_case_ppt_assets(
            result["selection_cases"],
            output_dir / "selection_recovery_ppt_assets",
            manifest_filename="selection_recovery_manifest_seed42.csv",
        )
        combined_selection_path = (
            output_dir / "selection_recovery_cardinality234_seed42.png"
        )
        render_selection_recovery_figure(
            result["selection_cases"], combined_selection_path,
            args.analysis_dpi,
        )
        individual_case_dir = output_dir / "individual_selection_case_figures"
        individual_case_dir.mkdir(parents=True, exist_ok=True)
        individual_case_paths = []
        for case in result["selection_cases"]:
            path = individual_case_dir / (
                f"cardinality_{case['cardinality']}_"
                f"sample_{case['sample_index']:04d}.png"
            )
            render_selection_recovery_figure(
                [case], path, args.analysis_dpi,
                title=(
                    f"COVA-MTL selection recovery "
                    f"({case['cardinality']}/5 modalities)"
                ),
            )
            individual_case_paths.append(str(path))
        selection_case_artifacts = {
            **selection_asset_info,
            "combined_figure_png": str(combined_selection_path),
            "combined_figure_pdf": str(
                combined_selection_path.with_suffix(".pdf")
            ),
            "individual_figure_dir": str(individual_case_dir),
            "individual_figure_pngs": individual_case_paths,
        }
    if missing_selection_cardinalities:
        print(
            "WARNING: no strict Plain-wrong/COVA-all-correct example exists "
            "for cardinalities " + str(missing_selection_cardinalities) + ". "
            "No example was fabricated for those cardinalities."
        )

    case_payload = []
    for case in result["cases"]:
        case_payload.append({
            key: value for key, value in case.items()
            if key not in {"images", "frames", "keypoint_assets", "priority"}
        })
    cases_json_path = output_dir / "missing_modality_case_metadata_seed42.json"
    write_json(cases_json_path, {
        "selection_rule": (
            "For each missing visual modality, select the test sample with the "
            "largest number of plain-baseline errors among cases where the plain "
            "baseline is wrong on at least one task and COVA-MTL predicts all four "
            "tasks correctly."
        ),
        "cases": case_payload,
    })
    case_artifacts = {}
    if result["cases"]:
        case_artifacts = export_missing_case_ppt_assets(
            result["cases"], output_dir / "missing_case_ppt_assets"
        )
    else:
        print(
            "WARNING: no honest paired case satisfied plain_wrong > 0 and "
            "cova_wrong == 0; "
            "the case-study figure was not fabricated."
        )

    summary = {
        "experiment": "COVA-MTL vs plain shared baseline qualitative comparison",
        "seed": 42,
        "plain_training_runs_required": 1,
        "plain_checkpoint_reused_if_present": True,
        "cova_checkpoint_retrained": False,
        "cova_run_dir": str(cova_run_dir),
        "plain_run_dir": str(args.plain_run_dir),
        "sample_count": int(result["sample_count"]),
        "requested_cardinalities": [
            int(value) for value in args.target_selected_cardinalities
        ],
        "generated_cardinalities": generated_cardinalities,
        "cardinality_results": cardinality_summaries,
        "baseline_definition": {
            "visual": "one shared standard ResNet-18 for face/body/scene",
            "vector": "one shared MLP for padded posture/gesture",
            "temporal": "mean",
            "fusion": "masked mean",
            "training": "dense recognition only",
        },
        "panel_protocol": (
            "All panels use only clips assigned to the most frequent identical "
            "exact-cardinality routing pattern. Panels (b) and (d) therefore "
            "contain exactly the same target-cardinality modality classes rather "
            "than the five-class union of multiple masks. Panel (d) is produced "
            "by a fresh partial-modality forward pass through the dense-trained "
            "plain checkpoint; inactive branches are not executed or plotted."
        ),
        "case_asset_protocol": {
            "scene": (
                "The 2x2 model-input mosaic is split and saved separately as "
                "Front, Left, Right, and In-cabin; no scene mosaic PNG is saved."
            ),
            "posture": (
                "26-point Halpe body skeleton on black; green radius-5 dots "
                "with thinner 2-pixel connection lines"
            ),
            "gesture": (
                "42-point Halpe two-hand skeleton on black; green radius-5 "
                "dots with thinner 2-pixel connection lines"
            ),
            "pose_coordinates": (
                "Posture and Gesture from the same sample share one coordinate "
                "range and aspect-preserving scale across representative and "
                "four temporal frames."
            ),
        },
        "selection_recovery_protocol": {
            "plain_input": (
                "Dense-trained Plain checkpoint with only the COVA-selected "
                "branches actually executed"
            ),
            "cova_input": "COVA-MTL under the identical selected mask",
            "strict_condition": (
                "Plain is wrong on at least one task and COVA-MTL is correct "
                "on all four tasks"
            ),
            "found_cardinalities": found_selection_cardinalities,
            "missing_cardinalities": missing_selection_cardinalities,
        },
        "metrics": result["metrics"],
        "artifacts": {
            "routing_pattern_histogram_csv": str(route_histogram_path),
            "normal_selection_prediction_audit_csv": str(
                prediction_audit_path
            ),
            "selection_recovery_metadata": str(selection_metadata_path),
            "selection_recovery": selection_case_artifacts,
            "case_metadata": str(cases_json_path),
            **case_artifacts,
        },
        "interpretation_limit": (
            "This is a single-seed end-to-end qualitative baseline. It does not "
            "isolate one component and t-SNE cluster gaps are not quantitative "
            "performance evidence. Use the original-space geometry CSV and task "
            "metrics as companion evidence."
        ),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = output_dir / "plain_cova_qualitative_summary_seed42.json"
    write_json(summary_path, summary)
    print("\nPlain baseline training/evaluation and qualitative export completed.")
    print(f"cardinality output root: {output_dir}")
    print(f"generated cardinalities: {generated_cardinalities}")
    print(
        "strict selection recovery cardinalities: "
        f"{found_selection_cardinalities}"
    )
    print(
        "selection recovery figure: "
        f"{selection_case_artifacts.get('combined_figure_png', 'not generated')}"
    )
    print(
        "missing-case PPT assets: "
        f"{case_artifacts.get('asset_root', 'not generated')}"
    )
    print(f"summary: {summary_path}")
    del cova_model, router, plain_model
    release_seed_resources()
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_float_list(text: str, expected: int, name: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"{name} requires {expected} comma-separated values")
    return values


def parse_int_list(text: str, name: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must contain comma-separated integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicate seeds")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "COVA-MTL on AIDE. The default command performs the seed-42 "
            "plain-baseline training (once) and qualitative comparison."
        )
    )
    parser.add_argument(
        "--stage",
        choices=(
            "teacher", "cache", "router", "calibrate", "sparse",
            "recalibrate", "eval", "all", "aggregate", "smoke",
            "representation_analysis", "modality_tsne", "plain_qualitative",
        ),
        default="plain_qualitative",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/AIDE_Dataset"))
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("RUNS/AIDE_COVA_MTL_v6_3_MultiSeed5"),
    )
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument(
        "--multi-seed", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "run seeds sequentially in seed_<N> subdirectories; enabled by default. "
            "Use --no-multi-seed for the original single-seed behavior"
        ),
    )
    parser.add_argument(
        "--seeds", type=str, default="42,43,44,45,46",
        help="comma-separated model seeds used when --multi-seed is enabled",
    )
    parser.add_argument("--seed", type=int, default=42, help="single model seed")
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="fixed data-split seed shared by every model seed",
    )
    parser.add_argument(
        "--continue-on-seed-error", action=argparse.BooleanOptionalAction, default=False,
        help="record a failed seed and continue with later seeds instead of stopping",
    )
    parser.add_argument(
        "--require-cuda", action=argparse.BooleanOptionalAction, default=True,
        help="stop training instead of silently running on CPU; smoke stage is exempt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--router-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--t-target", type=int, default=16)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--face-pad-top", type=int, default=200)
    parser.add_argument("--face-pad-bottom", type=int, default=20)
    parser.add_argument("--face-pad-left", type=int, default=100)
    parser.add_argument("--face-pad-right", type=int, default=50)

    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--preview-frames", type=int, default=4)
    parser.add_argument("--preview-size", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--scmc", dest="use_scmc",
        action=argparse.BooleanOptionalAction, default=True,
        help="enable SCMC; use --no-scmc for the direct mask-aware TSMF ablation",
    )
    parser.add_argument("--scmc-hidden", type=int, default=128)
    parser.add_argument("--scmc-strength", type=float, default=0.25)
    parser.add_argument("--train-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-ckpt", type=Path, default=None)

    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--sparse-epochs", type=int, default=20)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--sparse-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--subset-loss-weight", type=float, default=0.5)
    parser.add_argument("--subset-samples-per-batch", type=int, default=2)
    parser.add_argument("--scout-alignment-weight", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-confidence", type=float, default=0.6)
    parser.add_argument("--failure-dropout", type=float, default=0.2)
    parser.add_argument(
        "--route-exploration-probability", type=float, default=0.25,
        help="probability of training on the next self-referenced admissible subset",
    )
    parser.add_argument("--router-hidden", type=int, default=256)
    parser.add_argument("--router-listwise-weight", type=float, default=1.0)
    parser.add_argument("--router-smooth-weight", type=float, default=0.5)
    parser.add_argument("--router-sufficiency-weight", type=float, default=1.0)
    parser.add_argument("--router-temperature", type=float, default=0.1)
    parser.add_argument("--router-availability-dropout", type=float, default=0.15)
    parser.add_argument("--sparse-availability-dropout", type=float, default=0.15)
    parser.add_argument("--calibration-coverage", type=float, default=0.90)
    parser.add_argument(
        "--require-calibration", action=argparse.BooleanOptionalAction, default=True,
        help="require a validation-derived regret calibration before routing",
    )

    parser.add_argument(
        "--modality-costs", type=str, default="1.0,1.0,1.0,0.05,0.05",
        help="reporting only; never used by the ACSR selection rule",
    )
    parser.add_argument("--latency-warmup-batches", type=int, default=3)
    parser.add_argument(
        "--eval-failures", action=argparse.BooleanOptionalAction, default=True,
        help="also evaluate each single-stream failure during the final eval stage",
    )
    parser.add_argument(
        "--eval-all-subsets", action=argparse.BooleanOptionalAction, default=True,
        help="evaluate all 31 non-empty fixed modality subsets",
    )
    parser.add_argument(
        "--full-run-dir", type=Path, default=None,
        help=(
            "completed seed-42 Full internal-ablation run; auto-detected by default"
        ),
    )
    parser.add_argument(
        "--no-calibration-run-dir", type=Path, default=None,
        help=(
            "completed seed-42 no_subset_calibration run; auto-detected by default"
        ),
    )
    parser.add_argument(
        "--analysis-output-dir", type=Path, default=None,
        help=(
            "representation-analysis output directory; default: "
            "<internal-ablation-root>/configuration_representation_seed42"
        ),
    )
    parser.add_argument(
        "--modality-output-dir", type=Path, default=None,
        help=(
            "modality-wise t-SNE output directory; default: "
            "<internal-ablation-root>/modalitywise_tsne_seed42"
        ),
    )
    parser.add_argument(
        "--cova-run-dir", type=Path, default=None,
        help="completed Full COVA-MTL seed-42 directory; auto-detected by default",
    )
    parser.add_argument(
        "--plain-run-dir", type=Path,
        default=Path(
            ""
            "RUNS/AIDE_COVA_MTL_PlainShared_seed42"
        ),
        help="checkpoint and result directory for the one-time plain baseline",
    )
    parser.add_argument(
        "--qualitative-output-dir", type=Path, default=None,
        help="output directory for the 2x2 t-SNE and missing-modality cases",
    )
    parser.add_argument("--plain-epochs", type=int, default=50)
    parser.add_argument("--plain-lr", type=float, default=1e-4)
    parser.add_argument(
        "--force-retrain-plain", action=argparse.BooleanOptionalAction,
        default=False,
        help="discard checkpoint reuse and train the plain baseline again",
    )
    parser.add_argument(
        "--analysis-max-samples", type=int, default=0,
        help="maximum test samples; 0 evaluates the complete test split",
    )
    parser.add_argument(
        "--tsne-samples", type=int, default=180,
        help="maximum paired samples displayed in each t-SNE panel",
    )
    parser.add_argument(
        "--target-selected-cardinalities", type=str, default="2,3,4",
        help=(
            "comma-separated exact routing cardinalities exported independently "
            "using their most frequent identical pattern (default: 2,3,4)"
        ),
    )
    parser.add_argument(
        "--tsne-perplexity", type=float, default=30.0,
        help="requested t-SNE perplexity (automatically capped below point count)",
    )
    parser.add_argument(
        "--tsne-iterations", type=int, default=1000,
        help="optimization iterations for every paired t-SNE embedding",
    )
    parser.add_argument(
        "--tsne-seed", type=int, default=42,
        help="fixed random state used by every t-SNE panel",
    )
    parser.add_argument(
        "--analysis-dpi", type=int, default=300,
        help="PNG resolution for representation-analysis figures",
    )
    parser.add_argument(
        "--save-feature-arrays", action=argparse.BooleanOptionalAction, default=True,
        help="save compressed task features, logits, masks, and labels as NPZ",
    )
    return parser


def configure_run_paths(args):
    """Derive every checkpoint/cache path from the active run directory."""
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
    if args.full_run_dir is not None:
        args.full_run_dir = args.full_run_dir.expanduser().resolve()
    if args.no_calibration_run_dir is not None:
        args.no_calibration_run_dir = args.no_calibration_run_dir.expanduser().resolve()
    if args.analysis_output_dir is not None:
        args.analysis_output_dir = args.analysis_output_dir.expanduser().resolve()
    if args.modality_output_dir is not None:
        args.modality_output_dir = args.modality_output_dir.expanduser().resolve()
    if args.cova_run_dir is not None:
        args.cova_run_dir = args.cova_run_dir.expanduser().resolve()
    args.plain_run_dir = args.plain_run_dir.expanduser().resolve()
    if args.qualitative_output_dir is not None:
        args.qualitative_output_dir = args.qualitative_output_dir.expanduser().resolve()
    if args.split_dir is None:
        args.split_dir = (
            args.data_root / "RUNS" / "split_lists"
            / f"splits_seed{args.split_seed}_8-2_full"
        )
    args.split_dir = args.split_dir.expanduser().resolve()
    args.seeds = parse_int_list(args.seeds, "seeds")
    args.target_selected_cardinalities = parse_int_list(
        args.target_selected_cardinalities, "target-selected-cardinalities"
    )
    args.modality_costs = parse_float_list(args.modality_costs, 5, "modality-costs")
    if any(cost <= 0 for cost in args.modality_costs):
        raise ValueError("All modality costs must be positive.")
    if args.scmc_hidden <= 0:
        raise ValueError("scmc-hidden must be positive.")
    if not 1 <= args.subset_samples_per_batch <= 8:
        raise ValueError("subset-samples-per-batch must be in [1, 8].")
    if not 0.0 <= args.scmc_strength <= 1.0:
        raise ValueError("scmc-strength must be in [0, 1].")
    if not 0.0 <= args.failure_dropout <= 1.0:
        raise ValueError("failure-dropout must be in [0, 1].")
    if not 0.0 <= args.route_exploration_probability <= 1.0:
        raise ValueError("route-exploration-probability must be in [0, 1].")
    if not 0.0 <= args.router_availability_dropout < 1.0:
        raise ValueError("router-availability-dropout must be in [0, 1).")
    if not 0.0 <= args.sparse_availability_dropout < 1.0:
        raise ValueError("sparse-availability-dropout must be in [0, 1).")
    if not 0.5 < args.calibration_coverage < 1.0:
        raise ValueError("calibration-coverage must be in (0.5, 1.0).")
    if args.router_temperature <= 0:
        raise ValueError("router-temperature must be positive.")
    if args.router_listwise_weight < 0:
        raise ValueError("router-listwise-weight must be non-negative.")
    if args.analysis_max_samples < 0:
        raise ValueError("analysis-max-samples must be non-negative.")
    if args.tsne_samples < 2:
        raise ValueError("tsne-samples must be at least 2.")
    if any(
        not 1 <= value <= len(MODALITY_NAMES)
        for value in args.target_selected_cardinalities
    ):
        raise ValueError(
            "Every target-selected-cardinalities value must be between 1 and "
            f"{len(MODALITY_NAMES)}."
        )
    if args.tsne_perplexity <= 0:
        raise ValueError("tsne-perplexity must be positive.")
    if args.tsne_iterations < 250:
        raise ValueError("tsne-iterations must be at least 250.")
    if args.plain_epochs < 1:
        raise ValueError("plain-epochs must be positive.")
    if args.plain_lr <= 0:
        raise ValueError("plain-lr must be positive.")
    if args.analysis_dpi < 72:
        raise ValueError("analysis-dpi must be at least 72.")
    if args.router_smooth_weight < 0 or args.router_sufficiency_weight < 0:
        raise ValueError("router loss weights must be non-negative.")
    return configure_run_paths(args)


def json_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict) -> None:
    """Use atomic replacement so an interrupted seed cannot corrupt metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_serializable(payload), handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def write_config(args) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "config.json", vars(args))


def shutdown_loaders(loaders: Optional[Dict[str, DataLoader]]) -> None:
    """Release persistent DataLoader workers before the next seed starts."""
    if not loaders:
        return
    for loader in loaders.values():
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            loader._iterator = None


def release_seed_resources(loaders: Optional[Dict[str, DataLoader]] = None) -> None:
    shutdown_loaders(loaders)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_seed(args, device: torch.device) -> Optional[dict]:
    """Run the unmodified v6.3 stage sequence for one isolated model seed."""
    seed_everything(args.seed)
    write_config(args)
    print("=" * 80)
    print(f"seed={args.seed} split_seed={args.split_seed} run_dir={args.run_dir}")
    print("=" * 80)

    if args.stage == "smoke":
        smoke_test(args, device)
        release_seed_resources()
        return None

    loaders = None
    try:
        loaders = build_loaders(args, cache_order=args.stage == "cache")
        if args.stage in ("teacher", "all"):
            train_teacher(args, loaders, device)
        if args.stage in ("cache", "all"):
            if not args.teacher_ckpt.exists():
                raise FileNotFoundError(f"Train the teacher first: {args.teacher_ckpt}")
            # Cache order is independent of shuffled training order and is
            # stable across resume/restart for each seed.
            cache_loaders = (
                loaders if args.stage == "cache"
                else build_loaders(args, cache_order=True)
            )
            try:
                run_cache(args, cache_loaders, device)
            finally:
                if cache_loaders is not loaders:
                    shutdown_loaders(cache_loaders)
        if args.stage in ("router", "all"):
            train_router(args, device)
        if args.stage in ("calibrate", "all"):
            if not args.router_ckpt.exists():
                raise FileNotFoundError(f"Train the router first: {args.router_ckpt}")
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
        release_seed_resources(loaders)


def make_seed_args(base_args, seed: int):
    seed_args = copy.deepcopy(base_args)
    seed_args.seed = int(seed)
    seed_args.run_dir = base_args.run_dir / f"seed_{seed}"
    return configure_run_paths(seed_args)


def add_evaluation_metrics(row: dict, prefix: str, metrics: dict) -> None:
    for metric_name in ("macc", "mf1", "model_latency_ms_per_clip"):
        if metric_name in metrics:
            row[f"{prefix}_{metric_name}"] = metrics[metric_name]
    for task in TASK_NAMES:
        if task in metrics.get("acc", {}):
            row[f"{prefix}_{task}_acc"] = metrics["acc"][task]
        if task in metrics.get("f1", {}):
            row[f"{prefix}_{task}_f1"] = metrics["f1"][task]


def seed_result_row(seed: int, result: dict) -> dict:
    dense = result["dense"]
    dynamic = result["dynamic"]
    row = {
        "seed": int(seed),
        "macc_drop_pp": result.get(
            "macc_drop_pp", 100.0 * (dense["macc"] - dynamic["macc"])
        ),
    }
    add_evaluation_metrics(row, "dense", dense)
    add_evaluation_metrics(row, "dynamic", dynamic)
    for task in TASK_NAMES:
        if task in dense.get("acc", {}) and task in dynamic.get("acc", {}):
            row[f"dynamic_minus_dense_{task}_acc_pp"] = 100.0 * (
                dynamic["acc"][task] - dense["acc"][task]
            )
        if task in dense.get("f1", {}) and task in dynamic.get("f1", {}):
            row[f"dynamic_minus_dense_{task}_f1_pp"] = 100.0 * (
                dynamic["f1"][task] - dense["f1"][task]
            )
    for metric_name in (
        "avg_modalities", "avg_cost", "full_cost", "relative_branch_cost",
        "available_baseline_cost", "relative_to_available_cost",
        "numerical_fallback_rate", "selected_full_available_rate",
        "confidence_supported_selection_rate", "mean_selected_relative_regret",
        "mean_selected_relative_upper_regret",
    ):
        if metric_name in dynamic:
            row[f"dynamic_{metric_name}"] = dynamic[metric_name]
    for modality in MODALITY_NAMES:
        if modality in dynamic.get("selection_rate", {}):
            row[f"selection_rate_{modality}"] = dynamic["selection_rate"][modality]

    fixed = result.get("matched_cardinality_best_fixed_subset", {})
    if fixed:
        row["fixed_subset"] = "+".join(fixed.get("modalities", []))
        for key in (
            "rounded_validation_dynamic_cardinality", "validation_macc",
            "validation_mf1", "test_macc", "test_mf1",
            "dynamic_minus_fixed_macc_pp",
        ):
            if key in fixed:
                row[f"fixed_{key}"] = fixed[key]

    for missing_name, failure in result.get("single_stream_failure", {}).items():
        for mode in ("static_available_streams", "dynamic"):
            metrics = failure.get(mode, {})
            prefix = f"{missing_name}_{mode}"
            for key in ("macc", "mf1", "model_latency_ms_per_clip", "avg_modalities"):
                if key in metrics:
                    row[f"{prefix}_{key}"] = metrics[key]
    return row


def aggregate_multi_seed_results(base_args, require_all: bool = True) -> dict:
    rows = []
    missing = []
    for seed in base_args.seeds:
        result_path = base_args.run_dir / f"seed_{seed}" / "final_test.json"
        if not result_path.exists():
            missing.append(int(seed))
            continue
        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        rows.append(seed_result_row(seed, result))

    if not rows:
        raise FileNotFoundError(
            f"No completed final_test.json files exist under {base_args.run_dir}"
        )
    if missing and require_all:
        raise FileNotFoundError(
            f"Missing final results for seeds {missing}. Resume them before aggregation."
        )

    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    per_seed_path = base_args.run_dir / "multi_seed_results.csv"
    frame.to_csv(per_seed_path, index=False)

    summary_rows = []
    summary_dict = {}
    for column in frame.columns:
        if column == "seed" or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        stats = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
        summary_dict[column] = stats
        summary_rows.append({"metric": column, **stats})

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(base_args.run_dir / "multi_seed_summary.csv", index=False)
    payload = {
        "method": "COVA-MTL v6.3 five-seed evaluation",
        "requested_seeds": [int(seed) for seed in base_args.seeds],
        "completed_seeds": [int(seed) for seed in frame["seed"].tolist()],
        "missing_seeds": missing,
        "split_seed": int(base_args.split_seed),
        "split_dir": str(base_args.split_dir),
        "standard_deviation": "sample standard deviation (ddof=1)",
        "per_seed_csv": str(per_seed_path),
        "summary_csv": str(base_args.run_dir / "multi_seed_summary.csv"),
        "metrics": summary_dict,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(base_args.run_dir / "multi_seed_summary.json", payload)

    print("\nFive-seed summary")
    print("-" * 80)
    for key in (
        "dense_macc", "dynamic_macc", "macc_drop_pp", "dense_mf1",
        "dynamic_mf1", "dynamic_avg_modalities", "dynamic_relative_branch_cost",
        "dense_model_latency_ms_per_clip", "dynamic_model_latency_ms_per_clip",
    ):
        if key in summary_dict:
            stats = summary_dict[key]
            print(f"{key}: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['n']})")
    print(f"saved: {per_seed_path}")
    print(f"saved: {base_args.run_dir / 'multi_seed_summary.csv'}")
    print(f"saved: {base_args.run_dir / 'multi_seed_summary.json'}")
    return payload


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and args.stage not in ("smoke", "aggregate") and device.type != "cuda":
        raise RuntimeError(
            "CUDA is not available. Check NVIDIA driver/CUDA/PyTorch before training, "
            "or pass --no-require-cuda intentionally."
        )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    if args.stage == "plain_qualitative":
        run_plain_baseline_qualitative_pipeline(args, device)
        return

    # Default execution is read-only modality analysis and returns before all
    # teacher/router/student training code.
    if args.stage == "modality_tsne":
        run_modalitywise_tsne_analysis(args, device)
        return

    # Default execution is analysis-only and returns before every training loop.
    if args.stage == "representation_analysis":
        run_configuration_representation_analysis(args, device)
        return

    if args.stage == "aggregate":
        aggregate_multi_seed_results(args, require_all=False)
        return

    if not args.multi_seed:
        run_single_seed(args, device)
        return

    # A smoke test checks graph correctness once; repeating it five times adds
    # no evidence and needlessly creates five output trees.
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
        status = {
            "requested_seeds": [int(seed) for seed in args.seeds],
            "split_seed": int(args.split_seed),
            "stage": args.stage,
            "seeds": {},
        }
    status["requested_seeds"] = [int(seed) for seed in args.seeds]
    status["split_seed"] = int(args.split_seed)
    status["stage"] = args.stage

    for index, seed in enumerate(args.seeds, 1):
        seed_args = make_seed_args(args, seed)
        print(f"\nMULTI-SEED {index}/{len(args.seeds)}: seed {seed}")
        status["seeds"][str(seed)] = {
            "status": "running",
            "run_dir": str(seed_args.run_dir),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(status_path, status)
        try:
            run_single_seed(seed_args, device)
            status["seeds"][str(seed)].update({
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "final_test": str(seed_args.run_dir / "final_test.json")
                if (seed_args.run_dir / "final_test.json").exists() else None,
            })
            write_json(status_path, status)
        except Exception as error:
            status["seeds"][str(seed)].update({
                "status": "failed",
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })
            write_json(status_path, status)
            if not args.continue_on_seed_error:
                raise
            print(f"seed {seed} failed; continuing because --continue-on-seed-error is enabled")

    if args.stage in ("eval", "all"):
        aggregate_multi_seed_results(
            args, require_all=not args.continue_on_seed_error
        )


if __name__ == "__main__":
    main()
