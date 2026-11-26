"""
COVA-MTL v6.3 ACVE counterfactual-value representation ablation
================================================================

This file preserves the supplied v6.3 model and compares three progressively
richer ACVE output representations under one shared subset-capable Teacher and
one shared 31-subset cache:

1. scalar_deterministic    : one mean four-task regret per candidate;
2. taskwise_deterministic  : four task-wise mean regrets per candidate;
3. taskwise_probabilistic  : four task-wise means and four uncertainty scales.

Only ACVE and the route-conditioned sparse Student are trained independently.
The Teacher, scout representation, counterfactual targets, data split, SSR
mean-admissibility gate, optimizer schedule, and random seed are matched.  This
prevents the Teacher/cache drift that invalidated the earlier end-to-end ACVE
ablation.  Running the file without arguments executes all three variants and
writes integrated JSON/CSV results.

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
python ablations/acve_value_representation.py --data-root /path/to/AIDE --run-dir /path/to/RUNS/ACVE_representation
python ablations/acve_value_representation.py --representation-ablation-seeds 42,43,44,45,46
python ablations/acve_value_representation.py --stage all --acve-representation taskwise_probabilistic --run-dir /path/to/RUNS/single_full

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
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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
    ensure_splits(args.data_root, args.split_dir, args.seed)
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
    """Predicts the configured counterfactual value for all non-empty subsets.

    The target is relative to the set of streams that is physically available
    for the current sample. Explicit availability conditioning separates
    "not selected" from "not observable" and lets one 31-subset cache supervise
    normal, single-failure, and multiple-failure scenarios.
    """

    VALID_REPRESENTATIONS = (
        "scalar_deterministic",
        "taskwise_deterministic",
        "taskwise_probabilistic",
    )

    def __init__(
        self,
        scout_dim: int = 128,
        hidden: int = 256,
        representation: str = "taskwise_probabilistic",
    ):
        super().__init__()
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"Unknown ACVE representation: {representation}; "
                f"choose one of {self.VALID_REPRESENTATIONS}"
            )
        self.representation = representation
        self.task_wise = representation != "scalar_deterministic"
        self.probabilistic = representation == "taskwise_probabilistic"
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
        mean_dimension = 4 if self.task_wise else 1
        output_dimension = 2 * mean_dimension if self.probabilistic else mean_dimension
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, output_dimension),
        )

    def forward(
        self, scout_tokens: torch.Tensor, available: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        if not self.probabilistic:
            return raw, None
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
    task_mean: torch.Tensor, task_scale: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = task_mean.float().mean(-1)
    if task_scale is None:
        # Deterministic variants do not claim predictive uncertainty.  A zero
        # placeholder keeps routing/result interfaces shape-compatible; it is
        # never used as a calibrated risk term.
        return mean, torch.zeros_like(mean)
    # Root-sum-square propagation is conservative relative to simply averaging
    # the four predicted task scales and remains differentiable.
    scale = task_scale.float().square().sum(-1).sqrt() / task_scale.size(-1)
    return mean, scale.clamp_min(1e-4)


def calibrated_relative_upper_regret(
    router: AvailabilityConditionedSufficientRouter,
    task_mean: torch.Tensor,
    task_scale: Optional[torch.Tensor],
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
    if router.probabilistic:
        relative_scale = (
            scale.square() + reference_scale[:, None].square()
        ).sqrt().clamp_min(1e-4)
        availability_cardinality = available.sum(1).long() - 1
        candidate_cardinality = router.candidate_cardinality.long() - 1
        q = router.calibration_q.to(mean.device)[
            availability_cardinality[:, None], candidate_cardinality[None]
        ]
        relative_upper = relative_mean + q * relative_scale
    else:
        # Both deterministic variants use the same mean-admissibility gate and
        # mean-only same-cardinality tie-break.  They never borrow uncertainty
        # or calibration from the probabilistic Full model.
        relative_scale = torch.zeros_like(relative_mean)
        relative_upper = relative_mean.clone()
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
    selected_task_scale = (
        predicted_task_scale.float()[rows, best_candidate]
        if predicted_task_scale is not None
        else torch.zeros_like(selected_task_regret)
    )
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
        "acve_representation": router.representation,
        "probabilistic_uncertainty": bool(router.probabilistic),
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
    if router.task_wise:
        regression_target = target_task_regret.float()
    else:
        regression_target = target_task_regret.float().mean(-1, keepdim=True)
    valid_outputs = valid[:, :, None].expand_as(predicted_task_mean)
    error = predicted_task_mean.float() - regression_target
    if router.probabilistic:
        scale = predicted_task_scale.float().clamp_min(1e-4)
        nll_values = torch.log(scale) + 0.5 * (error / scale).square()
        nll = nll_values[valid_outputs].mean()
    else:
        nll = predicted_task_mean.new_zeros(())
    smooth = F.smooth_l1_loss(
        predicted_task_mean.float()[valid_outputs],
        regression_target[valid_outputs],
    )

    predicted_mean, predicted_scale = predicted_regret_statistics(
        predicted_task_mean, predicted_task_scale
    )
    target_mean = target_task_regret.float().mean(-1)
    sufficient_target = (target_mean <= 0.0).float()
    if router.probabilistic:
        sufficiency_logit = -predicted_mean / predicted_scale.detach().clamp_min(1e-4)
    else:
        # A fixed temperature turns deterministic regret into a boundary logit
        # without fabricating an uncertainty estimate.
        sufficiency_logit = -predicted_mean / max(float(args.router_temperature), 1e-4)
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
    regression = (
        nll + args.router_smooth_weight * smooth
        if router.probabilistic else smooth
    )
    loss = (
        regression
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
        args.embed, args.router_hidden, args.acve_representation
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
        args.embed, args.router_hidden, args.acve_representation
    ).to(device)
    load_checkpoint(args.router_ckpt, router)
    router.eval()
    if not router.probabilistic:
        report = {
            "method": "not applicable to deterministic ACVE representation",
            "acve_representation": router.representation,
            "calibration_applicable": False,
            "coverage_target": None,
            "q_by_available_and_selected_cardinality": None,
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
        args.embed, args.router_hidden, args.acve_representation
    ).to(device)
    load_checkpoint(args.router_ckpt, router)
    calibration_path = (
        args.student_calibration_path
        if prefer_student_calibration and args.student_calibration_path.exists()
        else args.teacher_calibration_path
    )
    if router.probabilistic and calibration_path.exists():
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
    elif router.probabilistic and args.require_calibration:
        raise FileNotFoundError(
            f"Run --stage calibrate first; calibration not found: {calibration_path}"
        )
    router.loaded_calibration = (
        calibration_path
        if router.probabilistic and calibration_path.exists() else None
    )
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
        "acve_representation": args.acve_representation,
        "task_wise_counterfactual_output": bool(router.task_wise),
        "probabilistic_uncertainty_output": bool(router.probabilistic),
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


# -----------------------------------------------------------------------------
# ACVE representation-ablation diagnostics
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_representation_diagnostics(
    args,
    device: torch.device,
    split: str,
    availability_scope: str = "normal",
    cache_source: str = "student",
) -> dict:
    """Evaluate representation quality under the actual self-referenced rule.

    ``teacher`` isolates ACVE output semantics against the common counterfactual
    targets used to train all three variants. ``student`` measures the deployed
    router/Student pair after independent sparse-Student adaptation.
    """
    if cache_source == "teacher":
        cache_path = args.oracle_cache_dir / f"{split}.pt"
        prefer_student_calibration = False
    elif cache_source == "student":
        cache_path = args.student_cache_dir / f"{split}.pt"
        prefer_student_calibration = True
    else:
        raise ValueError("cache_source must be 'teacher' or 'student'")
    if not cache_path.exists():
        raise FileNotFoundError(f"Representation diagnostic cache not found: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    scout_all = cache["scout_tokens"].float()
    loss_all = cache["subset_loss"].float()
    prediction_all = cache.get("predictions")
    label_all = cache.get("labels")
    router = load_router_from_args(
        args,
        device,
        prefer_student_calibration=prefer_student_calibration,
    )
    router.eval()

    if availability_scope == "normal":
        patterns = torch.ones(1, 5, dtype=torch.bool, device=device)
    elif availability_scope == "primary":
        patterns = deterministic_availability_patterns(device, all_patterns=False)
    else:
        raise ValueError("availability_scope must be 'normal' or 'primary'")

    sufficient_target_parts: List[torch.Tensor] = []
    sufficient_prediction_parts: List[torch.Tensor] = []
    selected_prediction_parts: List[torch.Tensor] = []
    selected_label_parts: List[torch.Tensor] = []
    selection_sum = torch.zeros(5, dtype=torch.float64)
    mean_absolute_sum = 0.0
    mean_element_count = 0
    task_absolute_sum = 0.0
    task_element_count = 0
    cardinality_absolute_sum = 0.0
    violation_count = 0
    selected_positive_regret_sum = 0.0
    selected_regret_sum = 0.0
    coverage_success = 0
    coverage_count = 0
    predicted_scale_sum = 0.0
    predicted_scale_count = 0
    route_count = 0
    candidate_cardinality = router.candidate_cardinality.to(device)
    batch_size = max(1, int(args.router_batch_size))

    for pattern in patterns:
        for start in range(0, scout_all.size(0), batch_size):
            end = min(start + batch_size, scout_all.size(0))
            scout = scout_all[start:end].to(device, non_blocking=True)
            subset_loss = loss_all[start:end].to(device, non_blocking=True)
            available = pattern[None].expand(end - start, -1)
            routed_scout = scout * available[:, :, None].to(scout.dtype)
            with autocast("cuda", enabled=device.type == "cuda"):
                output_mean, output_scale = router(routed_scout, available)
            target_task, valid = availability_relative_targets(
                subset_loss, available, router.candidate_masks
            )
            rows = torch.arange(end - start, device=device)
            reference_candidate = masks_to_subset_ids(available) - 1
            predicted_relative_output = (
                output_mean.float()
                - output_mean.float()[rows, reference_candidate][:, None, :]
            )
            predicted_relative_mean = predicted_relative_output.mean(-1)
            target_task = target_task.float()
            target_mean = target_task.mean(-1)

            mean_absolute_sum += float(
                (predicted_relative_mean - target_mean).abs()[valid].sum().cpu()
            )
            mean_element_count += int(valid.sum().item())
            if router.task_wise:
                valid_tasks = valid[:, :, None].expand_as(target_task)
                task_absolute_sum += float(
                    (predicted_relative_output - target_task).abs()[valid_tasks].sum().cpu()
                )
                task_element_count += int(valid_tasks.sum().item())

            predicted_sufficient = valid & (predicted_relative_mean <= 0.0)
            target_sufficient = valid & (target_mean <= 0.0)
            sufficient_target_parts.append(target_sufficient[valid].cpu())
            sufficient_prediction_parts.append(predicted_sufficient[valid].cpu())

            oracle_cardinality = candidate_cardinality[None].expand_as(valid).masked_fill(
                ~target_sufficient, 6
            ).min(1).values
            predicted_cardinality = candidate_cardinality[None].expand_as(valid).masked_fill(
                ~predicted_sufficient, 6
            ).min(1).values
            predicted_cardinality = torch.where(
                predicted_cardinality == 6,
                available.sum(1),
                predicted_cardinality,
            )
            cardinality_absolute_sum += float(
                (predicted_cardinality.float() - oracle_cardinality.float()).abs().sum().cpu()
            )

            if router.probabilistic:
                _, _, relative_upper, _, aggregate_scale = calibrated_relative_upper_regret(
                    router, output_mean, output_scale, available
                )
                coverage_success += int((target_mean[valid] <= relative_upper[valid]).sum().item())
                coverage_count += int(valid.sum().item())
                predicted_scale_sum += float(aggregate_scale[valid].sum().cpu())
                predicted_scale_count += int(valid.sum().item())

            route = select_self_referenced_subset(router, scout, available)
            selected_index = route["subset_id"].long() - 1
            selected_actual_regret = target_mean[rows, selected_index]
            violation_count += int((selected_actual_regret > 0.0).sum().item())
            selected_positive_regret_sum += float(
                selected_actual_regret.clamp_min(0.0).sum().cpu()
            )
            selected_regret_sum += float(selected_actual_regret.sum().cpu())
            selection_sum += route["mask"].double().sum(0).cpu()
            route_count += end - start

            if prediction_all is not None and label_all is not None:
                local_rows = torch.arange(end - start)
                chosen_subset_id = route["subset_id"].long().cpu()
                selected_prediction_parts.append(
                    prediction_all[start:end][local_rows, chosen_subset_id].cpu()
                )
                selected_label_parts.append(label_all[start:end].cpu())

    target_binary = torch.cat(sufficient_target_parts).numpy().astype(np.int64)
    prediction_binary = torch.cat(sufficient_prediction_parts).numpy().astype(np.int64)
    sufficiency_macro_f1 = f1_score(
        target_binary,
        prediction_binary,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    cached_metrics = None
    if selected_prediction_parts:
        cached_metrics = metrics_from_label_prediction(
            torch.cat(selected_label_parts), torch.cat(selected_prediction_parts)
        )
    empirical_coverage = (
        coverage_success / max(coverage_count, 1)
        if router.probabilistic else None
    )
    calibration_error = (
        abs(empirical_coverage - float(args.calibration_coverage))
        if router.probabilistic else None
    )
    return {
        "split": split,
        "cache_source": cache_source,
        "availability_scope": availability_scope,
        "acve_representation": router.representation,
        "task_wise": bool(router.task_wise),
        "probabilistic": bool(router.probabilistic),
        "regret_mae": mean_absolute_sum / max(mean_element_count, 1),
        "task_wise_mae": (
            task_absolute_sum / max(task_element_count, 1)
            if router.task_wise else None
        ),
        "sufficiency_macro_f1": float(sufficiency_macro_f1),
        "cardinality_mae": cardinality_absolute_sum / max(route_count, 1),
        "violation_rate": violation_count / max(route_count, 1),
        "mean_positive_selected_regret": (
            selected_positive_regret_sum / max(route_count, 1)
        ),
        "mean_selected_actual_regret": selected_regret_sum / max(route_count, 1),
        "avg_modalities": float(selection_sum.sum() / max(route_count, 1)),
        "selection_rate": {
            name: float(selection_sum[index] / max(route_count, 1))
            for index, name in enumerate(MODALITY_NAMES)
        },
        "coverage_target": (
            float(args.calibration_coverage) if router.probabilistic else None
        ),
        "empirical_one_sided_coverage": empirical_coverage,
        "calibration_error": calibration_error,
        "mean_predicted_scale": (
            predicted_scale_sum / max(predicted_scale_count, 1)
            if router.probabilistic else None
        ),
        "cached_selected_metrics": cached_metrics,
        "cache": str(cache_path),
        "number_of_routing_decisions": int(route_count),
    }


def remove_resumable_checkpoints(args, include_teacher: bool = False) -> List[str]:
    paths = [
        args.run_dir / "router" / "last.pt",
        args.run_dir / "sparse" / "last.pt",
    ]
    if include_teacher:
        paths.insert(0, args.run_dir / "teacher" / "last.pt")
    removed = []
    for path in paths:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def smoke_test(args, device: torch.device) -> None:
    args.pretrained = False
    # Keep the structural test light; production training still uses 16 frames.
    args.t_target = 4
    model = model_from_args(args, device).eval()
    router = AvailabilityConditionedSufficientRouter(
        args.embed, args.router_hidden, args.acve_representation
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
    expected_outputs = 4 if router.task_wise else 1
    assert subset_mean.shape == (batch, 31, expected_outputs)
    if router.probabilistic:
        assert subset_scale.shape == (batch, 31, expected_outputs)
        assert (subset_scale > 0).all()
    else:
        assert subset_scale is None
    assert route["reference_admissible"].all()
    assert (route["predicted_relative_mean_regret"] <= 1e-6).all()
    assert route["confidence_supported"].shape == (batch,)
    assert not failure_route["mask"][:, 4].any()
    print("smoke test passed")
    print("dense logits:", [tuple(x.shape) for x in logits])
    print("dynamic masks:", route["mask"].cpu().tolist())


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
            f"{name} requires comma-separated integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{name} cannot contain duplicate seeds")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COVA-MTL on AIDE")
    parser.add_argument(
        "--stage",
        choices=(
            "teacher", "cache", "router", "calibrate", "sparse",
            "recalibrate", "eval", "all", "smoke", "representation_ablation",
        ),
        default="representation_ablation",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/AIDE_Dataset"))
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("RUNS/AIDE_COVA_MTL_v6_3_ACVE_RepresentationAblation"),
    )
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--acve-representation",
        choices=AvailabilityConditionedSufficientRouter.VALID_REPRESENTATIONS,
        default="taskwise_probabilistic",
        help="ACVE output semantics for an ordinary single pipeline",
    )
    parser.add_argument(
        "--representation-ablation-seeds", type=str, default="42",
        help="comma-separated seeds; use 42,43,44,45,46 for five-seed results",
    )
    parser.add_argument(
        "--rerun-completed", action=argparse.BooleanOptionalAction, default=False,
        help="rerun a representation even when its result JSON exists",
    )
    parser.add_argument(
        "--keep-last-checkpoints", action=argparse.BooleanOptionalAction, default=False,
        help="retain resumable last.pt files after successful completion",
    )
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
    return parser


def finalize_args(args):
    args.data_root = args.data_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.split_dir is None:
        args.split_dir = args.data_root / "RUNS" / "split_lists" / "splits_seed42_8-2_full"
    args.split_dir = args.split_dir.expanduser().resolve()
    if isinstance(args.modality_costs, str):
        args.modality_costs = parse_float_list(args.modality_costs, 5, "modality-costs")
    else:
        args.modality_costs = [float(value) for value in args.modality_costs]
    if isinstance(args.representation_ablation_seeds, str):
        args.representation_ablation_seeds = parse_int_list(
            args.representation_ablation_seeds,
            "representation-ablation-seeds",
        )
    if args.acve_representation not in AvailabilityConditionedSufficientRouter.VALID_REPRESENTATIONS:
        raise ValueError(f"Invalid ACVE representation: {args.acve_representation}")
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
    if args.router_smooth_weight < 0 or args.router_sufficiency_weight < 0:
        raise ValueError("router loss weights must be non-negative.")
    args.teacher_ckpt = args.run_dir / "teacher" / "bestmacc.pt"
    args.oracle_cache_dir = args.run_dir / "oracle_cache"
    args.student_cache_dir = args.run_dir / "student_cache"
    args.router_ckpt = args.run_dir / "router" / "best.pt"
    args.teacher_calibration_path = args.run_dir / "router" / "calibration_teacher.json"
    args.student_calibration_path = args.run_dir / "router" / "calibration_student.json"
    args.sparse_ckpt = args.run_dir / "sparse" / "bestmacc.pt"
    return args


def write_run_config(args) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with open(args.run_dir / "config.json", "w", encoding="utf-8") as handle:
        serializable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        json.dump(serializable, handle, indent=2, ensure_ascii=False)


def run_single_pipeline(args, device: torch.device) -> Optional[dict]:
    """Preserve the original v6.3 stage interface for a single representation."""
    write_run_config(args)
    if args.stage == "smoke":
        smoke_test(args, device)
        return None
    loaders = build_loaders(args, cache_order=args.stage == "cache")
    if args.stage in ("teacher", "all"):
        train_teacher(args, loaders, device)
    if args.stage in ("cache", "all"):
        if not args.teacher_ckpt.exists():
            raise FileNotFoundError(f"Train the teacher first: {args.teacher_ckpt}")
        cache_loaders = build_loaders(args, cache_order=True)
        run_cache(args, cache_loaders, device)
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
        run_student_recalibration(args, recalibration_loaders, device)
    if args.stage in ("eval", "all"):
        return run_final_evaluation(args, loaders, device)
    return None


def prepare_shared_teacher_cache(base_args, seed: int, device: torch.device):
    shared = copy.deepcopy(base_args)
    shared.stage = "all"
    shared.seed = int(seed)
    shared.acve_representation = "taskwise_probabilistic"
    shared.run_dir = base_args.run_dir / "_shared" / f"seed_{seed}"
    shared = finalize_args(shared)
    write_run_config(shared)
    seed_everything(shared.seed)
    if not shared.teacher_ckpt.exists():
        loaders = build_loaders(shared, cache_order=False)
        train_teacher(shared, loaders, device)
    else:
        print(f"[shared] reuse teacher: {shared.teacher_ckpt}")
    required_cache = [
        shared.oracle_cache_dir / f"{split}.pt"
        for split in ("train", "val", "test")
    ]
    if not all(path.exists() for path in required_cache):
        cache_loaders = build_loaders(shared, cache_order=True)
        run_cache(shared, cache_loaders, device)
    else:
        print(f"[shared] reuse 31-subset cache: {shared.oracle_cache_dir}")
    if not base_args.keep_last_checkpoints:
        remove_resumable_checkpoints(shared, include_teacher=True)
    return shared


def flatten_representation_report(report: dict) -> dict:
    teacher_test = report["diagnostics"]["teacher_test_normal"]
    student_test = report["diagnostics"]["student_test_normal"]
    student_primary = report["diagnostics"]["student_test_primary"]
    dynamic = report["final_test"]["dynamic"]
    dense = report["final_test"]["dense"]
    return {
        "variant": report["variant"],
        "seed": report["seed"],
        "task_wise": report["task_wise"],
        "probabilistic": report["probabilistic"],
        "router_parameters": report["router_parameters"],
        # Pure ACVE metrics use the same held-out Teacher targets for all rows.
        "regret_mae": teacher_test["regret_mae"],
        "task_wise_mae": teacher_test["task_wise_mae"],
        "sufficiency_macro_f1": teacher_test["sufficiency_macro_f1"],
        "cardinality_mae": teacher_test["cardinality_mae"],
        "violation_rate": teacher_test["violation_rate"],
        "mean_positive_selected_regret": teacher_test["mean_positive_selected_regret"],
        "calibration_error": teacher_test["calibration_error"],
        "empirical_one_sided_coverage": teacher_test["empirical_one_sided_coverage"],
        # Deployment diagnostics use each variant's independently trained Student.
        "student_violation_rate": student_test["violation_rate"],
        "student_cardinality_mae": student_test["cardinality_mae"],
        "primary_violation_rate": student_primary["violation_rate"],
        "primary_cardinality_mae": student_primary["cardinality_mae"],
        "dense_macc": dense["macc"],
        "dynamic_macc": dynamic["macc"],
        "dynamic_mf1": dynamic["mf1"],
        "macc_drop_pp": report["final_test"]["macc_drop_pp"],
        "avg_modalities": dynamic["avg_modalities"],
        "relative_branch_cost": dynamic["relative_branch_cost"],
        "latency_ms_per_clip": dynamic["model_latency_ms_per_clip"],
    }


def aggregate_representation_rows(rows: List[dict]) -> List[dict]:
    order = {
        "scalar_deterministic": 0,
        "taskwise_deterministic": 1,
        "taskwise_probabilistic": 2,
    }
    frame = pd.DataFrame(rows)
    metric_columns = [
        column for column in frame.columns
        if column not in ("variant", "seed", "task_wise", "probabilistic")
    ]
    aggregate = []
    for variant, group in frame.groupby("variant", sort=False):
        item = {
            "variant": variant,
            "task_wise": bool(group["task_wise"].iloc[0]),
            "probabilistic": bool(group["probabilistic"].iloc[0]),
            "number_of_seeds": int(len(group)),
            "seeds": [int(value) for value in group["seed"].tolist()],
        }
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.dropna()
            item[f"{metric}_mean"] = float(valid.mean()) if len(valid) else None
            item[f"{metric}_std"] = (
                float(valid.std(ddof=1)) if len(valid) > 1
                else (0.0 if len(valid) == 1 else None)
            )
        aggregate.append(item)
    return sorted(aggregate, key=lambda item: order[item["variant"]])


def run_representation_ablation(base_args, device: torch.device) -> dict:
    variants = (
        "scalar_deterministic",
        "taskwise_deterministic",
        "taskwise_probabilistic",
    )
    root = base_args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    reports, rows = [], []
    for seed in base_args.representation_ablation_seeds:
        shared = prepare_shared_teacher_cache(base_args, seed, device)
        for variant_name in variants:
            variant = copy.deepcopy(base_args)
            variant.stage = "all"
            variant.seed = int(seed)
            variant.acve_representation = variant_name
            variant.run_dir = root / variant_name / f"seed_{seed}"
            variant.eval_failures = False
            variant.eval_all_subsets = False
            variant = finalize_args(variant)
            # These artifacts are identical within a seed.  Each representation
            # owns only its router, calibration, sparse Student, and diagnostics.
            variant.teacher_ckpt = shared.teacher_ckpt
            variant.oracle_cache_dir = shared.oracle_cache_dir
            report_path = variant.run_dir / "representation_result.json"
            if report_path.exists() and variant.resume and not variant.rerun_completed:
                print(f"[representation ablation] reuse {variant_name}, seed={seed}")
                with open(report_path, "r", encoding="utf-8") as handle:
                    report = json.load(handle)
            else:
                print("=" * 88)
                print(
                    f"[representation ablation] start {variant_name}, seed={seed}, "
                    f"run_dir={variant.run_dir}"
                )
                print("=" * 88)
                seed_everything(variant.seed)
                write_run_config(variant)
                train_router(variant, device)
                run_teacher_calibration(variant, device)
                loaders = build_loaders(variant, cache_order=False)
                train_sparse(variant, loaders, device)
                cache_loaders = build_loaders(variant, cache_order=True)
                run_student_recalibration(variant, cache_loaders, device)
                final_test = run_final_evaluation(variant, loaders, device)

                # Calibration error must be evaluated on unseen test targets,
                # never on the validation residuals used to estimate q.
                student = model_from_args(variant, device)
                load_checkpoint(variant.sparse_ckpt, student)
                test_cache_path = variant.student_cache_dir / "test.pt"
                if not test_cache_path.exists():
                    build_oracle_cache_for_split(
                        variant,
                        "test",
                        cache_loaders["test"],
                        student,
                        device,
                        variant.student_cache_dir,
                        variant.sparse_ckpt,
                    )
                diagnostics = {
                    # Common Teacher targets isolate the ACVE representation.
                    "teacher_validation_normal": evaluate_representation_diagnostics(
                        variant, device, "val", "normal", cache_source="teacher"
                    ),
                    "teacher_test_normal": evaluate_representation_diagnostics(
                        variant, device, "test", "normal", cache_source="teacher"
                    ),
                    # Variant-specific Student targets quantify deployment behavior.
                    "student_test_normal": evaluate_representation_diagnostics(
                        variant, device, "test", "normal", cache_source="student"
                    ),
                    "student_test_primary": evaluate_representation_diagnostics(
                        variant, device, "test", "primary", cache_source="student"
                    ),
                }
                router_for_count = AvailabilityConditionedSufficientRouter(
                    variant.embed,
                    variant.router_hidden,
                    variant.acve_representation,
                )
                report = {
                    "experiment": "COVA-MTL v6.3 ACVE value-representation ablation",
                    "variant": variant_name,
                    "seed": int(seed),
                    "task_wise": bool(router_for_count.task_wise),
                    "probabilistic": bool(router_for_count.probabilistic),
                    "router_parameters": int(sum(
                        parameter.numel() for parameter in router_for_count.parameters()
                    )),
                    "shared_teacher": str(shared.teacher_ckpt),
                    "shared_oracle_cache": str(shared.oracle_cache_dir),
                    "diagnostics": diagnostics,
                    "final_test": final_test,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                if not variant.keep_last_checkpoints:
                    report["removed_resumable_checkpoints"] = remove_resumable_checkpoints(
                        variant
                    )
                variant.run_dir.mkdir(parents=True, exist_ok=True)
                with open(report_path, "w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2, ensure_ascii=False)
            reports.append(report)
            rows.append(flatten_representation_report(report))

    raw_frame = pd.DataFrame(rows)
    raw_frame.to_csv(root / "representation_ablation_runs.csv", index=False)
    aggregate = aggregate_representation_rows(rows)
    aggregate_csv = []
    for item in aggregate:
        aggregate_csv.append({
            key: value for key, value in item.items() if not isinstance(value, list)
        })
    pd.DataFrame(aggregate_csv).to_csv(
        root / "representation_ablation_summary.csv", index=False
    )
    summary = {
        "experiment": "COVA-MTL v6.3 ACVE value-representation ablation",
        "variants": list(variants),
        "fairness_protocol": [
            "one shared subset-capable Teacher per seed",
            "one shared 31-subset target cache per seed",
            "identical scout tokens, availability contexts, SSR mean gate, split, and schedule",
            "only ACVE output semantics and its compatible uncertainty tie-break differ",
            "each sparse Student is independently trained under its own frozen ACVE routes",
            "ACVE prediction metrics use the same held-out Teacher targets",
            "deployment metrics use each variant's independently trained Student",
            "calibration error is measured only for the probabilistic variant on held-out test targets",
        ],
        "runs": reports,
        "aggregate": aggregate,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(root / "representation_ablation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and args.stage != "smoke" and device.type != "cuda":
        raise RuntimeError(
            "CUDA is not available. Check NVIDIA driver/CUDA/PyTorch before training, "
            "or pass --no-require-cuda intentionally."
        )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    if args.stage == "representation_ablation":
        run_representation_ablation(args, device)
        return
    run_single_pipeline(args, device)


if __name__ == "__main__":
    main()
