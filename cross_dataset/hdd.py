#!/usr/bin/env python3
"""COVA-MTL v6.3 three-seed external validation on the HDD public dataset.

HDD provides one front-camera stream and an eight-channel vehicle-sensor array.
The five selectable computational branches are RGB, pedal, steering, motion,
and turn signal.  VBR is single-label classification from ``target/*.npy``;
TCR is multi-label recognition derived from the Cause/Attention EAF tiers.

Pipeline per model seed
-----------------------
teacher -> cache 31 subsets -> ACVE -> calibration -> sparse SCML student
        -> student recalibration -> dense/dynamic/failure evaluation

One session-level 70/15/15 split is created with ``--split-seed`` and shared by
seeds 42--44.  Original HDD files are never modified.

Expected layout
---------------
HDD_ROOT/
    camera/<session>/<frame>.jpg
    sensor/<session>.npy
    target/<session>.npy
    release_2019_07_08/EAF/*.eaf

Example
-------
python cross_dataset/hdd.py \
  --data-root /path/to/hdd_data \
  --run-dir /path/to/RUNS/HDD_COVA_v6_3_MultiSeed5 \
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
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

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


MODALITY_NAMES = ("rgb", "pedal", "steering", "motion", "signal")
TASK_NAMES = ("VBR", "TCR")
VBR_LABELS = (
    "Background", "intersection passing", "left turn", "right turn",
    "left lane change", "right lane change", "left lane branch",
    "right lane branch", "crosswalk passing", "railroad passing", "merge", "U-turn",
)
TCR_BACKGROUND_LABEL = "background_or_smooth"
TCR_POS_LABELS = (
    "traffic_control", "congestion", "vehicle_interaction",
    "pedestrian_context", "two_wheeler_context", "parked_or_road_obstacle",
)
TASK_CLASSES = (len(VBR_LABELS), len(TCR_POS_LABELS))
SENSOR_FEATURES = (
    "accel", "steer_angle", "steer_speed", "speed", "brake",
    "left_turn_signal", "right_turn_signal", "yaw",
)
SENSOR_GROUPS = {
    "pedal": (0, 4), "steering": (1, 2), "motion": (3, 7), "signal": (5, 6),
}
M = len(MODALITY_NAMES)
K = len(TASK_NAMES)
NUM_SUBSETS = (1 << M) - 1
ALLOWED_MODEL_SEEDS = (42, 43, 44)


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


def append_history(path: Path, row: dict) -> None:
    """Upsert one epoch row while preserving earlier resumed epochs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty and "epoch" in current and "epoch" in row:
        current = current[current.epoch != row["epoch"]]
    updated = pd.concat((current, pd.DataFrame([jsonable(row)])), ignore_index=True)
    if "epoch" in updated:
        updated = updated.sort_values("epoch")
    temporary = path.with_suffix(path.suffix + ".tmp")
    updated.to_csv(temporary, index=False)
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


# -----------------------------------------------------------------------------
# HDD session split, EAF label cache, and window dataset
# -----------------------------------------------------------------------------
def normalize_tcr_label(label: str) -> Optional[int]:
    text = (label or "").strip().lower()
    if not text:
        return None
    if "congestion" in text:
        return 1
    if any(token in text for token in ("sign", "red light", "yellow light", "traffic light")):
        return 0
    if "pedestrian" in text:
        return 3
    if "bicyclist" in text or "motorcyclist" in text:
        return 4
    if "parked" in text or "road work" in text or text.startswith("park"):
        return 5
    if any(token in text for token in ("vehicle", "cut-in", "merging", "turning")):
        return 2
    return None


def eaf_session_id(path: Path) -> str:
    parts = path.stem.split("-")
    return path.stem if len(parts) <= 1 else "".join(parts[:-1])


def build_eaf_map(target_ids: Sequence[str], eaf_dir: Path) -> Dict[str, Path]:
    paths = sorted(eaf_dir.glob("*.eaf"))
    direct = {eaf_session_id(path): path for path in paths}
    result = {}
    for session in target_ids:
        if session in direct:
            result[session] = direct[session]
            continue
        matches = [path for path in paths if session in path.stem or path.stem in session]
        if matches:
            result[session] = matches[0]
    return result


def parse_eaf_intervals(path: Path) -> List[Tuple[int, int, int, str, str]]:
    root = ET.parse(path).getroot()
    slots = {}
    for element in root.iter():
        if element.tag.split("}")[-1] == "TIME_SLOT":
            slots[element.attrib["TIME_SLOT_ID"]] = int(element.attrib["TIME_VALUE"])
    intervals = []
    for tier in root.iter():
        if tier.tag.split("}")[-1] != "TIER":
            continue
        tier_id = tier.attrib.get("TIER_ID", "")
        if "cause" not in tier_id.lower() and "attention" not in tier_id.lower():
            continue
        for annotation in tier.iter():
            if annotation.tag.split("}")[-1] != "ALIGNABLE_ANNOTATION":
                continue
            start = slots.get(annotation.attrib.get("TIME_SLOT_REF1"))
            end = slots.get(annotation.attrib.get("TIME_SLOT_REF2"))
            if start is None or end is None or end <= start:
                continue
            value = ""
            for child in annotation.iter():
                if child.tag.split("}")[-1] == "ANNOTATION_VALUE":
                    value = (child.text or "").strip()
                    break
            label = normalize_tcr_label(value)
            if label is not None:
                intervals.append((start, end, label, value, tier_id))
    return intervals


def build_tcr_cache(args) -> None:
    args.tcr_cache_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(args.target_dir.glob("*.npy"))
    summary_path = args.tcr_cache_dir / "summary.json"
    if targets and summary_path.exists() and all(
        (args.tcr_cache_dir / f"{path.stem}.npy").exists() for path in targets
    ):
        return
    mapping = build_eaf_map([path.stem for path in targets], args.eaf_dir)
    missing, counts = [], Counter()
    for target_path in tqdm(targets, desc="HDD TCR EAF cache"):
        session = target_path.stem
        length = int(np.load(target_path, mmap_mode="r").shape[0])
        labels = np.zeros((length, len(TCR_POS_LABELS)), dtype=np.uint8)
        eaf = mapping.get(session)
        if eaf is None:
            missing.append(session)
        else:
            for start_ms, end_ms, label, raw, tier in parse_eaf_intervals(eaf):
                start = max(0, int(math.floor(start_ms / 1000 * args.sampling_fps)))
                end = min(length, int(math.ceil(end_ms / 1000 * args.sampling_fps)))
                if end > start:
                    labels[start:end, label] = 1
                    counts[TCR_POS_LABELS[label]] += end - start
        np.save(args.tcr_cache_dir / f"{session}.npy", labels)
    write_json(summary_path, {
        "sampling_fps": args.sampling_fps,
        "positive_labels": TCR_POS_LABELS,
        "all_zero_semantics": TCR_BACKGROUND_LABEL,
        "frame_positive_counts": counts,
        "missing_eaf": missing,
        "original_files_modified": False,
    })


def available_sessions(args) -> List[str]:
    return [
        path.stem for path in sorted(args.target_dir.glob("*.npy"))
        if (args.sensor_dir / f"{path.stem}.npy").exists()
        and (args.camera_dir / path.stem).is_dir()
        and (args.tcr_cache_dir / f"{path.stem}.npy").exists()
    ]


def split_sessions(sessions: Sequence[str], seed: int) -> Tuple[List[str], List[str], List[str]]:
    sessions = list(sessions)
    random.Random(seed).shuffle(sessions)
    n_train, n_val = int(0.70 * len(sessions)), int(0.15 * len(sessions))
    return (sorted(sessions[:n_train]), sorted(sessions[n_train:n_train + n_val]),
            sorted(sessions[n_train + n_val:]))


def choose_vbr(labels: np.ndarray, args) -> int:
    labels = np.asarray(labels, dtype=np.int64)
    non_background = labels[labels != 0]
    if args.prefer_non_background and len(non_background) >= args.min_non_background_frames:
        return int(np.bincount(non_background, minlength=len(VBR_LABELS)).argmax())
    return int(np.bincount(labels, minlength=len(VBR_LABELS)).argmax())


def make_windows(args, sessions: Sequence[str]) -> pd.DataFrame:
    rows = []
    for session in sessions:
        vbr = np.load(args.target_dir / f"{session}.npy", mmap_mode="r")
        tcr = np.load(args.tcr_cache_dir / f"{session}.npy", mmap_mode="r")
        sensor = np.load(args.sensor_dir / f"{session}.npy", mmap_mode="r")
        image_count = len(list((args.camera_dir / session).glob("*.jpg")))
        length = min(len(vbr), len(tcr), len(sensor), image_count)
        for start in range(0, length - args.t_target + 1, args.window_stride):
            end = start + args.t_target
            multi = (np.asarray(tcr[start:end]).sum(0) >= args.min_tcr_pos_frames).astype(np.uint8)
            rows.append({
                "session_id": session, "start": start, "end": end,
                "vbr": choose_vbr(vbr[start:end], args),
                "tcr_multi": json.dumps(multi.tolist(), separators=(",", ":")),
                "tcr_num_pos": int(multi.sum()),
            })
    return pd.DataFrame(rows)


def ensure_splits(args) -> None:
    build_tcr_cache(args)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    required = [args.split_dir / f"{name}.csv" for name in ("train", "val", "test")]
    if all(path.exists() for path in required) and (args.split_dir / "split_meta.json").exists():
        return
    sessions = available_sessions(args)
    if len(sessions) < 3:
        raise RuntimeError("HDD requires at least three complete sessions")
    train, val, test = split_sessions(sessions, args.split_seed)
    sizes = {}
    for name, selected in (("train", train), ("val", val), ("test", test)):
        frame = make_windows(args, selected)
        if frame.empty:
            raise RuntimeError(f"HDD {name} split produced no windows")
        frame.to_csv(args.split_dir / f"{name}.csv", index=False)
        sizes[name] = {"sessions": len(selected), "windows": len(frame)}
    write_json(args.split_dir / "split_meta.json", {
        "split_seed": args.split_seed, "protocol": "session-disjoint 70/15/15",
        "t_target": args.t_target, "window_stride": args.window_stride,
        "sampling_fps": args.sampling_fps, "sizes": sizes,
        "vbr_labels": VBR_LABELS, "tcr_positive_labels": TCR_POS_LABELS,
        "sensor_groups": SENSOR_GROUPS,
    })


def sensor_statistics(args) -> Tuple[np.ndarray, np.ndarray]:
    path = args.split_dir / "sensor_stats.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32)
    frame = pd.read_csv(args.split_dir / "train.csv")
    sessions = sorted(frame.session_id.astype(str).unique())
    total, sums, squares = 0, np.zeros(8), np.zeros(8)
    for session in sessions:
        values = np.asarray(np.load(args.sensor_dir / f"{session}.npy", mmap_mode="r"), np.float64)
        if values.ndim != 2 or values.shape[1] != 8:
            raise RuntimeError(f"expected HDD sensor shape [N,8], got {values.shape}: {session}")
        total += len(values); sums += values.sum(0); squares += (values * values).sum(0)
    mean = sums / max(total, 1)
    std = np.sqrt(np.maximum(squares / max(total, 1) - mean * mean, 1e-8))
    std[std < 1e-6] = 1
    write_json(path, {"features": SENSOR_FEATURES, "sessions": sessions,
                      "mean": mean.tolist(), "std": std.tolist()})
    return mean.astype(np.float32), std.astype(np.float32)


class HDDCOVADataset(Dataset):
    def __init__(self, csv_path: Path, args, sensor_mean, sensor_std):
        self.frame, self.args = pd.read_csv(csv_path), args
        self.sensor_mean, self.sensor_std = sensor_mean, sensor_std
        self.sensor_cache = {}

    def __len__(self):
        return len(self.frame)

    def _image(self, path: Path) -> torch.Tensor:
        if cv2 is not None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read HDD frame: {path}")
            image = cv2.cvtColor(cv2.resize(image, (self.args.image_size, self.args.image_size)), cv2.COLOR_BGR2RGB)
        else:
            with Image.open(path) as handle:
                image = np.asarray(handle.convert("RGB").resize((self.args.image_size, self.args.image_size)))
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        session, start, end = str(row.session_id), int(row.start), int(row.end)
        rgb = torch.stack([
            self._image(self.args.camera_dir / session / f"{frame + 1:05d}.jpg")
            for frame in range(start, end)
        ], 1)
        if session not in self.sensor_cache:
            self.sensor_cache[session] = np.asarray(
                np.load(self.args.sensor_dir / f"{session}.npy", mmap_mode="r"), np.float32
            )
        sensor = (self.sensor_cache[session][start:end] - self.sensor_mean) / self.sensor_std
        branches = [torch.from_numpy(sensor[:, SENSOR_GROUPS[name]].astype(np.float32))
                    for name in MODALITY_NAMES[1:]]
        return (rgb, *branches, torch.tensor(int(row.vbr), dtype=torch.long),
                torch.tensor(json.loads(row.tcr_multi), dtype=torch.float32),
                f"{session}:{start}:{end}")


def build_loaders(args, cache_order: bool = False) -> Dict[str, DataLoader]:
    ensure_splits(args)
    mean, std = sensor_statistics(args)
    datasets = {name: HDDCOVADataset(args.split_dir / f"{name}.csv", args, mean, std)
                for name in ("train", "val", "test")}
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
    print("dataset sizes:", {name: len(dataset) for name, dataset in datasets.items()})
    return loaders


def move_batch(batch, device: torch.device):
    inputs = tuple(item.to(device, non_blocking=True) for item in batch[:5])
    labels = (batch[5].to(device, non_blocking=True), batch[6].to(device, non_blocking=True))
    return inputs, labels, batch[7]


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


class TokenSE(nn.Module):
    def __init__(self, embed: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, embed // reduction)
        self.net = nn.Sequential(nn.Linear(embed, hidden, bias=False), nn.ReLU(),
                                 nn.Linear(hidden, embed, bias=False))

    def forward(self, x):
        return x * torch.sigmoid(self.net(x.mean(1)) + self.net(x.amax(1)))[:, None]


class FrameSensorEncoder(nn.Module):
    def __init__(self, embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(128, embed), nn.GELU())
        self.se = TokenSE(embed)

    def forward(self, x):
        return self.se(self.net(x))


class PairTemporalReducer(nn.Module):
    """ETB-style learned reduction from T to T/2 for token sequences."""
    def __init__(self, embed: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(3 * embed, embed), nn.GELU(),
                                  nn.Linear(embed, embed))
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, x):
        if x.size(1) % 2:
            x = torch.cat((x, x[:, -1:]), 1)
        first, second = x[:, 0::2], x[:, 1::2]
        gate = torch.sigmoid(self.gate(torch.cat((first, second, second - first), -1)))
        return first + gate * (second - first)


class LightweightPreview(nn.Module):
    def __init__(self, embed: int, size: int, frames: int):
        super().__init__()
        self.size, self.frames = int(size), int(frames)
        self.rgb_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.rgb_project = nn.Linear(64, embed)
        self.sensor_project = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, embed))
            for _ in range(4)
        ])
        self.type_embedding = nn.Parameter(torch.randn(M, embed) * 0.02)

    @staticmethod
    def sensor_statistics(x):
        return torch.cat((x.mean(1), x.std(1, unbiased=False), x[:, 0], x[:, -1]), -1)

    def forward(self, inputs):
        rgb, *sensors = inputs
        batch, _, steps = rgb.shape[:3]
        count = min(max(self.frames, 1), steps)
        indices = torch.linspace(0, steps - 1, count, device=rgb.device).round().long()
        sampled = rgb.index_select(2, indices).permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampled = F.interpolate(sampled, (self.size, self.size), mode="bilinear", align_corners=False)
        rgb_token = self.rgb_project(self.rgb_cnn(sampled).flatten(1).view(batch, count, -1).mean(1))
        sensor_tokens = [project(self.sensor_statistics(value))
                         for project, value in zip(self.sensor_project, sensors)]
        return torch.stack((rgb_token, *sensor_tokens), 1) + self.type_embedding[None]


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
        weight = mask[:, :, None].to(scout.dtype)
        summary = (scout * weight).sum(1) / weight.sum(1).clamp_min(1)
        context = self.context(torch.cat((summary, mask.to(summary.dtype)), 1))
        output = []
        for m in range(M):
            gamma, beta = self.affine[m](context).chunk(2, 1)
            current = tokens[:, :, m]
            calibrated = current + self.strength * (
                gamma[:, None] * self.norm(current) + beta[:, None]
            )
            output.append(calibrated * mask[:, m, None, None].to(calibrated.dtype))
        return torch.stack(output, 2)


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
        batch, steps = tokens.shape[:2]
        key = self.project(tokens) + self.modality_embedding[None, None]
        key = key.flatten(0, 1)
        query = self.task_tokens[None].expand(batch * steps, -1, -1)
        padding = (~mask)[:, None].expand(-1, steps, -1).reshape(batch * steps, M)
        attended, _ = self.cross(query, key, key, key_padding_mask=padding, need_weights=False)
        state = self.norm1(query + attended)
        state = self.norm2(state + self.ffn(state))
        return state.view(batch, steps, K, -1).permute(0, 2, 1, 3).contiguous()


class COVAHDD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        self.scout = LightweightPreview(args.embed, args.preview_size, args.preview_frames)
        self.backbone = SharedResNet18PSMSBN(args.pretrained, args.train_base)
        self.visual_project = nn.Sequential(nn.Linear(512, args.embed), nn.GELU(), nn.LayerNorm(args.embed))
        self.sensor_encoders = nn.ModuleList([
            FrameSensorEncoder(args.embed, args.dropout) for _ in range(4)
        ])
        self.temporal_reducers = nn.ModuleList([
            PairTemporalReducer(args.embed) for _ in range(M)
        ])
        self.scmc = SubsetConditionedCalibration(args.embed, args.scmc_hidden, args.scmc_strength)
        self.fusion = MaskAwareFusion(args.embed, args.d_model, args.attention_heads, args.dropout)
        self.head_vbr = nn.Linear(args.d_model, len(VBR_LABELS))
        self.head_tcr = nn.Linear(args.d_model, len(TCR_POS_LABELS))

    def encode_modality(self, inputs, modality: int):
        if modality == 0:
            rgb = inputs[0]
            batch, _, steps, height, width = rgb.shape
            frames = rgb.permute(0, 2, 1, 3, 4).reshape(batch * steps, 3, height, width)
            tokens = self.visual_project(self.backbone(frames, 0)).view(batch, steps, -1)
        else:
            tokens = self.sensor_encoders[modality - 1](inputs[modality])
        return self.temporal_reducers[modality](tokens)

    def encode_all(self, inputs):
        scout = self.scout(inputs)
        order = torch.randperm(M).tolist() if self.training else list(range(M))
        encoded = [None] * M
        for modality in order:
            encoded[modality] = self.encode_modality(inputs, modality)
        return scout, torch.stack(encoded, 2)

    def encode_selected(self, mask, inputs):
        batch, steps = inputs[0].size(0), (inputs[0].size(2) + 1) // 2
        streams = []
        for modality in range(M):
            rows = torch.nonzero(mask[:, modality], as_tuple=False).flatten()
            if rows.numel() == 0:
                stream = inputs[0].new_zeros(batch, steps, self.embed)
            else:
                # Index only the branch that will actually be encoded. In
                # particular, sensor selection must not copy/execute RGB.
                selected_inputs = list(inputs)
                selected_inputs[modality] = inputs[modality].index_select(0, rows)
                values = self.encode_modality(tuple(selected_inputs), modality)
                stream = values.new_zeros(batch, steps, self.embed).index_copy(0, rows, values)
            streams.append(stream)
        return torch.stack(streams, 2)

    def predict(self, scout, tokens, mask):
        tokens = self.scmc(tokens, scout, mask)
        task_state = self.fusion(tokens, mask)
        pooled = task_state.mean(2)
        logits = (self.head_vbr(pooled[:, 0]), self.head_tcr(pooled[:, 1]))
        return logits, pooled.mean(1), task_state

    def forward_dense(self, inputs):
        scout, tokens = self.encode_all(inputs)
        mask = torch.ones(inputs[0].size(0), M, dtype=torch.bool, device=inputs[0].device)
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
def route_raw(model, router, inputs, available=None):
    scout = model.scout(inputs)
    route = select_subset(router, scout, available)
    tokens = model.encode_selected(route["mask"], inputs)
    logits, state, task_state = model.predict(scout, tokens, route["mask"])
    route.update({"logits": logits, "state": state, "task_state": task_state,
                  "tokens": tokens, "scout": scout})
    return route


# -----------------------------------------------------------------------------
# Losses and metrics
# -----------------------------------------------------------------------------
def training_weights(args, device):
    frame = pd.read_csv(args.split_dir / "train.csv")
    counts = np.bincount(frame.vbr.to_numpy(np.int64), minlength=len(VBR_LABELS)).astype(np.float64)
    vbr = 1.0 / np.sqrt(np.maximum(counts, 1))
    vbr = torch.tensor(vbr / vbr.mean(), dtype=torch.float32, device=device)
    tcr = np.asarray([json.loads(value) for value in frame.tcr_multi], np.float32)
    positive, negative = tcr.sum(0), len(tcr) - tcr.sum(0)
    pos_weight = torch.tensor(np.clip(np.sqrt(negative / np.maximum(positive, 1)), 0.25, 10),
                              dtype=torch.float32, device=device)
    return vbr, pos_weight


def multitask_loss(logits, labels, vbr_weight=None, tcr_pos_weight=None):
    vbr = F.cross_entropy(logits[0], labels[0], weight=vbr_weight)
    tcr = F.binary_cross_entropy_with_logits(logits[1], labels[1], pos_weight=tcr_pos_weight)
    return 0.5 * (vbr + tcr)


def normalized_task_loss(logits, labels):
    vbr = F.cross_entropy(logits[0], labels[0], reduction="none") / math.log(len(VBR_LABELS))
    tcr = F.binary_cross_entropy_with_logits(logits[1], labels[1], reduction="none").mean(1) / math.log(2)
    return torch.stack((vbr, tcr), 1)


def cross_configuration_kd(student, teacher, temperature, threshold):
    vbr_prob = torch.softmax(teacher[0].detach().float() / temperature, 1)
    vbr_valid = torch.softmax(teacher[0].detach().float(), 1).amax(1) >= threshold
    vbr = student[0].sum() * 0
    if vbr_valid.any():
        vbr = F.kl_div(torch.log_softmax(student[0][vbr_valid] / temperature, 1),
                       vbr_prob[vbr_valid], reduction="batchmean") * temperature ** 2
    tcr_prob = torch.sigmoid(teacher[1].detach().float() / temperature)
    binary_confidence = torch.maximum(torch.sigmoid(teacher[1].detach().float()),
                                      1 - torch.sigmoid(teacher[1].detach().float())).mean(1)
    tcr_valid = binary_confidence >= threshold
    tcr = student[1].sum() * 0
    if tcr_valid.any():
        tcr = F.binary_cross_entropy_with_logits(student[1][tcr_valid] / temperature,
                                                  tcr_prob[tcr_valid]) * temperature ** 2
    return 0.5 * (vbr + tcr)


def new_store():
    return {"vbr_labels": [], "vbr_scores": [], "tcr_labels": [], "tcr_scores": []}


def update_store(store, logits, labels):
    store["vbr_labels"].append(labels[0].detach().cpu())
    store["vbr_scores"].append(torch.softmax(logits[0].detach().float(), 1).cpu())
    store["tcr_labels"].append(labels[1].detach().cpu())
    store["tcr_scores"].append(torch.sigmoid(logits[1].detach().float()).cpu())


def safe_map(labels, scores, include=None):
    indices = range(labels.shape[1]) if include is None else include
    values, by_class = [], {}
    for index in indices:
        if np.unique(labels[:, index]).size < 2:
            by_class[str(index)] = None
            continue
        value = float(average_precision_score(labels[:, index], scores[:, index]))
        values.append(value); by_class[str(index)] = value
    return (float(np.mean(values)) if values else None), by_class


def finish_metrics(store):
    vbr_labels = torch.cat(store["vbr_labels"]).numpy()
    vbr_scores = torch.cat(store["vbr_scores"]).numpy()
    vbr_predictions = vbr_scores.argmax(1)
    vbr_onehot = np.eye(len(VBR_LABELS), dtype=np.int64)[vbr_labels]
    vbr_map_all, vbr_ap = safe_map(vbr_onehot, vbr_scores)
    vbr_map_fg, _ = safe_map(vbr_onehot, vbr_scores, range(1, len(VBR_LABELS)))
    tcr_labels = torch.cat(store["tcr_labels"]).numpy().astype(np.int64)
    tcr_scores = torch.cat(store["tcr_scores"]).numpy()
    tcr_predictions = (tcr_scores >= 0.5).astype(np.int64)
    tcr_map, tcr_ap = safe_map(tcr_labels, tcr_scores)
    any_positive = tcr_labels.sum(1) > 0
    any_hit = np.logical_and(tcr_predictions, tcr_labels).sum(1) > 0
    any_hit_accuracy = float(np.mean(np.where(any_positive, any_hit,
                                              tcr_predictions.sum(1) == 0)))
    joint_values = [value for value in (vbr_map_fg, tcr_map) if value is not None]
    joint_score = float(np.mean(joint_values)) if joint_values else 0.0
    return {
        "selection_metric": "mean(VBR foreground mAP, TCR mAP)",
        "joint_score": joint_score,
        "VBR": {
            "acc": float(accuracy_score(vbr_labels, vbr_predictions)),
            "macro_f1": float(f1_score(vbr_labels, vbr_predictions, average="macro", zero_division=0)),
            "mAP_all": vbr_map_all, "mAP_fg": vbr_map_fg,
            "AP_by_class": {VBR_LABELS[int(index)]: value for index, value in vbr_ap.items()},
        },
        "TCR": {
            "mAP": tcr_map,
            "micro_f1": float(f1_score(tcr_labels, tcr_predictions, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(tcr_labels, tcr_predictions, average="macro", zero_division=0)),
            "sample_f1": float(f1_score(tcr_labels, tcr_predictions, average="samples", zero_division=0)),
            "exact_match": float(np.mean(np.all(tcr_labels == tcr_predictions, axis=1))),
            "any_hit_accuracy": any_hit_accuracy,
            "AP_by_class": {TCR_POS_LABELS[int(index)]: value for index, value in tcr_ap.items()},
        },
    }


def model_from_args(args, device):
    return COVAHDD(args).to(device)


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
             post_route_missing: Optional[int] = None):
    model.eval()
    if router is not None:
        router.eval()
    store = new_store()
    selected_sum = torch.zeros(M)
    subset_count = torch.zeros(1 << M, dtype=torch.long)
    selected_total = cost_total = available_cost_total = 0.0
    sample_count = fallback_total = full_total = confidence_total = 0
    post_route_eligible_total = 0
    relative_mean_total = relative_upper_total = 0.0
    elapsed = 0.0
    measured = 0
    costs = torch.tensor(args.modality_costs, dtype=torch.float32)
    for batch_index, batch in enumerate(tqdm(loader, desc="evaluation", leave=False)):
        inputs, labels, _ = move_batch(batch, device)
        batch_size = inputs[0].size(0)
        available = None
        if forced_available is not None:
            available = torch.tensor(forced_available, dtype=torch.bool, device=device)[None].expand(batch_size, -1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with amp_context(device):
            if dynamic:
                result = route_raw(model, router, inputs, available)
                logits = result["logits"]
                if post_route_missing is not None:
                    execution_mask = result["mask"].clone()
                    eligible = execution_mask[:, post_route_missing] & (execution_mask.sum(1) > 1)
                    execution_mask[eligible, post_route_missing] = False
                    logits = model.predict(result["scout"], result["tokens"], execution_mask)[0]
                    post_route_eligible_total += int(eligible.sum())
            elif available is None:
                logits = model.forward_dense(inputs)
            else:
                scout = model.scout(inputs)
                tokens = model.encode_selected(available, inputs)
                logits = model.predict(scout, tokens, available)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        if batch_index >= args.latency_warmup_batches:
            elapsed += time.perf_counter() - start
            measured += batch_size
        update_store(store, logits, labels)
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
    metrics = finish_metrics(store)
    metrics["model_latency_ms_per_clip"] = 1000.0 * elapsed / max(measured, 1)
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
            "selection_rate": {name: float(selected_sum[i] / max(sample_count, 1))
                               for i, name in enumerate(MODALITY_NAMES)},
            "selection_pattern_rate": {
                "+".join(MODALITY_NAMES[m] for m in range(M) if subset & (1 << m)):
                    float(subset_count[subset] / max(sample_count, 1))
                for subset in range(1, 1 << M) if subset_count[subset] > 0
            },
        })
        if post_route_missing is not None:
            metrics["post_route_missing_modality"] = MODALITY_NAMES[post_route_missing]
            metrics["post_route_failure_eligible_rate"] = (
                post_route_eligible_total / max(sample_count, 1)
            )
    return metrics


@torch.no_grad()
def evaluate_all_subsets(model, loader, device):
    model.eval()
    masks = mask_table(device)
    stores = {subset: new_store() for subset in range(1, 1 << M)}
    for batch in tqdm(loader, desc=f"{NUM_SUBSETS}-subset evaluation", leave=False):
        inputs, labels, _ = move_batch(batch, device)
        batch_size = inputs[0].size(0)
        with amp_context(device):
            scout, tokens = model.encode_all(inputs)
            for subset in range(1, 1 << M):
                mask = masks[subset][None].expand(batch_size, -1)
                logits = model.predict(scout, tokens, mask)[0]
                update_store(stores[subset], logits, labels)
    by_subset, by_cardinality = {}, {}
    for subset, store in stores.items():
        names = [MODALITY_NAMES[m] for m in range(M) if subset & (1 << m)]
        by_subset[str(subset)] = {"modalities": names, "cardinality": len(names),
                                  **finish_metrics(store)}
    for cardinality in range(1, M + 1):
        group = [value for value in by_subset.values() if value["cardinality"] == cardinality]
        scores = np.asarray([value["joint_score"] for value in group])
        by_cardinality[str(cardinality)] = {
            "subset_count": len(group), "mean_joint_score": float(scores.mean()),
            "best_joint_score": float(scores.max()), "worst_joint_score": float(scores.min()),
        }
    return {"by_cardinality": by_cardinality, "by_subset": by_subset}


# -----------------------------------------------------------------------------
# Stage 1: subset-capable dense teacher
# -----------------------------------------------------------------------------
def train_teacher(args, loaders, device):
    stage_dir = args.run_dir / "teacher"
    last, best = stage_dir / "last.pt", stage_dir / "bestscore.pt"
    model = model_from_args(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay)
    scaler = make_scaler(device)
    vbr_weight, tcr_pos_weight = training_weights(args, device)
    start, best_score = 1, -1.0
    if args.resume and last.exists():
        checkpoint = load_checkpoint(last, model, optimizer, scaler)
        start = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best", {}).get("joint_score", -1))
    for epoch in range(start, args.teacher_epochs + 1):
        model.train()
        running = count = 0.0
        for batch in tqdm(loaders["train"], desc=f"teacher {epoch}", leave=False):
            inputs, labels, _ = move_batch(batch, device)
            batch_size = inputs[0].size(0)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                scout, tokens = model.encode_all(inputs)
                full_mask = torch.ones(batch_size, M, dtype=torch.bool, device=device)
                full_logits = model.predict(scout, tokens, full_mask)[0]
                full_loss = multitask_loss(full_logits, labels, vbr_weight, tcr_pos_weight)
                subset_loss = full_loss.new_zeros(())
                kd_loss = full_loss.new_zeros(())
                for _ in range(args.subset_samples_per_batch):
                    subset_mask = sample_balanced_masks(batch_size, device, include_full=True)
                    subset_logits = model.predict(scout, tokens, subset_mask)[0]
                    subset_loss += multitask_loss(subset_logits, labels, vbr_weight, tcr_pos_weight)
                    kd_loss += cross_configuration_kd(subset_logits, full_logits,
                                                      args.kd_temperature, args.kd_confidence)
                subset_loss /= args.subset_samples_per_batch
                kd_loss /= args.subset_samples_per_batch
                heavy_target = tokens.detach().mean(1)
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
        vbr_text = "nan" if val["VBR"]["mAP_fg"] is None else f"{val['VBR']['mAP_fg']:.4f}"
        tcr_text = "nan" if val["TCR"]["mAP"] is None else f"{val['TCR']['mAP']:.4f}"
        print(f"[teacher {epoch}] loss={running/max(count,1):.4f} "
              f"val_joint={val['joint_score']:.4f} VBR_fg_mAP={vbr_text} TCR_mAP={tcr_text}")
        append_history(stage_dir / "history.csv", {
            "epoch": epoch, "train_loss": running / max(count, 1),
            "val_joint_score": val["joint_score"],
            "val_VBR_mAP_fg": val["VBR"]["mAP_fg"],
            "val_VBR_mAP_all": val["VBR"]["mAP_all"],
            "val_TCR_mAP": val["TCR"]["mAP"],
        })
        if val["joint_score"] > best_score:
            best_score = val["joint_score"]
            save_checkpoint(best, epoch, model, optimizer, scaler, {"joint_score": best_score})
        save_checkpoint(last, epoch, model, optimizer, scaler, {"joint_score": best_score})
    return best


# -----------------------------------------------------------------------------
# Stage 2: exhaustive counterfactual cache and ACVE
# -----------------------------------------------------------------------------
@torch.no_grad()
def build_cache(args, split, loader, model, device, cache_dir, source_checkpoint):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}.pt"
    source_stat = Path(source_checkpoint).stat()
    source_signature = {
        "path": str(Path(source_checkpoint).resolve()),
        "size": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
    }
    split_path = args.split_dir / f"{split}.csv"
    split_stat = split_path.stat()
    cache_signature = {
        "model": source_signature,
        "split_csv": {
            "path": str(split_path.resolve()), "size": int(split_stat.st_size),
            "mtime_ns": int(split_stat.st_mtime_ns),
        },
        "task_count": K, "modality_count": M,
    }
    if path.exists():
        try:
            existing = torch.load(path, map_location="cpu", weights_only=False)
            if existing.get("cache_signature") == cache_signature:
                print(f"reuse counterfactual cache: {path}")
                return path
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
    model.eval()
    masks = mask_table(device)
    collected = {"scout_tokens": [], "subset_loss": [], "vbr_labels": [],
                 "tcr_labels": [], "sample_ids": []}
    for batch in tqdm(loader, desc=f"counterfactual cache {split}", leave=False):
        inputs, labels, sample_ids = move_batch(batch, device)
        batch_size = inputs[0].size(0)
        with amp_context(device):
            scout, tokens = model.encode_all(inputs)
            losses = torch.full((batch_size, 1 << M, K), torch.nan, device=device)
            for subset in range(1, 1 << M):
                mask = masks[subset][None].expand(batch_size, -1)
                logits = model.predict(scout, tokens, mask)[0]
                losses[:, subset] = normalized_task_loss(logits, labels)
        collected["scout_tokens"].append(scout.half().cpu())
        collected["subset_loss"].append(losses.float().cpu())
        collected["vbr_labels"].append(labels[0].cpu())
        collected["tcr_labels"].append(labels[1].cpu())
        collected["sample_ids"].extend(list(sample_ids))
    output = {key: torch.cat(value, 0) for key, value in collected.items()
              if key != "sample_ids"}
    output.update({
        "sample_ids": collected["sample_ids"], "mask_table": masks.cpu(),
        "modality_names": MODALITY_NAMES, "task_names": TASK_NAMES,
        "source_checkpoint": str(source_checkpoint),
        "source_signature": source_signature,
        "cache_signature": cache_signature,
        "target_semantics": (
            "two task-wise counterfactual losses for all 31 non-empty branch subsets: "
            "VBR CE/log(12) and mean TCR BCE/log(2)"
        ),
    })
    temporary = path.with_suffix(".pt.tmp")
    torch.save(output, temporary)
    temporary.replace(path)
    print(f"saved {path} ({len(collected['sample_ids'])} samples)")
    return path


def analyze_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    losses = cache["subset_loss"][:, 1:].mean(2)
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
        append_history(args.run_dir / "router" / "history.csv", {
            "epoch": epoch, "train_loss": running / max(count, 1), **val,
        })
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            save_checkpoint(best, epoch, router, optimizer, scaler, {"loss": best_loss})
        save_checkpoint(last, epoch, router, optimizer, scaler, {"loss": best_loss})
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
    vbr_weight, tcr_pos_weight = training_weights(args, device)
    scaler = make_scaler(device)
    last, best = args.run_dir / "sparse" / "last.pt", args.run_dir / "sparse" / "bestscore.pt"
    start, best_score = 1, -1.0
    if args.resume and last.exists():
        checkpoint = load_checkpoint(last, student, optimizer, scaler)
        start = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best", {}).get("joint_score", -1))
    for epoch in range(start, args.sparse_epochs + 1):
        student.train()
        student.scout.eval()
        freeze_bn_statistics(student)
        running = count = 0.0
        for batch in tqdm(loaders["train"], desc=f"sparse {epoch}", leave=False):
            inputs, labels, _ = move_batch(batch, device)
            available = random_available(inputs[0].size(0), device, args.sparse_availability_dropout)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), amp_context(device):
                scout, teacher_tokens = teacher.encode_all(inputs)
                teacher_logits = teacher.predict(scout, teacher_tokens, available)[0]
                route = select_subset(router, scout, available)
                selected_mask = explore_route(route["mask"], route["alternative_mask"],
                                              route["has_alternative"],
                                              args.route_exploration_probability)
            execution_mask = randomly_drop_one(selected_mask, args.failure_dropout)
            with amp_context(device):
                # dtype-safe: mask and indices only; no Half -> Float indexed assignment.
                student_tokens = student.encode_selected(selected_mask, inputs)
                student_logits = student.predict(scout.detach(), student_tokens, execution_mask)[0]
                classification = multitask_loss(student_logits, labels, vbr_weight, tcr_pos_weight)
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
        print(f"[sparse {epoch}] loss={running/max(count,1):.4f} "
              f"val_joint={val['joint_score']:.4f} avg_branches={val['avg_modalities']:.3f}")
        append_history(args.run_dir / "sparse" / "history.csv", {
            "epoch": epoch, "train_loss": running / max(count, 1),
            "val_joint_score": val["joint_score"],
            "val_VBR_mAP_fg": val["VBR"]["mAP_fg"],
            "val_TCR_mAP": val["TCR"]["mAP"],
            "val_avg_modalities": val["avg_modalities"],
            "val_relative_branch_cost": val["relative_branch_cost"],
        })
        if val["joint_score"] > best_score:
            best_score = val["joint_score"]
            save_checkpoint(best, epoch, student, optimizer, scaler, {"joint_score": best_score})
        save_checkpoint(last, epoch, student, optimizer, scaler, {"joint_score": best_score})
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
        sid, val_best = max(group, key=lambda item: item[1]["joint_score"])
        test_best = test_subsets["by_subset"][sid]
        matched_fixed = {
            "selection_split": "validation",
            "selection_metric": "mean(VBR foreground mAP, TCR mAP)",
            "rounded_validation_dynamic_cardinality": cardinality,
            "subset_id": int(sid), "modalities": test_best["modalities"],
            "validation_joint_score": val_best["joint_score"],
            "test_joint_score": test_best["joint_score"],
            "test_VBR_mAP_fg": test_best["VBR"]["mAP_fg"],
            "test_TCR_mAP": test_best["TCR"]["mAP"],
            "dynamic_minus_fixed_joint_pp": 100 * (
                dynamic["joint_score"] - test_best["joint_score"]
            ),
        }
    failures = {}
    if args.eval_failures:
        for missing, name in enumerate(MODALITY_NAMES):
            available = [True] * M
            available[missing] = False
            failures[f"missing_{name}"] = {
                "static_available_streams": evaluate(model, loaders["test"], device, args,
                                                     forced_available=available),
                "dynamic": evaluate(model, loaders["test"], device, args, dynamic=True,
                                    router=router, forced_available=available),
                "unexpected_post_route": evaluate(
                    model, loaders["test"], device, args, dynamic=True,
                    router=router, post_route_missing=missing,
                ),
            }
    result = {
        "method": "COVA-MTL v6.3 HDD five-branch two-task external validation",
        "scope_note": (
            "HDD uses five selectable computational branches: one RGB branch and four "
            "branches derived from one shared eight-channel sensor array. They are not "
            "five independent physical sensors."
        ),
        "seed": args.seed, "split_seed": args.split_seed, "run_dir": args.run_dir,
        "split_dir": args.split_dir,
        "modalities": MODALITY_NAMES,
        "tasks": {"VBR": VBR_LABELS, "TCR_positive": TCR_POS_LABELS},
        "window_protocol": {
            "frames": args.t_target, "stride": args.window_stride,
            "sampling_fps": args.sampling_fps,
        },
        "checkpoint": args.sparse_ckpt, "router_checkpoint": args.router_ckpt,
        "calibration": getattr(router, "loaded_calibration", None),
        "selection": (
            "minimum-cardinality availability-valid subset with predicted relative mean "
            "mean two-task loss no greater than the all-currently-available self-reference; calibrated "
            "upper risk breaks same-cardinality ties"
        ),
        "selection_has_no_cost_or_budget_term": True,
        "missing_branch_protocol": (
            "Known absence is supplied before SSR. Unexpected post-routing failure removes "
            "the requested branch only when at least one other selected branch remains; "
            "the eligible rate is reported for each condition."
        ),
        "modality_costs_reporting_only": dict(zip(MODALITY_NAMES, args.modality_costs)),
        "dense": dense, "dynamic": dynamic,
        "matched_cardinality_best_fixed_subset": matched_fixed,
        "exhaustive_nonempty_subsets": test_subsets,
        "single_branch_failure": failures,
        "joint_score_drop_pp": 100 * (dense["joint_score"] - dynamic["joint_score"]),
        "parameter_count": {
            "student": sum(p.numel() for p in model.parameters()),
            "ACVE": sum(p.numel() for p in router.parameters()),
            "SCMC": sum(p.numel() for p in model.scmc.parameters()),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(args.run_dir / "final_test.json", result)
    pd.DataFrame([result_row(args.seed, result)]).to_csv(
        args.run_dir / "final_test_summary.csv", index=False
    )
    if test_subsets:
        subset_rows = []
        for subset_id, value in test_subsets["by_subset"].items():
            subset_rows.append({
                "subset_id": int(subset_id),
                "modalities": "+".join(value["modalities"]),
                "cardinality": value["cardinality"],
                "joint_score": value["joint_score"],
                "VBR_mAP_fg": value["VBR"]["mAP_fg"],
                "VBR_mAP_all": value["VBR"]["mAP_all"],
                "VBR_acc": value["VBR"]["acc"],
                "TCR_mAP": value["TCR"]["mAP"],
                "TCR_micro_f1": value["TCR"]["micro_f1"],
            })
        pd.DataFrame(subset_rows).sort_values("subset_id").to_csv(
            args.run_dir / "exhaustive_31_subsets.csv", index=False
        )
    failure_rows = []
    for condition, value in failures.items():
        for mode in ("static_available_streams", "dynamic", "unexpected_post_route"):
            metric = value[mode]
            failure_rows.append({
                "condition": condition, "mode": mode,
                "joint_score": metric["joint_score"],
                "VBR_mAP_fg": metric["VBR"]["mAP_fg"],
                "TCR_mAP": metric["TCR"]["mAP"],
                "avg_modalities": metric.get("avg_modalities"),
                "relative_branch_cost": metric.get("relative_branch_cost"),
            })
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(
            args.run_dir / "single_branch_failure.csv", index=False
        )
    pd.DataFrame([
        {"selection_pattern": pattern, "rate": rate}
        for pattern, rate in dynamic.get("selection_pattern_rate", {}).items()
    ]).to_csv(args.run_dir / "dynamic_selection_patterns.csv", index=False)
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
    return result


# -----------------------------------------------------------------------------
# Smoke test, CLI, three-seed orchestration
# -----------------------------------------------------------------------------
def smoke_test(args, device):
    args = copy.deepcopy(args)
    args.pretrained = False
    args.image_size = 64
    model = model_from_args(args, device).eval()
    router = ACVE(args.embed, args.router_hidden).to(device).eval()
    args.t_target = 4
    inputs = (torch.rand(2, 3, 4, 64, 64, device=device),
              *[torch.rand(2, 4, 2, device=device) for _ in range(4)])
    with torch.no_grad():
        scout, tokens = model.encode_all(inputs)
        full = torch.ones(2, M, dtype=torch.bool, device=device)
        task_mean, task_scale = router(scout, full)
        logits = model.predict(scout, tokens, full)[0]
        route = route_raw(model, router, inputs)
        missing = full.clone(); missing[:, 1] = False
        failure = select_subset(router, scout, missing)
    assert scout.shape == (2, M, args.embed)
    assert tokens.shape == (2, 2, M, args.embed)
    assert task_mean.shape == (2, NUM_SUBSETS, K)
    assert task_scale.shape == task_mean.shape and (task_scale > 0).all()
    assert logits[0].shape == (2, len(VBR_LABELS))
    assert logits[1].shape == (2, len(TCR_POS_LABELS))
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
    parser = argparse.ArgumentParser(description="COVA-MTL v6.3 on HDD (seeds 42, 43, and 44)")
    parser.add_argument("--stage", choices=("teacher", "cache", "router", "calibrate", "sparse",
                                             "recalibrate", "eval", "all", "aggregate", "smoke"),
                        default="all")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("HDD_ROOT", "data/hdd_data")))
    parser.add_argument("--run-dir", type=Path,
                        default=Path(os.environ.get("HDD_RUN_DIR", "RUNS/HDD_COVA_v6_3_MultiSeed5")))
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--tcr-cache-dir", type=Path, default=None)
    parser.add_argument("--multi-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", default="42,43,44",
                        help="completed model seeds to run/aggregate (default: 42,43,44)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--continue-on-seed-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-completed-seeds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--t-target", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--sampling-fps", type=float, default=3.0)
    parser.add_argument("--prefer-non-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-non-background-frames", type=int, default=2)
    parser.add_argument("--min-tcr-pos-frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--preview-size", type=int, default=64)
    parser.add_argument("--preview-frames", type=int, default=4)
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
    parser.add_argument("--modality-costs", default="1,0.05,0.05,0.05,0.05",
                        help="reporting only; never used by SSR")
    parser.add_argument("--latency-warmup-batches", type=int, default=3)
    parser.add_argument("--eval-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-all-subsets", action=argparse.BooleanOptionalAction, default=True)
    return parser


def configure_paths(args):
    args.teacher_ckpt = args.run_dir / "teacher" / "bestscore.pt"
    args.oracle_cache_dir = args.run_dir / "oracle_cache"
    args.student_cache_dir = args.run_dir / "student_cache"
    args.router_ckpt = args.run_dir / "router" / "best.pt"
    args.teacher_calibration_path = args.run_dir / "router" / "calibration_teacher.json"
    args.student_calibration_path = args.run_dir / "router" / "calibration_student.json"
    args.sparse_ckpt = args.run_dir / "sparse" / "bestscore.pt"
    return args


def finalize_args(args):
    args.data_root = args.data_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.seeds = parse_int_list(args.seeds)
    invalid_seeds = [seed for seed in args.seeds if seed not in ALLOWED_MODEL_SEEDS]
    if invalid_seeds:
        raise ValueError(
            f"this three-seed code only permits {ALLOWED_MODEL_SEEDS}; "
            f"remove unsupported seeds {invalid_seeds}"
        )
    if not args.multi_seed and args.seed not in ALLOWED_MODEL_SEEDS:
        raise ValueError(
            f"single-seed execution also only permits {ALLOWED_MODEL_SEEDS}; "
            f"received --seed {args.seed}"
        )
    args.modality_costs = parse_float_list(args.modality_costs, M)
    args.camera_dir = args.data_root / "camera"
    args.sensor_dir = args.data_root / "sensor"
    args.target_dir = args.data_root / "target"
    args.eaf_dir = args.data_root / "release_2019_07_08" / "EAF"
    if args.stage not in ("smoke", "aggregate"):
        for path in (args.camera_dir, args.sensor_dir, args.target_dir, args.eaf_dir):
            if not path.exists():
                raise FileNotFoundError(f"required HDD path not found: {path}")
    if args.split_dir is None:
        args.split_dir = args.data_root / "RUNS" / (
            f"HDD_COVA_splits_seed{args.split_seed}_T{args.t_target}_stride{args.window_stride}_70-15-15"
        )
    args.split_dir = args.split_dir.expanduser().resolve()
    if args.tcr_cache_dir is None:
        args.tcr_cache_dir = args.data_root / "RUNS" / "HDD_TCR_EAF_multilabel_cache"
    args.tcr_cache_dir = args.tcr_cache_dir.expanduser().resolve()
    if args.embed <= 0 or args.d_model <= 0 or args.router_hidden <= 0:
        raise ValueError("model dimensions must be positive")
    if args.d_model % args.attention_heads:
        raise ValueError("d-model must be divisible by attention-heads")
    if args.t_target < 2 or args.window_stride < 1 or args.preview_frames < 1:
        raise ValueError("invalid temporal settings")
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
    return configure_paths(args)


def result_row(seed, result):
    dense, dynamic = result["dense"], result["dynamic"]
    row = {
        "seed": seed,
        "dense_joint_score": dense["joint_score"],
        "dense_VBR_mAP_fg": dense["VBR"]["mAP_fg"],
        "dense_VBR_mAP_all": dense["VBR"]["mAP_all"],
        "dense_VBR_acc": dense["VBR"]["acc"],
        "dense_VBR_macro_f1": dense["VBR"]["macro_f1"],
        "dense_TCR_mAP": dense["TCR"]["mAP"],
        "dense_TCR_micro_f1": dense["TCR"]["micro_f1"],
        "dense_TCR_macro_f1": dense["TCR"]["macro_f1"],
        "dense_TCR_sample_f1": dense["TCR"]["sample_f1"],
        "dense_TCR_exact_match": dense["TCR"]["exact_match"],
        "dense_TCR_any_hit_accuracy": dense["TCR"]["any_hit_accuracy"],
        "dynamic_joint_score": dynamic["joint_score"],
        "dynamic_VBR_mAP_fg": dynamic["VBR"]["mAP_fg"],
        "dynamic_VBR_mAP_all": dynamic["VBR"]["mAP_all"],
        "dynamic_VBR_acc": dynamic["VBR"]["acc"],
        "dynamic_VBR_macro_f1": dynamic["VBR"]["macro_f1"],
        "dynamic_TCR_mAP": dynamic["TCR"]["mAP"],
        "dynamic_TCR_micro_f1": dynamic["TCR"]["micro_f1"],
        "dynamic_TCR_macro_f1": dynamic["TCR"]["macro_f1"],
        "dynamic_TCR_sample_f1": dynamic["TCR"]["sample_f1"],
        "dynamic_TCR_exact_match": dynamic["TCR"]["exact_match"],
        "dynamic_TCR_any_hit_accuracy": dynamic["TCR"]["any_hit_accuracy"],
        "joint_score_drop_pp": result["joint_score_drop_pp"],
        "dynamic_avg_modalities": dynamic["avg_modalities"],
        "dynamic_avg_cost": dynamic["avg_cost"],
        "dynamic_full_cost": dynamic["full_cost"],
        "dynamic_relative_branch_cost": dynamic["relative_branch_cost"],
        "dynamic_relative_to_available_cost": dynamic["relative_to_available_cost"],
        "dynamic_numerical_fallback_rate": dynamic["numerical_fallback_rate"],
        "dynamic_selected_full_available_rate": dynamic["selected_full_available_rate"],
        "dynamic_confidence_supported_selection_rate": dynamic[
            "confidence_supported_selection_rate"
        ],
        "dynamic_mean_selected_relative_regret": dynamic["mean_selected_relative_regret"],
        "dynamic_mean_selected_relative_upper_regret": dynamic[
            "mean_selected_relative_upper_regret"
        ],
        "dense_latency_ms": dense["model_latency_ms_per_clip"],
        "dynamic_latency_ms": dynamic["model_latency_ms_per_clip"],
        "latency_reduction_percent": 100 * (
            dense["model_latency_ms_per_clip"] - dynamic["model_latency_ms_per_clip"]
        ) / max(dense["model_latency_ms_per_clip"], 1e-12),
    }
    for name, count in result.get("parameter_count", {}).items():
        row[f"parameters_{name}"] = count
    for modality in MODALITY_NAMES:
        row[f"selection_rate_{modality}"] = dynamic["selection_rate"][modality]
    fixed = result.get("matched_cardinality_best_fixed_subset", {})
    if fixed:
        row["fixed_subset"] = "+".join(fixed["modalities"])
        row["fixed_test_joint_score"] = fixed["test_joint_score"]
        row["dynamic_minus_fixed_joint_pp"] = fixed["dynamic_minus_fixed_joint_pp"]
    for name, failure in result.get("single_branch_failure", {}).items():
        row[f"{name}_dynamic_joint_score"] = failure["dynamic"]["joint_score"]
        row[f"{name}_static_joint_score"] = failure["static_available_streams"]["joint_score"]
        row[f"{name}_dynamic_VBR_mAP_fg"] = failure["dynamic"]["VBR"]["mAP_fg"]
        row[f"{name}_dynamic_VBR_acc"] = failure["dynamic"]["VBR"]["acc"]
        row[f"{name}_dynamic_VBR_macro_f1"] = failure["dynamic"]["VBR"]["macro_f1"]
        row[f"{name}_dynamic_TCR_mAP"] = failure["dynamic"]["TCR"]["mAP"]
        row[f"{name}_dynamic_TCR_micro_f1"] = failure["dynamic"]["TCR"]["micro_f1"]
        row[f"{name}_dynamic_TCR_macro_f1"] = failure["dynamic"]["TCR"]["macro_f1"]
        row[f"{name}_dynamic_avg_modalities"] = failure["dynamic"]["avg_modalities"]
        row[f"{name}_dynamic_relative_branch_cost"] = failure["dynamic"][
            "relative_branch_cost"
        ]
        unexpected = failure["unexpected_post_route"]
        row[f"{name}_unexpected_joint_score"] = unexpected["joint_score"]
        row[f"{name}_unexpected_VBR_mAP_fg"] = unexpected["VBR"]["mAP_fg"]
        row[f"{name}_unexpected_TCR_mAP"] = unexpected["TCR"]["mAP"]
        row[f"{name}_unexpected_eligible_rate"] = unexpected[
            "post_route_failure_eligible_rate"
        ]
    return row


def grouped_scalar_summary(frame, group_columns, value_columns):
    """Return mean/std/min/max tables without pandas version-specific named aggregation."""
    records = []
    for group_key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(group_columns, group_key))
        for column in value_columns:
            values = group[column].dropna().to_numpy(float)
            if not len(values):
                continue
            records.append({
                **identity, "metric": column, "n": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()), "max": float(values.max()),
            })
    return pd.DataFrame(records)


def expected_seed_artifacts(args):
    artifacts = [
        "config.json",
        "teacher/bestscore.pt", "teacher/last.pt", "teacher/history.csv",
        "oracle_cache/train.pt", "oracle_cache/val.pt", "oracle_cache/test.pt",
        "router/best.pt", "router/last.pt", "router/history.csv",
        "router/calibration_teacher.json", "router/calibration_student.json",
        "sparse/bestscore.pt", "sparse/last.pt", "sparse/history.csv",
        "student_cache/val.pt",
        "final_test.json", "final_test_summary.csv", "dynamic_selection_patterns.csv",
    ]
    if args.eval_all_subsets:
        artifacts.append("exhaustive_31_subsets.csv")
    if args.eval_failures:
        artifacts.append("single_branch_failure.csv")
    return artifacts


def aggregate(args, require_all=True):
    rows, missing, loaded = [], [], []
    for seed in args.seeds:
        path = args.run_dir / f"seed_{seed}" / "final_test.json"
        if not path.exists():
            missing.append(seed)
            continue
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        loaded.append((seed, result))
        rows.append(result_row(seed, result))
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
        "method": "COVA-MTL v6.3 HDD three-seed external validation",
        "requested_seeds": args.seeds, "completed_seeds": frame.seed.tolist(),
        "missing_seeds": missing, "split_seed": args.split_seed,
        "split_dir": args.split_dir, "std": "sample standard deviation (ddof=1)",
        "metrics": summary, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(args.run_dir / "multi_seed_summary.json", payload)

    # Preserve every paper-facing distribution that is intentionally too large
    # for the one-row-per-seed table above.
    pattern_rows = []
    subset_rows = []
    failure_rows = []
    class_rows = []
    for seed, result in loaded:
        for pattern, rate in result["dynamic"].get("selection_pattern_rate", {}).items():
            pattern_rows.append({"seed": seed, "selection_pattern": pattern, "rate": rate})
        for subset_id, value in result.get("exhaustive_nonempty_subsets", {}).get(
                "by_subset", {}).items():
            subset_rows.append({
                "seed": seed, "subset_id": int(subset_id),
                "modalities": "+".join(value["modalities"]),
                "cardinality": value["cardinality"],
                "joint_score": value["joint_score"],
                "VBR_mAP_fg": value["VBR"]["mAP_fg"],
                "VBR_mAP_all": value["VBR"]["mAP_all"],
                "VBR_acc": value["VBR"]["acc"],
                "VBR_macro_f1": value["VBR"]["macro_f1"],
                "TCR_mAP": value["TCR"]["mAP"],
                "TCR_micro_f1": value["TCR"]["micro_f1"],
                "TCR_macro_f1": value["TCR"]["macro_f1"],
            })
        for condition, modes in result.get("single_branch_failure", {}).items():
            for mode, metric in modes.items():
                failure_rows.append({
                    "seed": seed, "condition": condition, "mode": mode,
                    "joint_score": metric["joint_score"],
                    "VBR_mAP_fg": metric["VBR"]["mAP_fg"],
                    "VBR_acc": metric["VBR"]["acc"],
                    "VBR_macro_f1": metric["VBR"]["macro_f1"],
                    "TCR_mAP": metric["TCR"]["mAP"],
                    "TCR_micro_f1": metric["TCR"]["micro_f1"],
                    "TCR_macro_f1": metric["TCR"]["macro_f1"],
                    "avg_modalities": metric.get("avg_modalities"),
                    "relative_branch_cost": metric.get("relative_branch_cost"),
                    "latency_ms": metric.get("model_latency_ms_per_clip"),
                    "post_route_failure_eligible_rate": metric.get(
                        "post_route_failure_eligible_rate"
                    ),
                })
        for evaluation in ("dense", "dynamic"):
            for task in ("VBR", "TCR"):
                for label, ap in result[evaluation][task].get("AP_by_class", {}).items():
                    class_rows.append({
                        "seed": seed, "evaluation": evaluation, "task": task,
                        "class": label, "AP": ap,
                    })

    if pattern_rows:
        patterns = pd.DataFrame(pattern_rows)
        patterns.to_csv(args.run_dir / "multi_seed_selection_patterns.csv", index=False)
        grouped_scalar_summary(patterns, ["selection_pattern"], ["rate"]).to_csv(
            args.run_dir / "multi_seed_selection_patterns_summary.csv", index=False
        )
    if subset_rows:
        subsets = pd.DataFrame(subset_rows).sort_values(["subset_id", "seed"])
        subsets.to_csv(args.run_dir / "multi_seed_31_subsets.csv", index=False)
        grouped_scalar_summary(
            subsets, ["subset_id", "modalities", "cardinality"],
            ["joint_score", "VBR_mAP_fg", "VBR_mAP_all", "VBR_acc",
             "VBR_macro_f1", "TCR_mAP", "TCR_micro_f1", "TCR_macro_f1"],
        ).to_csv(args.run_dir / "multi_seed_31_subsets_summary.csv", index=False)
    if failure_rows:
        failures = pd.DataFrame(failure_rows)
        failures.to_csv(args.run_dir / "multi_seed_single_branch_failure.csv", index=False)
        grouped_scalar_summary(
            failures, ["condition", "mode"],
            ["joint_score", "VBR_mAP_fg", "VBR_acc", "VBR_macro_f1",
             "TCR_mAP", "TCR_micro_f1", "TCR_macro_f1", "avg_modalities",
             "relative_branch_cost", "latency_ms", "post_route_failure_eligible_rate"],
        ).to_csv(args.run_dir / "multi_seed_single_branch_failure_summary.csv", index=False)
    if class_rows:
        classes = pd.DataFrame(class_rows)
        classes.to_csv(args.run_dir / "multi_seed_class_AP.csv", index=False)
        grouped_scalar_summary(
            classes, ["evaluation", "task", "class"], ["AP"]
        ).to_csv(args.run_dir / "multi_seed_class_AP_summary.csv", index=False)

    expected = expected_seed_artifacts(args)
    artifact_seeds = {}
    for seed in args.seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        files = []
        for relative in expected:
            path = seed_dir / relative
            files.append({
                "relative_path": relative,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            })
        artifact_seeds[str(seed)] = {
            "complete": all(item["exists"] for item in files),
            "missing": [item["relative_path"] for item in files if not item["exists"]],
            "files": files,
        }
    write_json(args.run_dir / "results_manifest_seed42-44.json", {
        "requested_seeds": args.seeds,
        "ignored_seed_directories": [
            path.name for path in sorted(args.run_dir.glob("seed_*"))
            if path.is_dir() and path.name not in {f"seed_{seed}" for seed in args.seeds}
        ],
        "seeds": artifact_seeds,
        "aggregate_outputs": [
            "multi_seed_results.csv", "multi_seed_summary.csv", "multi_seed_summary.json",
            "multi_seed_selection_patterns.csv", "multi_seed_selection_patterns_summary.csv",
            "multi_seed_31_subsets.csv", "multi_seed_31_subsets_summary.csv",
            "multi_seed_single_branch_failure.csv",
            "multi_seed_single_branch_failure_summary.csv",
            "multi_seed_class_AP.csv", "multi_seed_class_AP_summary.csv",
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    for key in ("dense_joint_score", "dynamic_joint_score", "joint_score_drop_pp",
                "dynamic_VBR_mAP_fg", "dynamic_TCR_mAP",
                "dynamic_avg_modalities", "dynamic_relative_branch_cost"):
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
        aggregate(args, require_all=not args.continue_on_seed_error)
        return
    if args.stage != "smoke":
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
    # A previous five-seed run may have left seed 45/46 status entries. Preserve
    # them as historical information, but never include them in the active run.
    status["requested_seeds"] = args.seeds
    status["split_seed"] = args.split_seed
    status["ignored_historical_seeds"] = sorted(
        int(seed) for seed in status.get("seeds", {})
        if int(seed) not in args.seeds
    )
    write_json(status_path, status)
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
