"""Robustness to corrupted but available inputs (Appendix A.8).

Stored Students, ACVEs and Student calibrations of seeds 42-44 (no retraining).
Test inputs are corrupted while every modality is still marked available, so
the system is not told about the degradation. Dense and Dynamic inference are
evaluated with the original evaluate_masks(). Corruptions are deterministic
per clip (same corruption for Dense, Dynamic and every seed).

Conditions
  clean           no corruption (must reproduce the paper)
  occlusion       a square patch covering 50% of the face and body crops
                  (same position in all 16 frames)
  noise           Gaussian noise, sigma 25 (of 255), on face, body and scene
  blur            Gaussian blur, kernel 11, sigma 3, on face, body and scene
  low_light       intensity scaled by 0.35 on face, body and scene
  face_miss       face detector failure: the face crop is replaced by a wrong
                  region (random half-size window of the body crop, resized)
  pose_jitter     pose-estimation error: keypoint x, y jittered with sigma
                  equal to 5% of the clip's keypoint extent
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
SOURCE = REPO / "main" / "train_multiseed.py"
RUNS = RUNS_ROOT / "AIDE_COVA_MTL_v6_3_MultiSeed5"
DATA_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
OUT = RUNS_ROOT / "analysis" / "corruption_robustness"
CONDITIONS = ["clean", "occlusion", "noise", "blur", "low_light", "face_miss", "pose_jitter"]


def load_module():
    spec = importlib.util.spec_from_file_location("cova", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cova"] = module
    spec.loader.exec_module(module)
    return module


cova = load_module()


def gaussian_kernel(size=11, sigma=3.0):
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


KERNEL = gaussian_kernel()


def blur(video):  # uint8 [3, T, H, W]
    x = video.float().permute(1, 0, 2, 3)  # T,3,H,W
    k = KERNEL.to(x)
    kx = k.view(1, 1, 1, -1).repeat(3, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(3, 1, 1, 1)
    pad = KERNEL.numel() // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kx, groups=3)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), ky, groups=3)
    return x.permute(1, 0, 2, 3).round().clamp(0, 255).to(torch.uint8)


def noise(video, gen):
    x = video.float() + torch.randn(video.shape, generator=gen) * 25.0
    return x.round().clamp(0, 255).to(torch.uint8)


def dark(video):
    return (video.float() * 0.35).round().clamp(0, 255).to(torch.uint8)


def occlude(video, gen):
    _, _, h, w = video.shape
    side = int(round((0.5 * h * w) ** 0.5))
    y = int(torch.randint(0, h - side + 1, (1,), generator=gen))
    x = int(torch.randint(0, w - side + 1, (1,), generator=gen))
    out = video.clone()
    out[:, :, y:y + side, x:x + side] = 0
    return out


def wrong_region(body, gen):
    _, t, h, w = body.shape
    side = h // 2
    y = int(torch.randint(0, h - side + 1, (1,), generator=gen))
    x = int(torch.randint(0, w - side + 1, (1,), generator=gen))
    patch = body[:, :, y:y + side, x:x + side].float().permute(1, 0, 2, 3)
    patch = F.interpolate(patch, (h, w), mode="bilinear", align_corners=False)
    return patch.permute(1, 0, 2, 3).round().clamp(0, 255).to(torch.uint8)


def jitter(kp, gen):  # [3, T, K, 1]: x, y, score
    out = kp.clone()
    for c in (0, 1):
        coord = kp[c]
        extent = float(coord.max() - coord.min()) if coord.numel() else 0.0
        out[c] = coord + torch.randn(coord.shape, generator=gen) * 0.05 * max(extent, 1.0)
    return out


class CorruptedDataset(cova.CarDataset):
    condition = "clean"

    def __getitem__(self, index):
        kp, context, body, face, *labels = super().__getitem__(index)
        cond = self.condition
        if cond == "clean":
            return (kp, context, body, face, *labels)
        gen = torch.Generator().manual_seed(zlib.crc32(f"{cond}-{index}".encode()))
        if cond == "occlusion":
            face, body = occlude(face, gen), occlude(body, gen)
        elif cond == "noise":
            face, body, context = noise(face, gen), noise(body, gen), noise(context, gen)
        elif cond == "blur":
            face, body, context = blur(face), blur(body), blur(context)
        elif cond == "low_light":
            face, body, context = dark(face), dark(body), dark(context)
        elif cond == "face_miss":
            face = wrong_region(body, gen)
        elif cond == "pose_jitter":
            kp = jitter(kp, gen)
        else:
            raise ValueError(cond)
        return (kp, context, body, face, *labels)


def seed_args(seed):
    return cova.finalize_args(cova.build_parser().parse_args([
        "--no-multi-seed", "--seed", str(seed),
        "--data-root", str(DATA_ROOT),
        "--run-dir", str(RUNS / f"seed_{seed}"),
        "--no-pretrained", "--num-workers", "6",
    ]))


def keep(metrics):
    out = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    for key in ("acc", "f1", "selection_rate"):
        if key in metrics:
            out[key] = metrics[key]
    return out


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [42, 43, 44]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "corruption.json"
    results = json.load(open(out_path)) if out_path.exists() else {}
    device = torch.device("cuda")
    for seed in seeds:
        args = seed_args(seed)
        paper = json.load(open(args.run_dir / "final_test.json"))
        model = cova.model_from_args(args, device)
        cova.load_checkpoint(args.sparse_ckpt, model)
        router = cova.load_router_from_args(args, device)
        dataset = CorruptedDataset(
            args.split_dir / "test.csv", t_target=args.t_target, stride=args.stride,
            face_pad_top=args.face_pad_top, face_pad_bottom=args.face_pad_bottom,
            face_pad_left=args.face_pad_left, face_pad_right=args.face_pad_right,
        )
        results.setdefault(str(seed), {})
        for cond in CONDITIONS:
            if cond in results[str(seed)]:
                continue
            dataset.condition = cond
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=6, pin_memory=True, drop_last=False,
            )
            dense = cova.evaluate_masks(model, loader, device, False, None, args)
            dyn = cova.evaluate_masks(model, loader, device, True, router, args)
            results[str(seed)][cond] = {"dense": keep(dense), "dynamic": keep(dyn)}
            print(f"seed {seed} {cond:11s} dense {dense['macc']*100:.2f} dynamic {dyn['macc']*100:.2f} "
                  f"n {dyn['avg_modalities']:.3f} cost {dyn['relative_branch_cost']*100:.2f}"
                  + (f" | paper dense {paper['dense']['macc']*100:.2f} dynamic {paper['dynamic']['macc']*100:.2f}"
                     if cond == "clean" else ""), flush=True)
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
        del model, router
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
