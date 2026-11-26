"""FLOPs, parameter count, and peak GPU memory of Dense and Dynamic inference (Section 4.4).

Stored seed-42 Student, ACVE, and calibration (no retraining). Every AIDE test clip is processed with batch
size 1. For each clip, the forward FLOPs (torch.utils.flop_counter) and the peak allocated GPU memory above
the resident weights are recorded for Dense inference (forward_dense) and Dynamic inference
(route_from_raw: scout, ACVE, SSR, selected encoders, SCML), both under autocast.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
AIDE_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
SOURCE = REPO / "main" / "train_multiseed.py"
RUN = RUNS_ROOT / "AIDE_COVA_MTL_v6_3_MultiSeed5" / "seed_42"
OUT = RUNS_ROOT / "analysis" / "compute_profile"

spec = importlib.util.spec_from_file_location("cova", SOURCE)
cova = importlib.util.module_from_spec(spec)
sys.modules["cova"] = cova
spec.loader.exec_module(cova)


def flops_of(fn):
    counter = FlopCounterMode(display=False)
    with counter:
        out = fn()
    return counter.get_total_flops(), out


def peak_of(fn, base):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20, out


def main():
    device = torch.device("cuda")
    args = cova.finalize_args(cova.build_parser().parse_args([
        "--no-multi-seed", "--run-dir", str(RUN), "--no-pretrained",
        "--data-root", str(AIDE_ROOT),
    ]))
    model = cova.model_from_args(args, device)
    cova.load_checkpoint(args.sparse_ckpt, model)
    model.eval()
    router = cova.load_router_from_args(args, device)
    router.eval()
    model.requires_grad_(False)
    router.requires_grad_(False)
    params = {"student": sum(p.numel() for p in model.parameters()),
              "acve": sum(p.numel() for p in router.parameters())}
    ds = cova.CarDataset(
        args.split_dir / "test.csv", t_target=args.t_target, stride=args.stride,
        face_pad_top=args.face_pad_top, face_pad_bottom=args.face_pad_bottom,
        face_pad_left=args.face_pad_left, face_pad_right=args.face_pad_right,
    )
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            data = cova.move_batch(default_collate([ds[i]]), device)
            kp, context, body, face = data[:4]
            dense = lambda: model.forward_dense(kp, context, body, face)
            dyn = lambda: cova.route_from_raw(model, router, kp, context, body, face)
            with torch.autocast("cuda"):
                if i == 0:
                    dense(); dyn()
                base = torch.cuda.memory_allocated()
                d_mem, _ = peak_of(dense, base)
                y_mem, res = peak_of(dyn, base)
                d_flops, _ = flops_of(dense)
                y_flops, _ = flops_of(dyn)
            rows.append({"dense_gflops": d_flops / 1e9, "dynamic_gflops": y_flops / 1e9,
                         "dense_peak_mib": d_mem, "dynamic_peak_mib": y_mem,
                         "modalities": float(res["mask"].float().sum())})
            if i % 100 == 0:
                print(i, rows[-1], flush=True)
    stats = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1)),
                 "min": float(np.min([r[k] for r in rows])), "max": float(np.max([r[k] for r in rows]))}
             for k in rows[0]}
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "clips": len(rows),
              "params": params, "weights_mib": base / 2**20, "stats": stats,
              "flops_reduction_pct": 100 * (1 - stats["dynamic_gflops"]["mean"] / stats["dense_gflops"]["mean"])}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT / "compute_seed42.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
