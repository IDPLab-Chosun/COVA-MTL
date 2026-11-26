"""End-to-end per-clip latency of Dense and Dynamic inference (Section 4.9).

Stored seed-42 Student, ACVE and Student calibration (no retraining). Test
clips are processed one at a time (batch 1, no worker processes), as in an
in-vehicle pipeline. For every clip the following stages are timed:
  read       JPEG reading and decoding of the 16 selected frames of the four
             views (cv2.imread)
  crop       annotation JSON, cropping the annotated body and face boxes, resizing, colour
             conversion, scene mosaic and tensor assembly
  h2d        host-to-device transfer and normalisation (move_batch)
  model      Dense: forward_dense; Dynamic: route_from_raw (scout, ACVE, SSR,
             selected encoders, SCML), with autocast as in evaluate_masks
Body/face boxes and keypoints are dataset annotations, so no detector or pose
estimator runs. A first untimed pass warms the page cache; the second pass is
reported.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
AIDE_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
SOURCE = REPO / "main" / "train_multiseed.py"
RUN = RUNS_ROOT / "AIDE_COVA_MTL_v6_3_MultiSeed5" / "seed_42"
OUT = RUNS_ROOT / "analysis" / "end_to_end_latency"
WARMUP = 10


def load_cova():
    spec = importlib.util.spec_from_file_location("cova", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cova"] = module
    spec.loader.exec_module(module)
    return module


cova = load_cova()
READ_TIME = [0.0]
_imread = cova.cv2.imread


def timed_imread(*a, **k):
    t = time.perf_counter()
    out = _imread(*a, **k)
    READ_TIME[0] += time.perf_counter() - t
    return out


class TimedCV2:
    def __getattr__(self, name):
        return timed_imread if name == "imread" else getattr(cova_cv2, name)


cova_cv2 = cova.cv2
cova.cv2 = TimedCV2()


def sync():
    torch.cuda.synchronize()


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
    ds = cova.CarDataset(
        args.split_dir / "test.csv", t_target=args.t_target, stride=args.stride,
        face_pad_top=args.face_pad_top, face_pad_bottom=args.face_pad_bottom,
        face_pad_left=args.face_pad_left, face_pad_right=args.face_pad_right,
    )
    n = len(ds)
    for i in range(n):  # untimed pass: page cache
        ds[i]
    rows = []
    n_mod = []
    with torch.no_grad():
        for i in range(n):
            READ_TIME[0] = 0.0
            t0 = time.perf_counter()
            sample = ds[i]
            t1 = time.perf_counter()
            read = READ_TIME[0]
            batch = default_collate([sample])
            sync()
            t2 = time.perf_counter()
            data = cova.move_batch(batch, device)
            sync()
            t3 = time.perf_counter()
            kp, context, body, face = data[:4]
            with torch.autocast("cuda"):
                model.forward_dense(kp, context, body, face)
            sync()
            t4 = time.perf_counter()
            with torch.autocast("cuda"):
                result = cova.route_from_raw(model, router, kp, context, body, face)
            sync()
            t5 = time.perf_counter()
            n_mod.append(float(result["mask"].float().sum()))
            if i >= WARMUP:
                rows.append({"read": read * 1e3, "crop": (t1 - t0 - read) * 1e3 + (t2 - t1) * 1e3,
                             "h2d": (t3 - t2) * 1e3, "dense_model": (t4 - t3) * 1e3,
                             "dynamic_model": (t5 - t4) * 1e3})
    keys = rows[0].keys()
    stats = {k: {"mean_ms": float(np.mean([r[k] for r in rows])),
                 "std_ms": float(np.std([r[k] for r in rows], ddof=1)),
                 "median_ms": float(np.median([r[k] for r in rows]))} for k in keys}
    pre = [r["read"] + r["crop"] + r["h2d"] for r in rows]
    dense = [p + r["dense_model"] for p, r in zip(pre, rows)]
    dyn = [p + r["dynamic_model"] for p, r in zip(pre, rows)]
    for name, v in (("preprocessing", pre), ("dense_e2e", dense), ("dynamic_e2e", dyn)):
        stats[name] = {"mean_ms": float(np.mean(v)), "std_ms": float(np.std(v, ddof=1)),
                       "median_ms": float(np.median(v))}
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "clips": len(rows), "warmup": WARMUP, "avg_modalities": float(np.mean(n_mod)),
              "stats": stats,
              "model_reduction_pct": 100 * (1 - stats["dynamic_model"]["mean_ms"] / stats["dense_model"]["mean_ms"]),
              "e2e_reduction_pct": 100 * (1 - stats["dynamic_e2e"]["mean_ms"] / stats["dense_e2e"]["mean_ms"])}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "e2e_seed42.json", "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2)
    for k, v in stats.items():
        print(f"{k:14s} {v['mean_ms']:9.3f} ± {v['std_ms']:7.3f} ms (median {v['median_ms']:.3f})")
    print(f"avg modalities {report['avg_modalities']:.3f} | model reduction {report['model_reduction_pct']:.2f}% "
          f"| e2e reduction {report['e2e_reduction_pct']:.2f}%")


if __name__ == "__main__":
    main()
