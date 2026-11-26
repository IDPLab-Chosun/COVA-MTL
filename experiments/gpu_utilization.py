"""GPU utilization of Dense and Dynamic inference (Section 4.4).

Stored seed-42 Student, ACVE, and calibration (no retraining). The first N AIDE test clips are decoded once
and kept on the GPU, so that the measured phases contain model execution only. Each phase runs the clips
for several passes while nvidia-smi samples utilization.gpu every 50 ms; the mean over the phase (first and
last 0.5 s dropped) is reported with the mean per-clip latency. Batch sizes 1 and 8.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
OUT = RUNS_ROOT / "analysis" / "gpu_utilization"
N_CLIPS, PASSES = 96, 6

spec = importlib.util.spec_from_file_location("cova", SOURCE)
cova = importlib.util.module_from_spec(spec)
sys.modules["cova"] = cova
spec.loader.exec_module(cova)


def sample_util(fn):
    log = OUT / "smi.tmp"
    proc = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu", "--format=csv,noheader,nounits",
                             "-lms", "50", "-f", str(log)])
    time.sleep(1.0)
    torch.cuda.synchronize()
    t0 = time.time()
    n = fn()
    torch.cuda.synchronize()
    t1 = time.time()
    time.sleep(0.3)
    proc.terminate()
    proc.wait()
    vals = []
    for line in log.read_text().strip().splitlines():
        ts, u = [x.strip() for x in line.split(",")]
        t = time.mktime(time.strptime(ts.split(".")[0], "%Y/%m/%d %H:%M:%S")) + float("0." + ts.split(".")[1])
        if t0 + 0.5 <= t <= t1 - 0.5:
            vals.append(float(u))
    return {"util_mean": float(np.mean(vals)), "util_std": float(np.std(vals, ddof=1)), "samples": len(vals),
            "seconds": t1 - t0, "ms_per_clip": 1000 * (t1 - t0) / n}


def main():
    device = torch.device("cuda")
    args = cova.finalize_args(cova.build_parser().parse_args([
        "--no-multi-seed", "--run-dir", str(RUN), "--no-pretrained",
        "--data-root", str(AIDE_ROOT),
    ]))
    model = cova.model_from_args(args, device)
    cova.load_checkpoint(args.sparse_ckpt, model)
    model.eval().requires_grad_(False)
    router = cova.load_router_from_args(args, device)
    router.eval().requires_grad_(False)
    ds = cova.CarDataset(
        args.split_dir / "test.csv", t_target=args.t_target, stride=args.stride,
        face_pad_top=args.face_pad_top, face_pad_bottom=args.face_pad_bottom,
        face_pad_left=args.face_pad_left, face_pad_right=args.face_pad_right,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [ds[i] for i in range(N_CLIPS)]
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "clips": N_CLIPS, "passes": PASSES}
    for bs in (1, 8):
        batches = [cova.move_batch(default_collate(clips[i:i + bs]), device) for i in range(0, N_CLIPS, bs)]

        def run(kind):
            def go():
                with torch.no_grad(), torch.autocast("cuda"):
                    for _ in range(PASSES):
                        for b in batches:
                            kp, context, body, face = b[:4]
                            if kind == "dense":
                                model.forward_dense(kp, context, body, face)
                            else:
                                cova.route_from_raw(model, router, kp, context, body, face)
                return PASSES * N_CLIPS
            return go

        for kind in ("dense", "dynamic"):
            try:
                run(kind)()  # warm-up pass set
                report[f"bs{bs}_{kind}"] = sample_util(run(kind))
            except Exception as e:  # e.g. routing written for batch 1 only
                report[f"bs{bs}_{kind}"] = {"error": repr(e)}
            print(bs, kind, report[f"bs{bs}_{kind}"], flush=True)
        del batches
        torch.cuda.empty_cache()
    json.dump(report, open(OUT / "gpu_util_seed42.json", "w"), indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
