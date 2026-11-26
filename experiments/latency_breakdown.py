"""Component-wise latency of Dense and Dynamic inference (Table 12).

Measures, with CUDA events after warm-up, the time of each inference component (preview scout, ACVE and
SSR, each branch encoder, SCML prediction) and of complete Dense and Dynamic forward passes for
representative subsets. Inputs are random tensors of the AIDE shapes, because the branch time does not
depend on the content. Batch sizes 1 and 8.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
AIDE_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
SOURCE = REPO / "main" / "train_multiseed.py"
RUN = RUNS_ROOT / "AIDE_COVA_MTL_v6_3_MultiSeed5" / "seed_42"
OUT = RUNS_ROOT / "analysis" / "latency_breakdown"
MODALITIES = ("face", "body", "scene", "posture", "gesture")
SUBSETS = {
    "all5": [1, 1, 1, 1, 1],
    "F+B+S+P": [1, 1, 1, 1, 0],
    "F+B+S": [1, 1, 1, 0, 0],
    "B+S+P+G": [0, 1, 1, 1, 1],
    "B+S+P": [0, 1, 1, 1, 0],
    "S+P+G": [0, 0, 1, 1, 1],
    "S+P": [0, 0, 1, 1, 0],
    "S": [0, 0, 1, 0, 0],
}
WARMUP, REPEAT = 10, 50


def load_cova():
    spec = importlib.util.spec_from_file_location("cova", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cova"] = module
    spec.loader.exec_module(module)
    return module


def timed(fn) -> dict:
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(REPEAT):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    values = np.array(times)
    return {"mean_ms": float(values.mean()), "std_ms": float(values.std(ddof=1)),
            "median_ms": float(np.median(values))}


def main() -> None:
    cova = load_cova()
    device = torch.device("cuda")
    args = cova.finalize_args(cova.build_parser().parse_args([
        "--no-multi-seed", "--run-dir", str(RUN), "--no-pretrained",
        "--data-root", str(AIDE_ROOT),
    ]))
    model = cova.model_from_args(args, device)
    cova.load_checkpoint(args.sparse_ckpt, model)
    model.eval()
    router = cova.load_router_from_args(args, device)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "warmup": WARMUP, "repeat": REPEAT, "batches": {}}

    for batch in (1, 8):
        kp = torch.randn(batch, 3, 16, 136, 1, device=device)
        context = torch.rand(batch, 3, 16, 224, 224, device=device)
        body = torch.rand(batch, 3, 16, 224, 224, device=device)
        face = torch.rand(batch, 3, 16, 224, 224, device=device)
        with torch.no_grad(), torch.autocast("cuda"):
            scout = model.scout(kp, context, body, face)
            full_mask = torch.ones(batch, 5, dtype=torch.bool, device=device)
            full_tokens = model.encode_selected(full_mask, kp, context, body, face)
        rows = {}
        rows["scout_preview"] = timed(lambda: model.scout(kp, context, body, face))
        rows["acve_ssr"] = timed(lambda: cova.select_self_referenced_subset(router, scout))
        for m, name in enumerate(MODALITIES):
            rows[f"branch_{name}"] = timed(lambda m=m: model.encode_modality(m, kp, context, body, face))
        rows["scml_predict_all5"] = timed(lambda: model.predict(scout, full_tokens, full_mask))
        rows["dense_forward"] = timed(lambda: model.forward_dense(kp, context, body, face))

        subsets = {}
        for name, pattern in SUBSETS.items():
            mask = torch.tensor(pattern, dtype=torch.bool, device=device)[None].expand(batch, -1)

            def dynamic_fixed_route(mask=mask):
                s = model.scout(kp, context, body, face)
                cova.select_self_referenced_subset(router, s)
                tokens = model.encode_selected(mask, kp, context, body, face)
                return model.predict(s, tokens, mask)

            def static_subset(mask=mask):
                s = model.scout(kp, context, body, face)
                tokens = model.encode_selected(mask, kp, context, body, face)
                return model.predict(s, tokens, mask)

            subsets[name] = {
                "with_router": timed(dynamic_fixed_route),
                "without_router": timed(static_subset),
            }
        report["batches"][str(batch)] = {"components": rows, "subsets": subsets}
        print(f"batch {batch}")
        for key, value in rows.items():
            print(f"  {key:22s} {value['mean_ms']:8.3f} ms")
        for key, value in subsets.items():
            print(f"  {key:10s} router {value['with_router']['mean_ms']:8.3f} | "
                  f"no router {value['without_router']['mean_ms']:8.3f} ms")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "latency_seed42.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("saved", OUT / "latency_seed42.json")


if __name__ == "__main__":
    main()
