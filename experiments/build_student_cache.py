"""Cache the outputs of the deployed Student on all 31 non-empty modality subsets.

For each seed, the final Student (sparse/bestmacc.pt) encodes every validation and test clip once and is
evaluated on all 31 subsets. The cache stores the preview (scout) tokens, normalized task losses,
predictions, and labels in the format of the Teacher oracle cache, so that routing decisions can be
replayed offline by the other programs in this folder.

Check: replaying SSR on the cache reproduces the Dense and Dynamic test mAcc in final_test.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
AIDE_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
SOURCE = REPO / "main" / "train_multiseed.py"
RUNS = RUNS_ROOT / "AIDE_COVA_MTL_v6_3_MultiSeed5"
OUT = RUNS_ROOT / "analysis" / "student_cache"
DATA_ROOT = AIDE_ROOT


def load_cova():
    spec = importlib.util.spec_from_file_location("cova", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cova"] = module
    spec.loader.exec_module(module)
    return module


def seed_args(cova, seed: int):
    args = cova.finalize_args(cova.build_parser().parse_args([
        "--no-multi-seed", "--seed", str(seed),
        "--data-root", str(DATA_ROOT),
        "--run-dir", str(RUNS / f"seed_{seed}"),
        "--no-pretrained", "--num-workers", "6",
    ]))
    return args


@torch.no_grad()
def build_cache(cova, split: str, loader, model, device, out_dir: Path, source: Path) -> Path:
    """Same content as the Teacher oracle cache, plus per-subset logits."""
    model.eval()
    masks = cova.mask_table(device)
    collected = {key: [] for key in ("scout_tokens", "subset_loss", "predictions", "logits", "labels")}
    for batch in cova.tqdm(loader, desc=f"student cache {split}", leave=False):
        data = cova.move_batch(batch, device)
        kp, context, body, face = data[:4]
        labels = cova.labels_tuple(data)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            scout_tokens, tokens = model.encode_all(kp, context, body, face)
            losses = torch.full((kp.size(0), 32, 4), torch.nan, device=device)
            predictions = torch.full((kp.size(0), 32, 4), -1, dtype=torch.long, device=device)
            logits_all = torch.full((kp.size(0), 32, sum(cova.TASK_CLASSES)), torch.nan, device=device)
            for subset_id in range(1, 32):
                current = masks[subset_id][None].expand(kp.size(0), -1)
                logits, _, _ = model.predict(scout_tokens, tokens, current)
                losses[:, subset_id] = cova.normalized_per_task_ce(logits, labels)
                predictions[:, subset_id] = torch.stack([logit.argmax(1) for logit in logits], 1)
                logits_all[:, subset_id] = torch.cat([logit.float() for logit in logits], 1)
        collected["scout_tokens"].append(scout_tokens.half().cpu())
        collected["subset_loss"].append(losses.float().cpu())
        collected["predictions"].append(predictions.cpu())
        collected["logits"].append(logits_all.float().cpu())
        collected["labels"].append(torch.stack(labels, 1).cpu())
    output = {key: torch.cat(parts, 0) for key, parts in collected.items()}
    output.update({
        "mask_table": masks.cpu(), "modality_names": cova.MODALITY_NAMES,
        "task_names": cova.TASK_NAMES, "task_classes": cova.TASK_CLASSES,
        "source_checkpoint": str(source),
        "target_semantics": "Student normalized per-task loss and logits for each of 31 non-empty subsets",
    })
    path = out_dir / f"{split}.pt"
    torch.save(output, path)
    print(f"student cache saved: {path} | samples={output['scout_tokens'].size(0)}")
    return path


@torch.no_grad()
def replay_original_rule(cova, args, cache_path: Path, device) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    router = cova.load_router_from_args(args, device)
    scout = cache["scout_tokens"].float().to(device)
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        route = cova.select_self_referenced_subset(router, scout)
    subset_id = route["subset_id"].cpu()
    rows = torch.arange(subset_id.numel())
    dynamic_pred = cache["predictions"][rows, subset_id]
    dense_pred = cache["predictions"][:, 31]
    labels = cache["labels"]
    return {
        "dense": cova.metrics_from_label_prediction(labels, dense_pred),
        "dynamic": cova.metrics_from_label_prediction(labels, dynamic_pred),
        "avg_modalities": float(route["mask"].float().sum(1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--no-verify", action="store_true")
    cli = parser.parse_args()
    cova = load_cova()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    report = {}
    for seed in [int(s) for s in cli.seeds.split(",")]:
        cova.seed_everything(seed)
        args = seed_args(cova, seed)
        out_dir = OUT / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        loaders = cova.build_loaders(args, cache_order=True)
        student = cova.model_from_args(args, device)
        cova.load_checkpoint(args.sparse_ckpt, student)
        for split in cli.splits.split(","):
            build_cache(cova, split, loaders[split], student, device, out_dir, args.sparse_ckpt)
        cova.shutdown_loaders(loaders)

        if cli.no_verify:
            del student
            torch.cuda.empty_cache()
            continue
        replay = replay_original_rule(cova, args, out_dir / "test.pt", device)
        with open(args.run_dir / "final_test.json", encoding="utf-8") as handle:
            paper = json.load(handle)
        report[seed] = {
            "replay_dense_macc": replay["dense"]["macc"],
            "paper_dense_macc": paper["dense"]["macc"],
            "replay_dynamic_macc": replay["dynamic"]["macc"],
            "paper_dynamic_macc": paper["dynamic"]["macc"],
            "replay_avg_modalities": replay["avg_modalities"],
            "paper_avg_modalities": paper["dynamic"]["avg_modalities"],
        }
        print(json.dumps({seed: report[seed]}, indent=2))
        del student
        torch.cuda.empty_cache()
    if cli.no_verify:
        return
    with open(OUT / "verify.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("saved", OUT / "verify.json")


if __name__ == "__main__":
    main()
