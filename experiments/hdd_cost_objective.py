"""HDD: minimum-cardinality versus branch-cost objective in SSR (Section 4.9).

Loads the stored HDD Student, ACVE, and Student calibration of seeds 42-44 (no retraining) and evaluates
on the test split Dense inference, Dynamic inference with SSR, and Dynamic inference whose primary SSR key
is the branch-cost sum (same admissibility gate and tie-break). The evaluation function and latency
protocol are those of cross_dataset/hdd.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
SOURCE = REPO / "cross_dataset" / "hdd.py"
BASE_RUN = RUNS_ROOT / "HDD_COVA_v6_3_MultiSeed5"
OUT = RUNS_ROOT / "analysis" / "hdd_cost_objective"
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 43, 44]


def load_module():
    spec = importlib.util.spec_from_file_location("cova_hdd", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cova_hdd"] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()
ORIGINAL_SELECT = mod.select_subset


def make_cost_select(costs_list):
    @torch.no_grad()
    def select_subset_cost(router, scout, available=None):
        M, NUM_SUBSETS = mod.M, mod.NUM_SUBSETS
        batch = scout.size(0)
        if available is None:
            available = torch.ones(batch, M, dtype=torch.bool, device=scout.device)
        available = available.bool()
        routed_scout = scout * available[:, :, None].to(scout.dtype)
        task_mean, task_scale = router(routed_scout, available)
        valid = ~(router.candidate_masks[None] & (~available)[:, None]).any(-1)
        relative_mean, relative_scale, upper, mean, scale = mod.calibrated_relative(
            router, task_mean, task_scale, available
        )
        finite = torch.isfinite(relative_mean) & torch.isfinite(upper)
        admissible = valid & finite & (relative_mean <= 0)
        costs = torch.tensor(costs_list, dtype=torch.float32, device=scout.device)
        cand_cost = (router.candidate_masks.float() * costs[None]).sum(1)
        prim = cand_cost[None].expand(batch, -1)
        minimum = prim.masked_fill(~admissible, float("inf")).min(1).values
        eligible = admissible & (prim <= minimum[:, None] + 1e-6)
        best = upper.masked_fill(~eligible, torch.inf).argmin(1)
        fallback = mod.masks_to_subset_ids(available) - 1
        has = admissible.any(1)
        best = torch.where(has, best, fallback)
        rows = torch.arange(batch, device=scout.device)
        return {
            "mask": router.candidate_masks[best], "subset_id": best + 1,
            "alternative_mask": router.candidate_masks[best],
            "alternative_subset_id": best + 1,
            "has_alternative": torch.zeros_like(has),
            "available": available, "fallback": ~has,
            "selected_full_available": best.eq(fallback),
            "confidence_supported": upper[rows, best] <= 0,
            "predicted_relative_mean_regret": relative_mean[rows, best],
            "predicted_relative_upper_regret": upper[rows, best],
            "predicted_task_regret": task_mean.float()[rows, best],
            "predicted_task_scale": task_scale.float()[rows, best],
        }
    return select_subset_cost


def summary(metrics):
    keep = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    for key in ("VBR", "TCR"):
        if key in metrics:
            keep[key] = {k: v for k, v in metrics[key].items() if not isinstance(v, (dict, list))}
    for key in ("selection_rate", "selection_pattern_rate"):
        if key in metrics:
            keep[key] = metrics[key]
    return keep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    parser = mod.build_parser()
    base = mod.finalize_args(parser.parse_args([
        "--data-root", os.environ.get("HDD_ROOT", str(REPO / "data" / "hdd_data")), "--run-dir", str(BASE_RUN), "--stage", "eval",
    ]))
    results = {}
    for seed in SEEDS:
        args = mod.make_seed_args(base, seed)
        args.split_dir = Path(json.load(open(args.run_dir / "config.json"))["split_dir"])
        config = json.load(open(args.run_dir / "config.json"))
        diff = {k: (config[k], str(getattr(args, k))) for k in config
                if hasattr(args, k) and str(config[k]) != str(getattr(args, k))
                and not isinstance(config[k], (list, dict))}
        print(f"seed {seed} config differences:", diff, flush=True)
        paper = json.load(open(args.run_dir / "final_test.json"))
        loaders = mod.build_loaders(args)
        model = mod.model_from_args(args, device)
        mod.load_checkpoint(args.sparse_ckpt, model)
        router = mod.load_router(args, device, prefer_student=True)
        mod.select_subset = ORIGINAL_SELECT
        dense = mod.evaluate(model, loaders["test"], device, args)
        card = mod.evaluate(model, loaders["test"], device, args, dynamic=True, router=router)
        mod.select_subset = make_cost_select(args.modality_costs)
        cost = mod.evaluate(model, loaders["test"], device, args, dynamic=True, router=router)
        cost_val = mod.evaluate(model, loaders["val"], device, args, dynamic=True, router=router)
        mod.select_subset = ORIGINAL_SELECT
        card_val = mod.evaluate(model, loaders["val"], device, args, dynamic=True, router=router)
        results[seed] = {
            "paper_dense_joint": paper["dense"]["joint_score"],
            "paper_dynamic_joint": paper["dynamic"]["joint_score"],
            "paper_dynamic_rel_cost": paper["dynamic"]["relative_branch_cost"],
            "dense": summary(dense), "dynamic_card": summary(card), "dynamic_cost": summary(cost),
            "val_dynamic_card": summary(card_val), "val_dynamic_cost": summary(cost_val),
            "modality_costs": list(args.modality_costs),
        }
        r = results[seed]
        print(f"seed {seed}: paper dense {r['paper_dense_joint']:.4f} dyn {r['paper_dynamic_joint']:.4f} | "
              f"replay dense {dense['joint_score']:.4f} card {card['joint_score']:.4f} "
              f"(cost {card['relative_branch_cost']:.4f}, n {card['avg_modalities']:.3f}, "
              f"lat {dense['model_latency_ms_per_clip']:.3f}/{card['model_latency_ms_per_clip']:.3f}) | "
              f"costobj {cost['joint_score']:.4f} (cost {cost['relative_branch_cost']:.4f}, "
              f"n {cost['avg_modalities']:.3f}, lat {cost['model_latency_ms_per_clip']:.3f}, "
              f"rgb {cost['selection_rate']['rgb']:.4f} vs {card['selection_rate']['rgb']:.4f})", flush=True)
        with open(OUT / "hdd_cost_routing.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        mod.shutdown_loaders(loaders)
        del model, router
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
