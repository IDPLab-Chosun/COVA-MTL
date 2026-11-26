"""Fixed subset selected on validation under the branch-cost budget of COVA-MTL (Table A11).

For each seed, the fixed subset with the highest validation mAcc among the subsets whose validation branch
cost does not exceed that of COVA-MTL is frozen and evaluated once on the test split.
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
spec = importlib.util.spec_from_file_location("routing", HERE / "routing_analysis.py")
routing = importlib.util.module_from_spec(spec)
sys.modules["routing"] = routing
spec.loader.exec_module(routing)

SEEDS = (42, 43, 44)
OUT = RUNS_ROOT / "analysis" / "fixed_subset_selection"


def choose(val_points, test_points, budget):
    """Index with the highest validation mAcc among points with validation cost <= budget."""
    usable = [i for i, p in enumerate(val_points) if p["relative_cost"] <= budget + 1e-9]
    best = max(usable, key=lambda i: val_points[i]["macc"])
    return {"val": val_points[best], "test": test_points[best]}


def run_seed(cova, seed, device):
    run = routing.RUNS / f"seed_{seed}"
    val = torch.load(routing.STUDENT_CACHE / f"seed_{seed}/val.pt", weights_only=False)
    test = torch.load(routing.STUDENT_CACHE / f"seed_{seed}/test.pt", weights_only=False)
    router = cova.AvailabilityConditionedSufficientRouter(128, 256).to(device)
    cova.load_checkpoint(run / "router/best.pt", router)
    router.eval()
    with open(run / "router/calibration_student.json", encoding="utf-8") as handle:
        q = torch.tensor(json.load(handle)["q_by_available_and_selected_cardinality"])

    def full(cache):
        return torch.ones(cache["labels"].size(0), 5, dtype=torch.bool)

    def cova_route(cache):
        available = full(cache)
        rel_mean, rel_scale, ref, valid, _ = routing.router_relative(router, cache["scout_tokens"].float(), available, device)
        return routing.select(rel_mean, rel_scale, ref, valid, routing.q_cells(q, available), "mean", "card")

    cova_val = routing.summarize(routing.evaluate_route(cova_route(val), val, full(val)))
    cova_test = routing.summarize(routing.evaluate_route(cova_route(test), test, full(test)))
    budget = cova_val["relative_cost"]
    out = {"seed": seed, "cova": {"val": cova_val, "test": cova_test}}

    # Fixed subsets.
    fixed_val = [routing.summarize(routing.evaluate_route(torch.full((val["labels"].size(0),), c), val, full(val))) for c in range(31)]
    fixed_test = [routing.summarize(routing.evaluate_route(torch.full((test["labels"].size(0),), c), test, full(test))) for c in range(31)]
    pick = choose(fixed_val, fixed_test, budget)
    index = fixed_val.index(pick["val"])
    pick["subset"] = "+".join(m for i, m in enumerate(routing.MODALITIES) if routing.MASKS[index, i])
    out["fixed_subset"] = pick

    return out


def main():
    cova = routing.load_cova()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {s: run_seed(cova, s, device) for s in SEEDS}
    json.dump(results, open(OUT / "fixed_subset_selection.json", "w"), indent=2)
    methods = ["cova", "fixed_subset"]
    lines = ["| Method | Val cost (%) | Test cost (%) | Test mAcc | Test mF1 | Test viol. >0 (%) |", "|---|---|---|---|---|---|"]
    for m in methods:
        rows = [results[s][m] for s in SEEDS]
        cell = lambda key, part: (
            "–" if part not in rows[0] else
            f"{100*np.mean([r[part][key] for r in rows]):.2f} ± {100*np.std([r[part][key] for r in rows], ddof=1):.2f}")
        lines.append(f"| {m} | {cell('relative_cost', 'val')} | {cell('relative_cost', 'test')} | "
                     f"{cell('macc', 'test')} | {cell('mf1', 'test')} | {cell('violation_rate_gt0', 'test')} |")
    lines.append("")
    lines.append("Per seed test mAcc: " + "; ".join(
        f"{s}: " + ", ".join(f"{m}={100*results[s][m]['test']['macc']:.2f}" for m in methods) for s in SEEDS))
    lines.append("Fixed subsets: " + ", ".join(f"{s}: {results[s]['fixed_subset']['subset']}" for s in SEEDS))
    text = "\n".join(lines)
    (OUT / "fixed_subset_selection.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
