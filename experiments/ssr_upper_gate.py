"""SSR with calibrated upper-bound admissibility gates (Section 3.5).

Compares the SSR rule (mean admissibility, minimum cardinality, calibrated tie-break) with gates that admit
a candidate only if its calibrated upper bound is not positive, calibrated at 80% and 90% coverage.
ACVE, calibration, and Student are the checkpoints of each seed; only the selection rule changes.
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
OUT = RUNS_ROOT / "analysis" / "ssr_upper_gate"
INF = float("inf")


def route(variant, rel_mean, rel_scale, ref, valid, cells, raw_mean):
    if variant == "full_ssr":
        return routing.select(rel_mean, rel_scale, ref, valid, cells, "mean", "card")
    if variant.startswith("upper_gate_"):
        return routing.select(rel_mean, rel_scale, ref, valid, cells, "upper", "card")
    raise ValueError(variant)


VARIANTS = ("full_ssr", "upper_gate_0.8", "upper_gate_0.9")


def run_seed(cova, seed, device):
    run = routing.RUNS / f"seed_{seed}"
    val = torch.load(routing.STUDENT_CACHE / f"seed_{seed}/val.pt", weights_only=False)
    test = torch.load(routing.STUDENT_CACHE / f"seed_{seed}/test.pt", weights_only=False)
    router = cova.AvailabilityConditionedSufficientRouter(128, 256).to(device)
    cova.load_checkpoint(run / "router/best.pt", router)
    router.eval()
    with open(run / "router/calibration_student.json", encoding="utf-8") as handle:
        q_paper = torch.tensor(json.load(handle)["q_by_available_and_selected_cardinality"])
    q_gate = {"upper_gate_0.8": routing.calibrate(router, val, device, 0.8), "upper_gate_0.9": q_paper}
    result = {}
    for name, pattern in routing.SCENARIOS.items():
        available = torch.tensor(pattern, dtype=torch.bool)[None].expand(test["labels"].size(0), -1)
        rel_mean, rel_scale, ref, valid, task_mean = routing.router_relative(
            router, test["scout_tokens"].float(), available, device)
        raw_mean = task_mean.mean(-1)
        result[name] = {}
        for variant in VARIANTS:
            cells = routing.q_cells(q_gate.get(variant, q_paper), available)
            choice = route(variant, rel_mean, rel_scale, ref, valid, cells, raw_mean)
            evaluated = routing.evaluate_route(choice, test, available)
            result[name][variant] = {
                **routing.summarize(evaluated),
                "all_available_route_rate": evaluated["all_available_route_rate"],
                "any_task_correct_to_wrong": evaluated["any_task_correct_to_wrong"],
            }
    return result


def main():
    cova = routing.load_cova()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {s: run_seed(cova, s, device) for s in SEEDS}
    json.dump(results, open(OUT / "ssr_upper_gate.json", "w"), indent=2)

    def ms(values, scale=100.0):
        values = np.asarray(values) * scale
        return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"

    singles = [k for k in routing.SCENARIOS if k != "full"]
    lines = ["| Variant | Dynamic mAcc | mF1 | Branches | Rel. cost (%) | All-available route (%) | Known-single mAcc |",
             "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        full = [results[s]["full"][v] for s in SEEDS]
        single = [np.mean([results[s][k][v]["macc"] for k in singles]) for s in SEEDS]
        lines.append(f"| {v} | {ms([r['macc'] for r in full])} | {ms([r['mf1'] for r in full])} | "
                     f"{ms([r['avg_branches'] for r in full], 1)} | {ms([r['relative_cost'] for r in full])} | "
                     f"{ms([r['all_available_route_rate'] for r in full])} | {ms(single)} |")
    text = "\n".join(lines)
    (OUT / "ssr_upper_gate.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
