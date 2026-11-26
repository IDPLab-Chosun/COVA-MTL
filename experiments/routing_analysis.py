"""Offline routing analysis on the cached Student outputs (seeds 42-44, no retraining).

Also the shared library of the other analysis programs (ACVE outputs, calibration, SSR selection, route
evaluation). Outputs per seed:
  - agreement of the Teacher-trained ACVE with the Teacher and Student test losses (Table 9);
  - paired sample-level and task-wise changes of the selected routes relative to all available
    modalities under full and single-missing availability (Tables 10 and 11);
  - SSR with the branch-cost sum instead of the cardinality as the selection objective (Section 4.9) and
    calibrated upper-bound gates (Section 3.5).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
AIDE_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
SOURCE = REPO / "main" / "train_multiseed.py"
RUNS = RUNS_ROOT / "AIDE_COVA_MTL_v6_3_MultiSeed5"
STUDENT_CACHE = RUNS_ROOT / "analysis" / "student_cache"
OUT = RUNS_ROOT / "analysis" / "routing_analysis"

MODALITIES = ("face", "body", "scene", "posture", "gesture")
TASKS = ("DER", "DBR", "TCR", "VBR")
COSTS = torch.tensor([1.0, 1.0, 1.0, 0.05, 0.05])
MASKS = torch.tensor([[bool(s & (1 << m)) for m in range(5)] for s in range(1, 32)])
CARD = MASKS.sum(1)
CAND_COST = (MASKS.float() * COSTS).sum(1)
SCENARIOS = {"full": [1, 1, 1, 1, 1]}
for _m, _name in enumerate(MODALITIES):
    _pattern = [1] * 5
    _pattern[_m] = 0
    SCENARIOS[f"missing_{_name}"] = _pattern
COVERAGES = (0.8, 0.9)
TASK_CLASSES = (5, 7, 3, 5)


def load_cova():
    spec = importlib.util.spec_from_file_location("cova", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cova"] = module
    spec.loader.exec_module(module)
    return module


def subset_index(mask: torch.Tensor) -> torch.Tensor:
    """Candidate index 0..30 of a boolean [N, 5] mask."""
    weights = torch.tensor([1, 2, 4, 8, 16])
    return (mask.long() * weights).sum(1) - 1


def metrics(labels: torch.Tensor, preds: torch.Tensor) -> dict:
    acc = {t: float((preds[:, i] == labels[:, i]).float().mean()) for i, t in enumerate(TASKS)}
    f1 = {
        t: float(f1_score(labels[:, i].numpy(), preds[:, i].numpy(), average="macro"))
        for i, t in enumerate(TASKS)
    }
    return {"acc": acc, "f1": f1, "macc": float(np.mean(list(acc.values()))),
            "mf1": float(np.mean(list(f1.values())))}


# -----------------------------------------------------------------------------
# ACVE outputs and calibration
# -----------------------------------------------------------------------------
@torch.no_grad()
def router_relative(router, scout: torch.Tensor, available: torch.Tensor, device):
    """Paired candidate-minus-reference predicted mean regret and scale."""
    routed = (scout * available[:, :, None].float()).to(device)
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        task_mean, task_scale = router(routed, available.to(device))
    task_mean, task_scale = task_mean.float().cpu(), task_scale.float().cpu()
    mean = task_mean.mean(-1)
    scale = (task_scale.square().sum(-1).sqrt() / 4).clamp_min(1e-4)
    rows = torch.arange(mean.size(0))
    ref = subset_index(available)
    rel_mean = mean - mean[rows, ref][:, None]
    rel_scale = (scale.square() + scale[rows, ref][:, None].square()).sqrt().clamp_min(1e-4)
    rel_mean[rows, ref] = 0.0
    valid = ~(MASKS[None] & (~available)[:, None]).any(-1)
    return rel_mean, rel_scale, ref, valid, task_mean


def true_relative(cache: dict, available: torch.Tensor):
    loss = cache["subset_loss"][:, 1:32].float()          # [N, 31, 4]
    ref = subset_index(available)
    rows = torch.arange(loss.size(0))
    task_rel = loss - loss[rows, ref][:, None, :]
    return task_rel.mean(-1), task_rel                    # [N, 31], [N, 31, 4]


def calibrate(router, cache: dict, device, coverage: float) -> torch.Tensor:
    """Re-implements the paper's one-sided (A, S)-conditional calibration."""
    scout = cache["scout_tokens"].float()
    residuals = {}
    for pattern in MASKS.flip(0):                          # all 31 availability states
        available = pattern[None].expand(scout.size(0), -1)
        rel_mean, rel_scale, ref, valid, _ = router_relative(router, scout, available, device)
        target, _ = true_relative(cache, available)
        standardized = (target - rel_mean) / rel_scale
        a = int(pattern.sum())
        for s in range(1, a + 1):
            use = valid & (CARD[None] == s)
            residuals.setdefault((a, s), []).append(standardized[use])
    q = torch.ones(5, 5)
    for (a, s), parts in residuals.items():
        values = torch.cat(parts)
        rank = min(values.numel(), math.ceil((values.numel() + 1) * coverage))
        q[a - 1, s - 1] = max(0.0, float(values.kthvalue(rank).values))
    return q


def q_cells(q: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    a = available.sum(1).long() - 1
    return q[a[:, None], (CARD - 1)[None]]


def select(rel_mean, rel_scale, ref, valid, margin, gate: str, key: str) -> torch.Tensor:
    """SSR-style lexicographic selection.

    gate  : "mean"  -> admissible if rel_mean <= 0           (paper, Eq. 15)
            "upper" -> admissible if rel_mean + margin * rel_scale <= 0
    key   : "card"  -> minimise the number of branches       (paper)
            "cost"  -> minimise the branch-cost sum
    margin: [N, 31] calibrated q cells or a scalar risk multiplier.
    Ties are broken by the calibrated upper value, as in the paper.
    """
    rows = torch.arange(rel_mean.size(0))
    upper = rel_mean + margin * rel_scale
    upper[rows, ref] = 0.0
    gate_value = rel_mean if gate == "mean" else upper
    admissible = valid & (gate_value <= 0)
    admissible[rows, ref] = True
    primary = (CARD.float() if key == "card" else CAND_COST)[None].expand_as(rel_mean)
    best = primary.masked_fill(~admissible, float("inf")).min(1).values
    eligible = admissible & (primary <= best[:, None] + 1e-6)
    return upper.masked_fill(~eligible, float("inf")).argmin(1)


# -----------------------------------------------------------------------------
# Route evaluation (performance, cost, paired regret)
# -----------------------------------------------------------------------------
def evaluate_route(choice: torch.Tensor, cache: dict, available: torch.Tensor,
                   executed_cost: torch.Tensor | None = None) -> dict:
    rows = torch.arange(choice.numel())
    labels = cache["labels"]
    preds = cache["predictions"][rows, choice + 1]
    ref = subset_index(available)
    ref_preds = cache["predictions"][rows, ref + 1]
    out = metrics(labels, preds)
    selected = MASKS[choice]
    available_cost = (available.float() * COSTS).sum(1)
    cost = executed_cost if executed_cost is not None else (selected.float() * COSTS).sum(1)
    out["relative_cost"] = float((cost / available_cost).mean())
    out["avg_branches"] = float(selected.float().sum(1).mean())
    out["all_available_route_rate"] = float((choice == ref).float().mean())
    out["selection_rate"] = {m: float(selected[:, i].float().mean()) for i, m in enumerate(MODALITIES)}

    mean_rel, task_rel = true_relative(cache, available)
    regret = mean_rel[rows, choice]
    task_regret = task_rel[rows, choice]
    out["paired_regret"] = {
        "violation_rate_gt0": float((regret > 0).float().mean()),
        "violation_rate_gt0.01": float((regret > 0.01).float().mean()),
        "violation_rate_gt0.05": float((regret > 0.05).float().mean()),
        "violation_rate_gt0.10": float((regret > 0.10).float().mean()),
        "mean_regret": float(regret.mean()),
        "mean_positive_regret": float(regret.clamp_min(0).mean()),
        "mean_regret_given_violation": float(regret[regret > 0].mean()) if (regret > 0).any() else 0.0,
        "p90_regret": float(regret.quantile(0.9)),
        "p99_regret": float(regret.quantile(0.99)),
        "max_regret": float(regret.max()),
        "task_mean_regret": {t: float(task_regret[:, i].mean()) for i, t in enumerate(TASKS)},
        "task_violation_rate": {t: float((task_regret[:, i] > 0).float().mean()) for i, t in enumerate(TASKS)},
    }
    correct_ref = ref_preds == labels
    correct_sel = preds == labels
    out["prediction_flips_vs_all_available"] = {
        t: {
            "correct_to_wrong": float((correct_ref[:, i] & ~correct_sel[:, i]).float().mean()),
            "wrong_to_correct": float((~correct_ref[:, i] & correct_sel[:, i]).float().mean()),
        }
        for i, t in enumerate(TASKS)
    }
    out["any_task_correct_to_wrong"] = float((correct_ref & ~correct_sel).any(1).float().mean())
    return out


def summarize(result: dict) -> dict:
    return {
        "macc": result["macc"], "mf1": result["mf1"],
        "relative_cost": result["relative_cost"], "avg_branches": result["avg_branches"],
        "violation_rate_gt0": result["paired_regret"]["violation_rate_gt0"],
        "mean_positive_regret": result["paired_regret"]["mean_positive_regret"],
    }


# -----------------------------------------------------------------------------
# ACVE validity for the deployed Student
# -----------------------------------------------------------------------------
def validity(rel_mean, rel_scale, ref, valid, q, available, cache) -> dict:
    target, _ = true_relative(cache, available)
    rows = torch.arange(rel_mean.size(0))
    use = valid.clone()
    use[rows, ref] = False
    pred_ok = rel_mean[use] <= 0
    true_ok = target[use] <= 0
    upper = rel_mean + q_cells(q, available) * rel_scale
    spearman = []
    for i in range(rel_mean.size(0)):
        cand = use[i]
        if cand.sum() < 3:
            continue
        a = rel_mean[i, cand].argsort().argsort().float()
        b = target[i, cand].argsort().argsort().float()
        spearman.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    return {
        "sign_accuracy": float((pred_ok == true_ok).float().mean()),
        "sufficiency_macro_f1": float(f1_score(true_ok.numpy(), pred_ok.numpy(), average="macro")),
        "false_sufficient_rate": float((pred_ok & ~true_ok).float().sum() / max(int(pred_ok.sum()), 1)),
        "mae": float((rel_mean[use] - target[use]).abs().mean()),
        "mean_spearman": float(np.nanmean(spearman)) if spearman else None,
        "upper_coverage": float((target[use] <= upper[use]).float().mean()),
    }


# -----------------------------------------------------------------------------
def run_seed(cova, seed: int, device) -> dict:
    run = RUNS / f"seed_{seed}"
    teacher = {s: torch.load(run / "oracle_cache" / f"{s}.pt", weights_only=False) for s in ("val", "test")}
    student = {s: torch.load(STUDENT_CACHE / f"seed_{seed}" / f"{s}.pt", weights_only=False) for s in ("val", "test")}
    assert torch.equal(teacher["test"]["labels"], student["test"]["labels"]), "test order mismatch"
    assert torch.equal(teacher["val"]["labels"], student["val"]["labels"]), "val order mismatch"

    router = cova.AvailabilityConditionedSufficientRouter(128, 256).to(device)
    cova.load_checkpoint(run / "router" / "best.pt", router)
    router.eval()
    with open(run / "router" / "calibration_student.json", encoding="utf-8") as handle:
        q_paper = torch.tensor(json.load(handle)["q_by_available_and_selected_cardinality"])
    with open(run / "router" / "calibration_teacher.json", encoding="utf-8") as handle:
        q_teacher = torch.tensor(json.load(handle)["q_by_available_and_selected_cardinality"])
    q_by_coverage = {c: calibrate(router, student["val"], device, c) for c in COVERAGES}
    report = {"seed": seed, "calibration_check": {
        "q_student_paper": q_paper.tolist(), "q_student_recomputed_0.9": q_by_coverage[0.9].tolist(),
        "max_abs_diff": float((q_paper - q_by_coverage[0.9]).abs().max()),
    }}

    test = student["test"]
    scenarios = {}
    for name, pattern in SCENARIOS.items():
        available = torch.tensor(pattern, dtype=torch.bool)[None].expand(test["labels"].size(0), -1)
        rel_mean, rel_scale, ref, valid, _ = router_relative(router, test["scout_tokens"].float(), available, device)
        q_paper_cells = q_cells(q_paper, available)
        variants = {
            "paper_mean_card": select(rel_mean, rel_scale, ref, valid, q_paper_cells, "mean", "card"),
            "mean_cost": select(rel_mean, rel_scale, ref, valid, q_paper_cells, "mean", "cost"),
        }
        for c, q in q_by_coverage.items():
            cells = q_cells(q, available)
            variants[f"upper{c}_card"] = select(rel_mean, rel_scale, ref, valid, cells, "upper", "card")
            variants[f"upper{c}_cost"] = select(rel_mean, rel_scale, ref, valid, cells, "upper", "cost")
        block = {
            "all_available": evaluate_route(ref, test, available),
            "variants": {k: evaluate_route(v, test, available) for k, v in variants.items()},
            "validity_vs_student": validity(rel_mean, rel_scale, ref, valid, q_paper, available, test),
            "validity_vs_teacher": validity(rel_mean, rel_scale, ref, valid, q_teacher, available, teacher["test"]),
        }
        scenarios[name] = block
    report["scenarios"] = scenarios
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    cli = parser.parse_args()
    cova = load_cova()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in cli.seeds.split(",")]:
        report = run_seed(cova, seed, device)
        with open(OUT / f"seed_{seed}.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        full = report["scenarios"]["full"]
        print(f"seed {seed}: calibration diff {report['calibration_check']['max_abs_diff']:.4f}")
        for key in ("paper_mean_card", "mean_cost", "upper0.9_card", "upper0.9_cost"):
            v = full["variants"][key]
            print(f"  {key:18s} mAcc={v['macc']:.4f} cost={v['relative_cost']:.3f} "
                  f"viol={v['paired_regret']['violation_rate_gt0']:.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
