"""Statistics of the selected routes on the cached Student outputs (seeds 42-44, no retraining).

  A. Routes with increased loss under full availability: share without a changed prediction, task-wise
     share of the positive loss increase, and rates by cardinality and by excluded modality (Section 4.8).
  C. Paired clip-level bootstrap confidence intervals with 10,000 resamples: Dynamic minus all available
     modalities (Section 4.4) and Dynamic minus the budget-matched fixed subset (Table A11).
  D. Coverage and violation rate of the calibrated upper bound on the selected routes (Section 4.7).
Also the helper functions used by the policy-selector and fixed-subset programs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import math
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
OUT = RUNS_ROOT / "analysis" / "route_statistics"
MOD = routing.MODALITIES
MASKS, CARD = routing.MASKS, routing.CARD
BOOT = 10000


def idx_of(name: str) -> int:
    keep = torch.tensor([[m in name.split("+") for m in MOD]])
    return int(routing.subset_index(keep)[0])


def name_of(c: int) -> str:
    return "+".join(m for i, m in enumerate(MOD) if MASKS[c, i])


def avail(n, pattern):
    return torch.tensor(pattern, dtype=torch.bool)[None].expand(n, -1)


def clip_correct(cache, choice):
    rows = torch.arange(choice.numel())
    preds = cache["predictions"][rows, choice + 1]
    return (preds == cache["labels"]).float()          # [N, 4]


def macc(cache, choice):
    return float(clip_correct(cache, choice).mean())


def restrict(c: int, available_row: torch.Tensor) -> int:
    """Fixed subset c with unavailable modalities removed (falls back to all available)."""
    keep = MASKS[c] & available_row
    if not keep.any():
        keep = available_row.clone()
    return int(routing.subset_index(keep[None])[0])


class Seed:
    def __init__(self, cova, seed, device):
        run = routing.RUNS / f"seed_{seed}"
        self.seed = seed
        self.device = device
        self.val = torch.load(routing.STUDENT_CACHE / f"seed_{seed}/val.pt", weights_only=False)
        self.test = torch.load(routing.STUDENT_CACHE / f"seed_{seed}/test.pt", weights_only=False)
        self.router = cova.AvailabilityConditionedSufficientRouter(128, 256).to(device)
        cova.load_checkpoint(run / "router/best.pt", self.router)
        self.router.eval()
        with open(run / "router/calibration_student.json", encoding="utf-8") as fh:
            self.q = torch.tensor(json.load(fh)["q_by_available_and_selected_cardinality"])

    def route(self, cache, pattern):
        a = avail(cache["labels"].size(0), pattern)
        rel_mean, rel_scale, ref, valid, _ = routing.router_relative(
            self.router, cache["scout_tokens"].float(), a, self.device)
        cells = routing.q_cells(self.q, a)
        choice = routing.select(rel_mean, rel_scale, ref, valid, cells, "mean", "card")
        return choice, rel_mean, rel_scale, ref, valid, cells, a


def cost_of(choice, available, costs):
    sel = MASKS[choice].float()
    return float(((sel * costs).sum(1) / (available.float() * costs).sum(1)).mean())


# -----------------------------------------------------------------------------
def part_a(s: Seed):
    """Harmful selections under full availability."""
    full = [1] * 5
    choice, rel_mean, rel_scale, ref, valid, cells, a = s.route(s.test, full)
    n = choice.numel()
    rows = torch.arange(n)
    mean_rel, task_rel = routing.true_relative(s.test, a)
    regret = mean_rel[rows, choice]
    pred = rel_mean[rows, choice]
    correct_ref = clip_correct(s.test, ref).bool()
    correct_sel = clip_correct(s.test, choice).bool()
    lost_any = (correct_ref & ~correct_sel).any(1)
    nonref = choice != ref
    out = {"n": n, "nonref_rate": float(nonref.float().mean())}

    by_card = {}
    for k in range(1, 6):
        m = CARD[choice] == k
        if m.any():
            by_card[k] = {"share": float(m.float().mean()),
                          "viol_gt0": float((regret[m] > 0).float().mean()),
                          "viol_gt010": float((regret[m] > 0.10).float().mean()),
                          "lost": float(lost_any[m].float().mean())}
    out["by_cardinality"] = by_card

    by_drop = {}
    sel = MASKS[choice]
    for i, mname in enumerate(MOD):
        m = ~sel[:, i]
        if m.any():
            by_drop[mname] = {"share": float(m.float().mean()),
                              "viol_gt0": float((regret[m] > 0).float().mean()),
                              "viol_gt010": float((regret[m] > 0.10).float().mean()),
                              "lost": float(lost_any[m].float().mean())}
    out["by_dropped_modality"] = by_drop

    # Predicted margin of the chosen non-reference route (rel_mean <= 0 by admissibility).
    pm = pred[nonref]
    rg = regret[nonref]
    lo = lost_any[nonref]
    qs = torch.quantile(pm, torch.tensor([0.25, 0.5, 0.75]))
    bins = [pm <= qs[0], (pm > qs[0]) & (pm <= qs[1]), (pm > qs[1]) & (pm <= qs[2]), pm > qs[2]]
    out["by_predicted_margin_quartile"] = [
        {"pred_range": [float(pm[b].min()), float(pm[b].max())], "n": int(b.sum()),
         "viol_gt0": float((rg[b] > 0).float().mean()), "viol_gt010": float((rg[b] > 0.10).float().mean()),
         "lost": float(lo[b].float().mean())} for b in bins]
    ra = pm.argsort().argsort().float()
    rb = rg.argsort().argsort().float()
    out["spearman_pred_vs_true_nonref"] = float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])

    harmful = regret > 0
    out["harmful"] = {
        "rate": float(harmful.float().mean()),
        "with_lost_prediction": float((harmful & lost_any).float().sum() / max(int(harmful.sum()), 1)),
        "without_lost_prediction": float((harmful & ~lost_any).float().sum() / max(int(harmful.sum()), 1)),
        "lost_rate_overall": float(lost_any.float().mean()),
        "task_share_of_positive_regret": {
            t: float(task_rel[rows, choice][:, j].clamp_min(0).sum() /
                     task_rel[rows, choice].clamp_min(0).sum()) for j, t in enumerate(routing.TASKS)},
    }
    return out


# -----------------------------------------------------------------------------
def paired_clip_scores(s: Seed, cost_fixed: str):
    """Per-clip mean correctness (over 4 tasks) for each method, every scenario."""
    n = s.test["labels"].size(0)
    out = {}
    for scen, pattern in routing.SCENARIOS.items():
        a_row = torch.tensor(pattern, dtype=torch.bool)
        choice, *_ = s.route(s.test, pattern)
        ref = routing.subset_index(avail(n, pattern))
        out[scen] = {
            "dynamic": clip_correct(s.test, choice).mean(1).numpy(),
            "available_all": clip_correct(s.test, ref).mean(1).numpy(),
        }
        if scen == "full":   # budget-matched fixed subset selected on validation (fixed_subset_selection.py)
            out[scen]["cost_matched"] = clip_correct(
                s.test, torch.full((n,), restrict(idx_of(cost_fixed), a_row))).mean(1).numpy()
    return out


def bootstrap(diff: np.ndarray, rng) -> dict:
    n = diff.size
    idx = rng.integers(0, n, size=(BOOT, n))
    means = diff[idx].mean(1)
    pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
    # exact two-sided sign test on clips whose score differs
    k, m = min(pos, neg), pos + neg
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return {"mean_pp": 100 * float(diff.mean()), "ci95_pp": [100 * float(np.quantile(means, 0.025)),
                                                             100 * float(np.quantile(means, 0.975))],
            "clips_better": pos, "clips_worse": neg, "sign_test_p": p}


def part_c(scores_by_seed):
    rng = np.random.default_rng(0)
    res = {}
    for scen in routing.SCENARIOS:
        res[scen] = {}
        for other in [k for k in next(iter(scores_by_seed.values()))[scen] if k != "dynamic"]:
            per_seed = {s: bootstrap(sc[scen]["dynamic"] - sc[scen][other], rng) for s, sc in scores_by_seed.items()}
            pooled = np.mean([sc[scen]["dynamic"] - sc[scen][other] for sc in scores_by_seed.values()], 0)
            res[scen][f"dynamic_minus_{other}"] = {"per_seed": per_seed, "pooled_over_seeds": bootstrap(pooled, rng)}
    return res


# -----------------------------------------------------------------------------
def part_d(s: Seed):
    """Calibration of the selected routes only."""
    res = {}
    for scen, pattern in routing.SCENARIOS.items():
        choice, rel_mean, rel_scale, ref, valid, cells, a = s.route(s.test, pattern)
        rows = torch.arange(choice.numel())
        target, _ = routing.true_relative(s.test, a)
        upper = rel_mean + cells * rel_scale
        nonref = choice != ref
        t_sel, u_sel = target[rows, choice][nonref], upper[rows, choice][nonref]
        use = valid.clone()
        use[rows, ref] = False
        block = {
            "nonref_rate": float(nonref.float().mean()),
            "selected_coverage": float((t_sel <= u_sel).float().mean()) if nonref.any() else None,
            "selected_violation_gt0": float((t_sel > 0).float().mean()) if nonref.any() else None,
            "selected_mean_upper": float(u_sel.mean()) if nonref.any() else None,
            "all_candidate_coverage": float((target[use] <= upper[use]).float().mean()),
            "by_cardinality": {},
        }
        for k in range(1, 5):
            m = CARD[choice][nonref] == k
            if m.any():
                block["by_cardinality"][k] = {"n": int(m.sum()),
                                              "coverage": float((t_sel[m] <= u_sel[m]).float().mean()),
                                              "violation_gt0": float((t_sel[m] > 0).float().mean())}
        res[scen] = block
    return res


# -----------------------------------------------------------------------------
def main():
    cova = routing.load_cova()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    vm = json.load(open(RUNS_ROOT / "analysis" / "fixed_subset_selection" / "fixed_subset_selection.json"))
    cost_fixed = {s: vm[str(s)]["fixed_subset"]["subset"] for s in SEEDS}
    results, scores = {}, {}
    for seed in SEEDS:
        s = Seed(cova, seed, device)
        results[seed] = {"A": part_a(s), "D": part_d(s)}
        scores[seed] = paired_clip_scores(s, cost_fixed[seed])
        print("seed", seed, "done", flush=True)
    results["C"] = part_c(scores)
    json.dump(results, open(OUT / "route_statistics.json", "w"), indent=2)
    print("saved", OUT / "route_statistics.json")


if __name__ == "__main__":
    main()
