"""Comparison of COVA-MTL with the two policy-based selectors under the same budget (Table A11).

The policies of policy_selector.py are trained with per-clip loss normalization and an annealed entropy
regularization over a log-spaced lambda grid. For each selector, the operating point with the highest
validation mAcc whose validation branch cost does not exceed that of COVA-MTL is chosen on validation only
and evaluated once on the test split, with paired clip-level bootstrap confidence intervals.
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
spec = importlib.util.spec_from_file_location("policy", HERE / "policy_selector.py")
policy = importlib.util.module_from_spec(spec)
sys.modules["policy"] = policy
spec.loader.exec_module(policy)
routing, stats = policy.routing, policy.stats

SEEDS = (42, 43, 44)
OUT = RUNS_ROOT / "analysis" / "policy_selector_comparison"
MASKS, COSTS = routing.MASKS, routing.COSTS
EPOCHS = 40
BETA0 = 0.05
GRIDS = {"raw": np.geomspace(0.005, 0.3, 18).round(5).tolist(),
         "norm": np.geomspace(0.02, 3.0, 18).round(5).tolist()}


def transform(loss_tab, how):
    if how == "raw":
        return loss_tab
    lo = loss_tab.min(1, keepdim=True).values
    hi = loss_tab.max(1, keepdim=True).values
    return (loss_tab - lo) / (hi - lo).clamp_min(1e-8)


def train_policy(kind, cache, lam, how, device, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = policy.Policy(31 if kind == "subset" else 5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scout = cache["scout_tokens"].float().to(device)
    loss_tab = transform(cache["subset_loss"][:, 1:32].float().mean(-1), how).to(device)
    cand_cost = (MASKS.float() * COSTS).sum(1).to(device)
    n = scout.size(0)
    steps = EPOCHS * ((n + 127) // 128)
    step = 0
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, 128):
            idx = order[start:start + 128]
            av = policy.random_availability(idx.numel(), g).to(device)
            prob = policy.subset_probs(kind, model(scout[idx], av), av)
            rel_cost = cand_cost[None] / (av.float() * COSTS.to(device)).sum(1, keepdim=True)
            beta = BETA0 * (1 - step / steps)
            entropy = -(prob * prob.clamp_min(1e-12).log()).sum(1).mean()
            objective = (prob * (loss_tab[idx] + lam * rel_cost)).sum(1).mean() - beta * entropy
            opt.zero_grad()
            objective.backward()
            opt.step()
            step += 1
    model.eval()
    return model


def diversity(choice):
    counts = torch.bincount(choice, minlength=31).float()
    return {"distinct_subsets": int((counts > 0).sum()),
            "top_subset_share": float(counts.max() / counts.sum()),
            "top_subset": stats.name_of(int(counts.argmax())) if hasattr(stats, "name_of") else int(counts.argmax())}


def run_seed(cova, seed, device):
    s = stats.Seed(cova, seed, device)
    teacher_train = torch.load(routing.RUNS / f"seed_{seed}/oracle_cache/train.pt", weights_only=False)
    full = [1] * 5
    n_val = s.val["labels"].size(0)
    cova_val_choice, *_ = s.route(s.val, full)
    budget = stats.cost_of(cova_val_choice, stats.avail(n_val, full), COSTS)
    out = {"cova_val_cost": budget}
    for kind in ("gate", "subset"):
        curve = []
        for how, grid in GRIDS.items():
            for lam in grid:
                model = train_policy(kind, teacher_train, lam, how, device, seed)
                vch = policy.policy_choice(kind, model, s.val, full, device)
                point = {"transform": how, "lambda": lam, "val_macc": stats.macc(s.val, vch),
                         "val_cost": stats.cost_of(vch, stats.avail(n_val, full), COSTS), "test": {}}
                for scen, pattern in routing.SCENARIOS.items():
                    tch = policy.policy_choice(kind, model, s.test, pattern, device)
                    a = stats.avail(tch.numel(), pattern)
                    point["test"][scen] = {
                        "macc": stats.macc(s.test, tch),
                        "mf1": routing.metrics(s.test["labels"], s.test["predictions"][torch.arange(tch.numel()), tch + 1])["mf1"],
                        "cost": stats.cost_of(tch, a, COSTS),
                        "avg_branches": float(routing.CARD[tch].float().mean()),
                        "clip_scores": stats.clip_correct(s.test, tch).mean(1).tolist(),
                    } | diversity(tch)
                curve.append(point)
                t = point["test"]["full"]
                print(f"seed {seed} {kind} {how} lam={lam}: val {point['val_macc']:.4f}@{point['val_cost']:.3f} "
                      f"test {t['macc']:.4f}@{t['cost']:.3f} br {t['avg_branches']:.2f} "
                      f"distinct {t['distinct_subsets']} top {t['top_subset_share']:.2f}", flush=True)
        usable = [p for p in curve if p["val_cost"] <= budget + 1e-9]
        chosen = max(usable, key=lambda p: p["val_macc"]) if usable else min(curve, key=lambda p: p["val_cost"])
        out[f"policy_{kind}"] = {
            "curve": [{k: v for k, v in p.items() if k != "test"} |
                      {"test_full": {k: v for k, v in p["test"]["full"].items() if k != "clip_scores"}} for p in curve],
            "chosen_transform": chosen["transform"], "chosen_lambda": chosen["lambda"], "chosen": chosen}
    out["cova_clip_scores"], out["cova"] = {}, {}
    for scen, pattern in routing.SCENARIOS.items():
        ch, *_ = s.route(s.test, pattern)
        out["cova_clip_scores"][scen] = stats.clip_correct(s.test, ch).mean(1).tolist()
        out["cova"][scen] = {"macc": stats.macc(s.test, ch),
                             "mf1": routing.metrics(s.test["labels"], s.test["predictions"][torch.arange(ch.numel()), ch + 1])["mf1"],
                             "cost": stats.cost_of(ch, stats.avail(ch.numel(), pattern), COSTS)} | diversity(ch)
    return out


def main():
    cova = routing.load_cova()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {seed: run_seed(cova, seed, device) for seed in SEEDS}
    rng = np.random.default_rng(0)
    summary = {}
    for kind in ("gate", "subset"):
        summary[kind] = {}
        for scen in routing.SCENARIOS:
            diffs = np.mean([np.array(results[s]["cova_clip_scores"][scen]) -
                             np.array(results[s][f"policy_{kind}"]["chosen"]["test"][scen]["clip_scores"]) for s in SEEDS], 0)
            ch = [results[s][f"policy_{kind}"]["chosen"]["test"][scen] for s in SEEDS]
            summary[kind][scen] = {
                "policy_macc": [c["macc"] for c in ch], "policy_mf1": [c["mf1"] for c in ch],
                "policy_cost": [c["cost"] for c in ch], "policy_distinct": [c["distinct_subsets"] for c in ch],
                "policy_top_share": [c["top_subset_share"] for c in ch],
                "cova_macc": [results[s]["cova"][scen]["macc"] for s in SEEDS],
                "cova_cost": [results[s]["cova"][scen]["cost"] for s in SEEDS],
                "cova_distinct": [results[s]["cova"][scen]["distinct_subsets"] for s in SEEDS],
                "cova_top_share": [results[s]["cova"][scen]["top_subset_share"] for s in SEEDS],
                "cova_minus_policy": stats.bootstrap(diffs, rng),
            }
        summary[kind]["chosen"] = [(results[s][f"policy_{kind}"]["chosen_transform"],
                                    results[s][f"policy_{kind}"]["chosen_lambda"]) for s in SEEDS]
        summary[kind]["val_cost"] = [results[s][f"policy_{kind}"]["chosen"]["val_cost"] for s in SEEDS]
    for s in SEEDS:
        for kind in ("gate", "subset"):
            for scen in routing.SCENARIOS:
                results[s][f"policy_{kind}"]["chosen"]["test"][scen].pop("clip_scores")
        results[s].pop("cova_clip_scores")
    json.dump({"per_seed": results, "summary": summary}, open(OUT / "policy_selector_comparison.json", "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
