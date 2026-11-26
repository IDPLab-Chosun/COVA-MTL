"""Policy-based modality selectors that follow the selection principles of AdaMML and DynMM.

Both policies use the same previews (scout tokens), the same Teacher supervision as ACVE, and the same
deployed Student outputs as COVA-MTL, so that only the selection rule differs:
  gate   : independent per-modality execution gates (AdaMML principle);
  subset : categorical choice among the valid subsets (DynMM principle).
Both minimize  E_pi[mean task loss of the chosen subset] + lambda * E_pi[relative branch cost]  with the
exact expectation over subsets. The policies are availability-conditioned. This file defines the policies
and their training; policy_selector_comparison.py produces the reported comparison (Table A11).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
AIDE_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
spec = importlib.util.spec_from_file_location("routing", HERE / "routing_analysis.py")
routing = importlib.util.module_from_spec(spec)
sys.modules["routing"] = routing
spec.loader.exec_module(routing)
spec_stats = importlib.util.spec_from_file_location("stats", HERE / "route_statistics.py")
stats = importlib.util.module_from_spec(spec_stats)
sys.modules["stats"] = stats
spec_stats.loader.exec_module(stats)

SEEDS = (42, 43, 44)
OUT = RUNS_ROOT / "analysis" / "policy_selector"
MASKS = routing.MASKS                     # [31, 5]
COSTS = routing.COSTS
LAMBDAS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
EPOCHS = 40


class Policy(nn.Module):
    def __init__(self, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(5 * 128 + 5), nn.Linear(5 * 128 + 5, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, out_dim))

    def forward(self, scout, available):
        x = scout * available[:, :, None].float()
        return self.net(torch.cat([x.flatten(1), available.float()], 1))


def subset_probs(kind, logits, available):
    """Probability of each of the 31 subsets; invalid subsets get 0."""
    valid = ~(MASKS[None].to(logits.device) & (~available)[:, None]).any(-1)       # [B, 31]
    if kind == "subset":
        return torch.softmax(logits.masked_fill(~valid, -1e9), -1)
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6) * available.float()             # [B, 5]
    m = MASKS[None].to(logits.device).float()                                       # [1, 31, 5]
    prob = (p[:, None] * m + (1 - p[:, None]) * (1 - m)).prod(-1) * valid.float()
    return prob / prob.sum(-1, keepdim=True).clamp_min(1e-12)                       # renormalize w/o empty set


def random_availability(n, generator):
    full = torch.rand(n, generator=generator) < 0.5
    pats = MASKS[torch.randint(0, 31, (n,), generator=generator)]
    pats[full] = True
    return pats


def train_policy(kind, cache, lam, device, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Policy(31 if kind == "subset" else 5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scout = cache["scout_tokens"].float().to(device)
    loss_tab = cache["subset_loss"][:, 1:32].float().mean(-1).to(device)          # [N, 31]
    cand_cost = (MASKS.float() * COSTS).sum(1).to(device)                           # [31]
    n = scout.size(0)
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, 128):
            idx = order[start:start + 128]
            av = random_availability(idx.numel(), g).to(device)
            prob = subset_probs(kind, model(scout[idx], av), av)
            rel_cost = cand_cost[None] / (av.float() * COSTS.to(device)).sum(1, keepdim=True)
            objective = (prob * (loss_tab[idx] + lam * rel_cost)).sum(1).mean()
            opt.zero_grad()
            objective.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def policy_choice(kind, model, cache, pattern, device):
    n = cache["labels"].size(0)
    av = torch.tensor(pattern, dtype=torch.bool)[None].expand(n, -1).to(device)
    logits = model(cache["scout_tokens"].float().to(device), av)
    if kind == "subset":
        return subset_probs(kind, logits, av).argmax(1).cpu()
    keep = (torch.sigmoid(logits) > 0.5) & av
    empty = ~keep.any(1)
    if empty.any():
        best = torch.sigmoid(logits).masked_fill(~av, -1).argmax(1)
        keep[empty, best[empty]] = True
    return routing.subset_index(keep.cpu())


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
        for lam in LAMBDAS:
            model = train_policy(kind, teacher_train, lam, device, seed)
            vch = policy_choice(kind, model, s.val, full, device)
            point = {"lambda": lam, "val_macc": stats.macc(s.val, vch),
                     "val_cost": stats.cost_of(vch, stats.avail(n_val, full), COSTS), "test": {}}
            for scen, pattern in routing.SCENARIOS.items():
                tch = policy_choice(kind, model, s.test, pattern, device)
                a = stats.avail(tch.numel(), pattern)
                point["test"][scen] = {
                    "macc": stats.macc(s.test, tch),
                    "mf1": routing.metrics(s.test["labels"], s.test["predictions"][torch.arange(tch.numel()), tch + 1])["mf1"],
                    "cost": stats.cost_of(tch, a, COSTS),
                    "avg_branches": float(routing.CARD[tch].float().mean()),
                    "clip_scores": stats.clip_correct(s.test, tch).mean(1).tolist(),
                }
            curve.append(point)
            print(f"seed {seed} {kind} lam={lam}: val {point['val_macc']:.4f}@{point['val_cost']:.3f} "
                  f"test {point['test']['full']['macc']:.4f}@{point['test']['full']['cost']:.3f}", flush=True)
        usable = [p for p in curve if p["val_cost"] <= budget + 1e-9]
        chosen = max(usable, key=lambda p: p["val_macc"]) if usable else min(curve, key=lambda p: p["val_cost"])
        out[f"policy_{kind}"] = {"curve": [{k: v for k, v in p.items() if k != "test"} |
                                           {"test_full": {k: v for k, v in p["test"]["full"].items() if k != "clip_scores"}}
                                           for p in curve],
                                 "chosen_lambda": chosen["lambda"], "chosen": chosen}
    # COVA clip scores for paired tests
    out["cova_clip_scores"] = {}
    out["cova"] = {}
    for scen, pattern in routing.SCENARIOS.items():
        ch, *_ = s.route(s.test, pattern)
        out["cova_clip_scores"][scen] = stats.clip_correct(s.test, ch).mean(1).tolist()
        out["cova"][scen] = {"macc": stats.macc(s.test, ch),
                             "mf1": routing.metrics(s.test["labels"], s.test["predictions"][torch.arange(ch.numel()), ch + 1])["mf1"],
                             "cost": stats.cost_of(ch, stats.avail(ch.numel(), pattern), COSTS)}
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
            summary[kind][scen] = {
                "policy_macc": [results[s][f"policy_{kind}"]["chosen"]["test"][scen]["macc"] for s in SEEDS],
                "policy_mf1": [results[s][f"policy_{kind}"]["chosen"]["test"][scen]["mf1"] for s in SEEDS],
                "policy_cost": [results[s][f"policy_{kind}"]["chosen"]["test"][scen]["cost"] for s in SEEDS],
                "cova_macc": [results[s]["cova"][scen]["macc"] for s in SEEDS],
                "cova_cost": [results[s]["cova"][scen]["cost"] for s in SEEDS],
                "cova_minus_policy": stats.bootstrap(diffs, rng),
            }
        summary[kind]["chosen_lambda"] = [results[s][f"policy_{kind}"]["chosen_lambda"] for s in SEEDS]
        summary[kind]["val_cost"] = [results[s][f"policy_{kind}"]["chosen"]["val_cost"] for s in SEEDS]
    for s in SEEDS:
        for kind in ("gate", "subset"):
            for scen in routing.SCENARIOS:
                results[s][f"policy_{kind}"]["chosen"]["test"][scen].pop("clip_scores")
        results[s].pop("cova_clip_scores")
    json.dump({"per_seed": results, "summary": summary}, open(OUT / "policy_selector.json", "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
