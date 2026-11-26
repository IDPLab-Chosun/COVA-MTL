import os
"""Dynamic inference versus the mean of the fixed subsets within its validation budget (Table A11).

Paired clip-level bootstrap with 10,000 resamples over the seed-averaged clip accuracies.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
AIDE_ROOT = Path(os.environ.get("AIDE_ROOT", REPO / "data" / "AIDE_Dataset"))
spec = importlib.util.spec_from_file_location('stats', HERE / 'route_statistics.py')
stats = importlib.util.module_from_spec(spec); sys.modules['stats'] = stats; spec.loader.exec_module(stats)
routing = stats.routing
OUT = RUNS_ROOT / 'analysis'; OUT.mkdir(parents=True, exist_ok=True)
cova = routing.load_cova(); dev = torch.device('cuda'); FULL = [1]*5
diffs, macc_mean, cost_mean, res = [], [], [], {}
for seed in stats.SEEDS:
    s = stats.Seed(cova, seed, dev)
    nv, nt = s.val['labels'].size(0), s.test['labels'].size(0)
    dch, *_ = s.route(s.test, FULL); vch, *_ = s.route(s.val, FULL)
    budget = stats.cost_of(vch, stats.avail(nv, FULL), routing.COSTS)
    inb = [c for c in range(31) if stats.cost_of(torch.full((nv,), c), stats.avail(nv, FULL), routing.COSTS) <= budget + 1e-9]
    fixed_scores = np.mean([stats.clip_correct(s.test, torch.full((nt,), c)).mean(1).numpy() for c in inb], 0)
    dyn_scores = stats.clip_correct(s.test, dch).mean(1).numpy()
    diffs.append(dyn_scores - fixed_scores)
    macc_mean.append(100 * fixed_scores.mean())
    cost_mean.append(100 * np.mean([stats.cost_of(torch.full((nt,), c), stats.avail(nt, FULL), routing.COSTS) for c in inb]))
    res[seed] = {'n_in_budget': len(inb), 'fixed_mean_macc': macc_mean[-1], 'fixed_mean_cost': cost_mean[-1]}
rng = np.random.default_rng(0)
res['summary'] = {'fixed_mean_macc': [float(np.mean(macc_mean)), float(np.std(macc_mean, ddof=1))],
                  'fixed_mean_cost': [float(np.mean(cost_mean)), float(np.std(cost_mean, ddof=1))],
                  'dynamic_minus_fixed_mean': stats.bootstrap(np.mean(diffs, 0), rng)}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / 'fixed_subset_mean.json', 'w'), indent=2)
