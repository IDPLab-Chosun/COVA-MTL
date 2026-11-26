import os
"""Test mAcc and branch cost of every fixed subset and of Dynamic inference under full availability.

Seeds 42-44, cached Student outputs. Used for the number of fixed subsets within the validation budget
whose mAcc Dynamic inference exceeds (Table A11).
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
out = {}
for seed in stats.SEEDS:
    s = stats.Seed(cova, seed, dev)
    nv, nt = s.val['labels'].size(0), s.test['labels'].size(0)
    dch, *_ = s.route(s.test, FULL); vch, *_ = s.route(s.val, FULL)
    dyn = stats.macc(s.test, dch); budget = stats.cost_of(vch, stats.avail(nv, FULL), routing.COSTS)
    rows = []
    for c in range(31):
        rows.append({'subset': stats.name_of(c), 'test_macc': stats.macc(s.test, torch.full((nt,), c)),
                     'val_macc': stats.macc(s.val, torch.full((nv,), c)),
                     'cost': stats.cost_of(torch.full((nt,), c), stats.avail(nt, FULL), routing.COSTS)})
    inb = [r for r in rows if r['cost'] <= budget + 1e-9]
    out[seed] = {'dynamic': dyn, 'dynamic_cost': stats.cost_of(dch, stats.avail(nt, FULL), routing.COSTS), 'budget': budget,
                 'n_beaten_all': sum(r['test_macc'] < dyn for r in rows), 'n_all': 31,
                 'n_beaten_in_budget': sum(r['test_macc'] < dyn for r in inb), 'n_in_budget': len(inb),
                 'mean_in_budget': float(np.mean([r['test_macc'] for r in inb])),
                 'median_in_budget': float(np.median([r['test_macc'] for r in inb])),
                 'mean_all': float(np.mean([r['test_macc'] for r in rows])), 'rows': rows}
    print(seed, {k: v for k, v in out[seed].items() if k != 'rows'}, flush=True)
json.dump(out, open(OUT / 'all_fixed_subsets.json', 'w'), indent=2)
