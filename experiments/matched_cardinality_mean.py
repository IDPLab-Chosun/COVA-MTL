import os
"""Dynamic inference versus the mean of the fixed subsets with the matched cardinality (Table A10).

For each seed, all fixed subsets with the rounded validation cardinality of Dynamic inference are compared
on the test split, with a paired clip-level bootstrap confidence interval.
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
MATCHED = {42: 4, 43: 3, 44: 4}
res, diffs, means = {}, [], []
for seed in stats.SEEDS:
    s = stats.Seed(cova, seed, dev); nt = s.test['labels'].size(0)
    dch, *_ = s.route(s.test, FULL); dyn_scores = stats.clip_correct(s.test, dch).mean(1).numpy(); dyn = float(dyn_scores.mean())
    k = MATCHED[seed]; subs = [c for c in range(31) if int(routing.CARD[c]) == k]
    sc = [stats.clip_correct(s.test, torch.full((nt,), c)).mean(1).numpy() for c in subs]
    m = [float(x.mean()) for x in sc]
    fixed_mean_scores = np.mean(sc, 0); diffs.append(dyn_scores - fixed_mean_scores); means.append(100 * fixed_mean_scores.mean())
    res[seed] = {'cardinality': k, 'n_subsets': len(subs), 'dynamic': 100 * dyn, 'mean': 100 * float(np.mean(m)),
                 'median': 100 * float(np.median(m)), 'n_beaten': int(sum(x < dyn for x in m)),
                 'subsets': {stats.name_of(c): 100 * v for c, v in zip(subs, m)}}
    print(seed, {kk: vv for kk, vv in res[seed].items() if kk != 'subsets'}, flush=True)
rng = np.random.default_rng(0)
res['summary'] = {'mean': [float(np.mean(means)), float(np.std(means, ddof=1))],
                  'dynamic_minus_mean': stats.bootstrap(np.mean(diffs, 0), rng)}
print(json.dumps(res['summary'], indent=1))
json.dump(res, open(OUT / 'matched_cardinality_mean.json', 'w'), indent=2)
