"""Dynamic inference versus the fixed subsets with the matched cardinality on the external datasets (Section 4.6).

Reads final_test.json of seeds 42-44 written by cross_dataset/{hdd,nthu_ddd,statefarm}.py. For each seed,
the fixed subsets whose cardinality equals the rounded validation cardinality of Dynamic inference are taken
from the exhaustive subset evaluation, and their mean and median test scores are compared with Dynamic
inference. The primary score is Joint mAP for HDD and mAcc for NTHU-DDD and StateFarm.
"""
import json
import os
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("COVA_RUNS", REPO / "RUNS"))
OUT = RUNS_ROOT / "analysis"
CONFIG = {"HDD": ("HDD_COVA_v6_3_MultiSeed5", "joint_score"),
          "NTHU-DDD": ("NTHU_DDD_COVA_v6_3_MultiSeed5_Fix", "macc"),
          "StateFarm": ("STATEFARM_COVA_v6_3_MultiSeed5", "macc")}


def main():
    out = {}
    for name, (run, key) in CONFIG.items():
        rows = []
        for seed in (42, 43, 44):
            result = json.load(open(RUNS_ROOT / run / f"seed_{seed}" / "final_test.json"))
            matched = result["matched_cardinality_best_fixed_subset"]
            k = matched["rounded_validation_dynamic_cardinality"]
            by_subset = result["exhaustive_nonempty_subsets"]["by_subset"]
            values = [v[key] for v in by_subset.values() if v["cardinality"] == k]
            dynamic = result["dynamic"][key]
            rows.append({"seed": seed, "cardinality": k, "n_subsets": len(values), "dynamic": 100 * dynamic,
                         "best_validation_selected": 100 * matched["test_" + key],
                         "mean": 100 * st.mean(values), "median": 100 * st.median(values),
                         "subsets_below_dynamic": sum(v < dynamic for v in values)})
        out[name] = rows
        col = lambda c: [r[c] for r in rows]
        print(f"{name}: Dynamic {st.mean(col('dynamic')):.2f}, matched-cardinality mean "
              f"{st.mean(col('mean')):.2f} +- {st.stdev(col('mean')):.2f}, median {st.mean(col('median')):.2f}, "
              f"Dynamic - mean {st.mean(col('dynamic')) - st.mean(col('mean')):+.2f}, "
              f"Dynamic - median {st.mean(col('dynamic')) - st.mean(col('median')):+.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "external_fixed_subset_mean.json", "w"), indent=1)


if __name__ == "__main__":
    main()
