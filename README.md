# COVA-MTL

**COVA-MTL: Counterfactual Value-Aware Modality Acquisition for Efficient and Robust Multi-Task Driving Perception**

COVA-MTL selects, for each sample, the modalities to encode before the high-cost branch encoders are executed,
and predicts the four AIDE tasks (DER, DBR, TCR, VBR) from the selected subset.

- **ACVE** (Availability-Conditioned Counterfactual Value Estimator) predicts, from lightweight previews and the
  availability state, the task-wise loss change of every valid subset relative to the full available configuration.
- **SSR** (Self-Referenced Subset Router) selects the minimum-cardinality subset among the candidates predicted not
  to increase the mean loss over the four tasks.
- **SCML** (Subset-Conditioned Multi-Task Learner) executes only the selected branch encoders and handles variable
  configurations with subset-conditioned calibration and mask-aware task-specific fusion.

## Repository layout

| Folder | Contents |
|---|---|
| `main/` | AIDE pipeline: Teacher, 31-subset cache, ACVE, calibration, Student, and evaluation |
| `cross_dataset/` | the same pipeline for NTHU-DDD, StateFarm, and HDD |
| `ablations/` | ablation programs on AIDE |
| `analysis/` | figures |
| `experiments/` | analyses that replay routing on stored checkpoints and cached outputs, and efficiency measurements |
| `splits/aide/` | clip lists of the AIDE splits with split seeds 42, 43, and 44 |
| `configs/` | saved argument records of the reported runs |

## Programs and results in the paper

| Paper | Program |
|---|---|
| Tables 4–7 and A10, COVA-MTL values in Figure 3 | `main/train_multiseed.py` |
| Table 8 and Tables A4–A9 | `cross_dataset/nthu_ddd.py`, `cross_dataset/statefarm.py`, `cross_dataset/hdd.py` |
| Tables 9–11 | `experiments/routing_analysis.py` |
| Table 12 | `experiments/latency_breakdown.py` |
| Table 13 | `ablations/main_components.py` |
| Table 14 | `ablations/acve_value_representation.py` |
| Table 15 | `ablations/ssr_components.py` |
| Table 16 | `ablations/scml_internal.py` (the Full row is the Full run of Table 13) |
| Table A2 | `ablations/task_composition.py` |
| Table A3 | `ablations/scml_encoder.py` |
| Table A11 | `experiments/policy_selector_comparison.py` (policies in `policy_selector.py`), `fixed_subset_selection.py`, `route_statistics.py`, `all_fixed_subsets.py`, `fixed_subset_mean.py` |
| Table A10, mean of the same-cardinality subsets | `experiments/matched_cardinality_mean.py` |
| Table A12 | `experiments/run_split_seeds.sh` |
| Table A13 | `experiments/corruption_robustness.py` |
| Section 3.5, calibrated upper-bound gates | `experiments/ssr_upper_gate.py` |
| Section 4.4, paired bootstrap confidence interval | `experiments/route_statistics.py` |
| Section 4.4, FLOPs, parameters, peak GPU memory, GPU utilization | `experiments/compute_profile.py`, `experiments/gpu_utilization.py` |
| Section 4.6, mean of the same-cardinality subsets on HDD | `experiments/external_fixed_subset_mean.py` |
| Sections 4.7 and 4.8, selected-route coverage and loss increase | `experiments/route_statistics.py` |
| Section 4.9, end-to-end latency and branch-cost objective | `experiments/end_to_end_latency.py`, `experiments/routing_analysis.py`, `experiments/hdd_cost_objective.py` |
| Figure 4 | `analysis/final_visualization.py` |
| Figure 5 | `analysis/qualitative_selection.py` |

## Names in the code

The code keeps the names used during development. They correspond to the paper as follows.

| Paper | Code |
|---|---|
| ACVE | `AvailabilityConditionedSufficientRouter` ("router", ACSR in comments); stages `router`, `calibrate`, `recalibrate` |
| SSR | `select_self_referenced_subset` (SRCR in comments) |
| SCML | `COVAMTLSCMC`; subset-conditioned calibration is `SubsetConditionedModalityCalibration` (SCMC), mask-aware fusion is `MaskAwareTSMF` |
| Lightweight previews | `MultiStreamScout` ("scout") |
| Dense inference | all physically available branches (`forward_dense`) |
| Dynamic inference | ACVE and SSR followed by the selected branches (`route_from_raw`) |

## Data

The datasets are not included and are subject to their own licenses. By default the programs read
`data/AIDE_Dataset`, `data/NTHU-DDD/original`, `data/StateFarm`, and `data/hdd_data`; pass `--data-root` to use
another location. The programs in `experiments/` read the AIDE location from `AIDE_ROOT` and the HDD location
from `HDD_ROOT`.

The AIDE split is a clip-level 64:16:20 split. `splits/aide/seed42` is the split of all main experiments, and
`seed43` and `seed44` are the additional splits of Table A12. Each file lists the clip IDs in the order produced
by the programs, which regenerate the same split from `--split-seed` when the split CSV files are missing.

## Environment

Python 3.10, PyTorch 2.5.1 with CUDA 12.1, and an NVIDIA GeForce RTX 4090 (Table 12 was measured on an RTX 4080).

```bash
pip install -r requirements.txt
```

## Reproduction

All settings of the reported runs are the defaults of the programs; `configs/` contains the saved argument
records. Seeds 42–44 are reported. Results are written to `RUNS/` (or `COVA_RUNS`).

```bash
# AIDE: Teacher, cache, ACVE, calibration, Student, and evaluation for seeds 42-44
python main/train_multiseed.py --stage all --data-root data/AIDE_Dataset --seeds 42,43,44

# Additional AIDE splits (Table A12)
bash experiments/run_split_seeds.sh

# External datasets
python cross_dataset/nthu_ddd.py --stage all --data-root data/NTHU-DDD/original --seeds 42,43,44
python cross_dataset/statefarm.py --stage all --data-root data/StateFarm --seeds 42,43,44
python cross_dataset/hdd.py --stage all --data-root data/hdd_data --seeds 42,43,44

# Ablations (seed 42), for example
python ablations/main_components.py --data-root data/AIDE_Dataset
```

The programs in `experiments/` use the AIDE runs above and do not retrain any model. Run them in this order:

```bash
python experiments/build_student_cache.py          # cache the Student outputs on the 31 subsets
python experiments/routing_analysis.py
python experiments/ssr_upper_gate.py
python experiments/fixed_subset_selection.py
python experiments/route_statistics.py
python experiments/policy_selector_comparison.py
python experiments/all_fixed_subsets.py
python experiments/fixed_subset_mean.py
python experiments/matched_cardinality_mean.py
python experiments/latency_breakdown.py
python experiments/end_to_end_latency.py
python experiments/compute_profile.py
python experiments/gpu_utilization.py
python experiments/corruption_robustness.py
python experiments/hdd_cost_objective.py           # requires the HDD run
python experiments/external_fixed_subset_mean.py   # requires the three external-dataset runs
```

Trained checkpoints are not included.
