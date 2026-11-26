#!/bin/bash
# Appendix A.7: retrain the full AIDE pipeline on the splits generated with split seeds 43 and 44
# (model seed = split seed). Split seed 42 is the main multi-seed run. All other settings are the
# defaults of main/train_multiseed.py; the generated splits equal splits/aide/seed43 and seed44.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
AIDE_ROOT=${AIDE_ROOT:-data/AIDE_Dataset}
for S in 43 44; do
  $PY main/train_multiseed.py --no-multi-seed --seed $S --split-seed $S --stage all \
    --data-root "$AIDE_ROOT" --run-dir RUNS/split_seeds/split${S}_seed${S}
done
