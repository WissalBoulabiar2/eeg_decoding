#!/bin/bash
set -e
# Configurable via env var so this script is not tied to one machine's
# local path (Lightning AI migration, Section 10 -- "do not hard-code
# local paths"). Set BCICIV_DATA_DIR before running, e.g.:
#   BCICIV_DATA_DIR=/data/BCICIV_2a_gdf ./run_stage1.sh
: "${BCICIV_DATA_DIR:?Set BCICIV_DATA_DIR to the folder containing A0kT.gdf files}"
DATA_DIR="$BCICIV_DATA_DIR"
COMMON="--data_dir $DATA_DIR --subjects 01,02,03 --epochs_per_subject 3 --seed 0"

echo "=== [1/4] Baseline ==="
python -m mudvi_baseline.run_baseline $COMMON --out mudvi_baseline/stage1_baseline.json
echo "=== [2/4] GP only ==="
python -m mudvi_baseline.run_baseline $COMMON --gradient_projection --out mudvi_baseline/stage1_gp.json
echo "=== [3/4] RSD only ==="
python -m mudvi_baseline.run_baseline $COMMON --relationship_shift_detection --out mudvi_baseline/stage1_rsd.json
echo "=== [4/4] GP + RSD ==="
python -m mudvi_baseline.run_baseline $COMMON --gradient_projection --relationship_shift_detection --out mudvi_baseline/stage1_gp_rsd.json
echo "=== ALL STAGE 1 RUNS COMPLETE ==="
