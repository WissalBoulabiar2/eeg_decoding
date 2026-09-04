#!/bin/bash
set -e
# Full method comparison suite: ER, MIR, GMED, Sequential (no-memory
# MUDVI baseline), MUDVI+GP+RSD, OCAR, OCAR++ -- each run in both the
# canonical subject order and one random order (via run_subject_order.py
# --num_orders 1) -- plus one Joint (offline upper bound) run, which has
# no subject-order dependence to sweep since it pools every subject's
# data at once (see joint_baseline.py).
#
# Dataset-agnostic: works for --dataset bci2a or high_gamma, set via env
# vars so this script is not tied to one dataset/machine (same
# no-hardcoded-paths convention as run_stage1.sh).
#
# Usage (High Gamma):
#   DATASET=high_gamma DATA_DIR=/kaggle/working/hgd/data \
#     SUBJECTS=1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
#     OUT_DIR=/kaggle/working/results ./run_full_suite.sh
#
# Usage (BCI IV-2a, once its .gdf files are available, e.g. from a
# Kaggle Dataset attached to the notebook):
#   DATASET=bci2a DATA_DIR=/kaggle/input/bci-iv-2a-gdf \
#     SUBJECTS=01,02,03,04,05,06,07,08,09 \
#     OUT_DIR=/kaggle/working/results ./run_full_suite.sh
: "${DATASET:?Set DATASET to bci2a or high_gamma}"
: "${DATA_DIR:?Set DATA_DIR to the dataset folder}"
: "${SUBJECTS:?Set SUBJECTS, e.g. 1,2,...,14 (high_gamma) or 01,...,09 (bci2a)}"
OUT_DIR="${OUT_DIR:-results}"
EPOCHS_PER_SUBJECT="${EPOCHS_PER_SUBJECT:-15}"
MEMORY_SIZE="${MEMORY_SIZE:-200}"
SEED="${SEED:-0}"
ORDER_SEED="${ORDER_SEED:-0}"

COMMON="--dataset $DATASET --data_dir $DATA_DIR --subjects $SUBJECTS --epochs_per_subject $EPOCHS_PER_SUBJECT --memory_size $MEMORY_SIZE --seed $SEED --out_dir $OUT_DIR"
ORDER_COMMON="$COMMON --num_orders 1 --order_seed $ORDER_SEED"

echo "=== [1/8] Sequential (no-memory MUDVI baseline), canonical + random order ==="
python -m mudvi_baseline.run_subject_order --method mudvi --memory_size 0 \
    --dataset $DATASET --data_dir $DATA_DIR --subjects $SUBJECTS \
    --epochs_per_subject $EPOCHS_PER_SUBJECT --seed $SEED --out_dir $OUT_DIR \
    --num_orders 1 --order_seed $ORDER_SEED

echo "=== [2/8] ER, canonical + random order ==="
python -m mudvi_baseline.run_subject_order --method er $ORDER_COMMON

echo "=== [3/8] MIR, canonical + random order ==="
python -m mudvi_baseline.run_subject_order --method mir $ORDER_COMMON

echo "=== [4/8] GMED, canonical + random order ==="
python -m mudvi_baseline.run_subject_order --method gmed $ORDER_COMMON

echo "=== [5/8] MUDVI + GP + RSD, canonical + random order ==="
python -m mudvi_baseline.run_subject_order --method mudvi \
    --gradient_projection --relationship_shift_detection $ORDER_COMMON

echo "=== [6/8] OCAR, canonical + random order ==="
python -m mudvi_baseline.run_subject_order --method mudvi --ocar $ORDER_COMMON

echo "=== [7/8] OCAR++, canonical + random order ==="
python -m mudvi_baseline.run_subject_order --method mudvi --ocar_plusplus $ORDER_COMMON

echo "=== [8/8] Joint training (offline upper bound, single run, no order sweep) ==="
python -m mudvi_baseline.run_experiment --method joint $COMMON

echo "=== FULL SUITE COMPLETE: results under $OUT_DIR ==="
