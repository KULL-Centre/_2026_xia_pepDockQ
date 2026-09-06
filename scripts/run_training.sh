#!/bin/bash
# =============================================================================
# pepDockQ training runner
#
# Launches `python -m pepdockq.train` with the paths for graph features and
# the train / valid / test split files. train.py itself creates a timestamped
# subdirectory under OUTPUT_DIR containing config.json, log.txt, checkpoints
# and TensorBoard event files, so this script does not tee its own log.
#
# Usage:
#   DATA_PATH=... TRAIN_LIST=... VALID_LIST=... TEST_LIST=... \
#   [OUTPUT_DIR=outputs] [EPOCHS=6] [BATCH_SIZE=32] [NUM_WORKERS=8] \
#   [LR=5e-5] [DEVICE=cuda] [PYTHON=python] \
#   scripts/run_training.sh  [extra args passed through to pepdockq.train]
#
# Required environment variables:
#   DATA_PATH    directory of pre-computed graph .pt features
#                (typically <feature_dir>/graph, output of features.graph)
#   TRAIN_LIST   text file with one training PDB id per line
#   VALID_LIST   TSV file for validation split (see docs/data_format.md)
#   TEST_LIST    TSV file for test split, same schema as VALID_LIST
#
# Optional:
#   OUTPUT_DIR   root directory for experiment outputs   (default: outputs)
#   EPOCHS       number of training epochs               (default: 6)
#   BATCH_SIZE   pairs per batch                         (default: 32)
#   NUM_WORKERS  DataLoader worker processes             (default: 8)
#   LR           Adam learning rate                      (default: 5e-5)
#   DEVICE       torch device                            (default: cuda)
#   PYTHON       python interpreter                      (default: python)
#
# Any additional arguments after the script name are forwarded to
# pepdockq.train, e.g.:
#   EPOCHS=20 ... scripts/run_training.sh --hidden_channels 128 --lambda_pair 0.5
# =============================================================================
set -euo pipefail

# Locate the repo root (parent of this script's directory) and cd there so
# `python -m pepdockq.train` resolves regardless of where the user invoked us
# from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# Config file (dotenv-style). Every entry inside uses `${VAR:=default}` so any
# variable already set in the caller's environment wins over the config.
# -----------------------------------------------------------------------------
CONFIG=${CONFIG:-configs/train.env}
if [[ -f "$CONFIG" ]]; then
    echo "[TRAIN] loading config: $CONFIG"
    # shellcheck disable=SC1090
    source "$CONFIG"
else
    echo "[TRAIN] config file not found: $CONFIG (falling back to environment only)"
fi

# -----------------------------------------------------------------------------
# Required parameters
# -----------------------------------------------------------------------------
: "${DATA_PATH:?Set DATA_PATH (dir of pre-computed graph .pt features)}"
: "${TRAIN_LIST:?Set TRAIN_LIST (text file with one PDB id per line)}"
: "${VALID_LIST:?Set VALID_LIST (validation split TSV)}"
: "${TEST_LIST:?Set TEST_LIST (test split TSV)}"

# -----------------------------------------------------------------------------
# Optional parameters (with defaults matching pepdockq.train)
# -----------------------------------------------------------------------------
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
EPOCHS=${EPOCHS:-6}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-5e-5}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

mkdir -p "$OUTPUT_DIR"

echo "[TRAIN] launching pepdockq.train ..."
"$PYTHON" -m pepdockq.train \
    --data_path   "$DATA_PATH" \
    --train_list  "$TRAIN_LIST" \
    --valid_list  "$VALID_LIST" \
    --test_list   "$TEST_LIST" \
    --output_dir  "$OUTPUT_DIR" \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr          "$LR" \
    --device      "$DEVICE" \
    "$@"
echo "[TRAIN] finished. Outputs under: $OUTPUT_DIR/<timestamp>/"
