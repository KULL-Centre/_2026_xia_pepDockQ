#!/bin/bash
# =============================================================================
# pepDockQ end-to-end training pipeline
#
# Runs the five training stages sequentially:
#   1. features.esmif_train — ESM-IF features for native and decoys    (GPU)
#   2. features.esm         — ESM-2 per-chain sequence embeddings      (GPU)
#   3. features.dockq       — Compute ground-truth DockQ scores        (CPU)
#   4. features.graph_train — Build training HeteroData graph .pt files (CPU)
#   5. train                — Train HGNN quality estimation model      (GPU)
#
# The script is run from anywhere; it locates the repo root from its own path
# and invokes the package via `python -m pepdockq.<...>`.
#
# Usage:
#   PDB_DIR=... WORK_DIR=... TRAIN_LIST=... VALID_LIST=... TEST_LIST=... \
#   ZSCALE_PATH=... NCPUS=... [PYTHON=...] scripts/run_training.sh [extra train args]
#
# Required environment variables:
#   PDB_DIR      input folder containing per-complex subdirectories (e.g. train_data/pdb)
#   WORK_DIR     working dir; all intermediate features live in its subfolders
#   TRAIN_LIST   CSV file with pdb_id, receptor_chains, peptide_chains
#   VALID_LIST   CSV validation split, plus max_pdb_receptor_seq_identity
#   TEST_LIST    CSV test split, same schema as VALID_LIST
#   ZSCALE_PATH  path to the amino-acid z-scale CSV
#   NCPUS        number of worker processes for step 4
#
# Optional:
#   OUTPUT_DIR   root directory for experiment outputs (default: outputs)
#   EPOCHS       number of training epochs             (default: 6)
#   BATCH_SIZE   pairs per batch                       (default: 32)
#   NUM_WORKERS  DataLoader worker processes           (default: 8)
#   LR           Adam learning rate                    (default: 5e-5)
#   DEVICE       torch device                          (default: cuda)
#   PYTHON       python interpreter to use             (default: python)
# =============================================================================
set -euo pipefail

# Locate the repo root (parent of this script's directory) and cd there so
# `python -m pepdockq.…` resolves regardless of where the user invoked us from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# Config file (dotenv-style). Every entry inside uses `${VAR:=default}` so any
# variable already set in the caller's environment wins over the config.
# -----------------------------------------------------------------------------
CONFIG=${CONFIG:-configs/train.env}
if [[ -f "$CONFIG" ]]; then
    echo "[TRAIN-PIPELINE] loading config: $CONFIG"
    # shellcheck disable=SC1090
    source "$CONFIG"
else
    echo "[TRAIN-PIPELINE] config file not found: $CONFIG (falling back to environment only)"
fi

# -----------------------------------------------------------------------------
# Required parameters
# -----------------------------------------------------------------------------
: "${PDB_DIR:?Set PDB_DIR (input folder of complex PDB subdirs, e.g. train_data/pdb)}"
: "${WORK_DIR:?Set WORK_DIR (working dir for intermediate features)}"
: "${TRAIN_LIST:?Set TRAIN_LIST (training split list file)}"
: "${VALID_LIST:?Set VALID_LIST (validation split TSV file)}"
: "${TEST_LIST:?Set TEST_LIST (test split TSV file)}"
: "${ZSCALE_PATH:?Set ZSCALE_PATH (path to amino acid z-scale CSV)}"
: "${NCPUS:?Set NCPUS (number of worker processes for step 4)}"
PYTHON=${PYTHON:-python}

# -----------------------------------------------------------------------------
# Optional parameters (with defaults matching pepdockq.train)
# -----------------------------------------------------------------------------
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
EPOCHS=${EPOCHS:-6}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-5e-5}
DEVICE=${DEVICE:-cuda}

# -----------------------------------------------------------------------------
# Derived paths (stored under WORK_DIR, consistent with run_prediction.sh)
# -----------------------------------------------------------------------------
FASTA_DIR=$WORK_DIR/fasta
INFO_DIR=$WORK_DIR/info
ESMIF_DIR=$WORK_DIR/esmif_embedding
ESM_DIR=$WORK_DIR/esm_embedding
DOCKQ_DIR=$WORK_DIR/dockq
FEATURE_DIR=$WORK_DIR/features
GRAPH_DIR=$FEATURE_DIR/graph
LOG_DIR=$WORK_DIR/log
mkdir -p "$LOG_DIR" "$DOCKQ_DIR"

# -----------------------------------------------------------------------------
# Step 1 — ESM-IF inverse-folding features (GPU)
# -----------------------------------------------------------------------------
echo "[TRAIN-PIPELINE] Step 1: running features.esmif_train ..."
"$PYTHON" -m pepdockq.features.esmif_train \
    --pdb_dir   "$PDB_DIR" \
    --fasta_dir "$FASTA_DIR" \
    --info_dir  "$INFO_DIR" \
    --esmif_dir "$ESMIF_DIR" \
    --input_csvs "$TRAIN_LIST" "$VALID_LIST" "$TEST_LIST" \
    2>&1 | tee "$LOG_DIR/step1_esmif.log"
echo "[TRAIN-PIPELINE] Step 1 finished."

# -----------------------------------------------------------------------------
# Step 2 — ESM-2 per-chain embeddings (GPU)
# -----------------------------------------------------------------------------
echo "[TRAIN-PIPELINE] Step 2: running features.esm ..."
"$PYTHON" -m pepdockq.features.esm \
    --fasta_dir "$FASTA_DIR" \
    --esm_dir   "$ESM_DIR" \
    --input_csv "$TRAIN_LIST" "$VALID_LIST" "$TEST_LIST" \
    2>&1 | tee "$LOG_DIR/step2_esm.log"
echo "[TRAIN-PIPELINE] Step 2 finished."

# -----------------------------------------------------------------------------
# Step 3 — Compute ground-truth DockQ scores (CPU)
# -----------------------------------------------------------------------------
echo "[TRAIN-PIPELINE] Step 3: running features.dockq ..."
"$PYTHON" -m pepdockq.features.dockq \
    --pdb_dir     "$PDB_DIR" \
    --output_dir  "$DOCKQ_DIR" \
    --input_csvs  "$TRAIN_LIST" "$VALID_LIST" "$TEST_LIST" \
    2>&1 | tee "$LOG_DIR/step3_dockq.log"
echo "[TRAIN-PIPELINE] Step 3 finished."

# -----------------------------------------------------------------------------
# Step 4 — Build training HeteroData graph features (CPU, multi-process)
# -----------------------------------------------------------------------------
echo "[TRAIN-PIPELINE] Step 4: running features.graph_train ..."
"$PYTHON" -m pepdockq.features.graph_train \
    --decoy_dir   "$PDB_DIR" \
    --info_dir    "$INFO_DIR" \
    --feature_dir "$FEATURE_DIR" \
    --dockq_dir   "$DOCKQ_DIR" \
    --zscale_path "$ZSCALE_PATH" \
    --esm_dir     "$ESM_DIR" \
    --esmif_dir   "$ESMIF_DIR" \
    --ncpus       "$NCPUS" \
    --input_csvs  "$TRAIN_LIST" "$VALID_LIST" "$TEST_LIST" \
    2>&1 | tee "$LOG_DIR/step4_graph.log"
echo "[TRAIN-PIPELINE] Step 4 finished."

# -----------------------------------------------------------------------------
# Step 5 — HGNN model training (GPU)
# -----------------------------------------------------------------------------
echo "[TRAIN-PIPELINE] Step 5: launching pepdockq.train ..."
"$PYTHON" -m pepdockq.train \
    --data_path   "$GRAPH_DIR" \
    --train_list  "$TRAIN_LIST" \
    --valid_list  "$VALID_LIST" \
    --test_list   "$TEST_LIST" \
    --output_dir  "$OUTPUT_DIR" \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr          "$LR" \
    --device      "$DEVICE" \
    "$@" \
    2>&1 | tee "$LOG_DIR/step5_train.log"

echo "[TRAIN-PIPELINE] Step 5 finished. Outputs under: $OUTPUT_DIR/"
