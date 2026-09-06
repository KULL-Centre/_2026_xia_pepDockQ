#!/bin/bash
# =============================================================================
# pepDockQ end-to-end inference pipeline
#
# Runs the four inference stages sequentially, on the local machine:
#   1. features.esmif   — ESM-IF inverse-folding features        (GPU)
#   2. features.esm     — ESM-2 per-chain embeddings              (GPU)
#   3. features.graph   — build HeteroData graph .pt files        (CPU)
#   4. predict          — HGNN forward → CSV                      (GPU)
#
# The script is run from anywhere; it locates the repo root from its own path
# and invokes the package via `python -m pepdockq.<...>`. If you are on Slurm,
# submit this script inside an `srun` / `sbatch` allocation instead of using
# stage-specific batch scripts.
#
# Usage:
#   PDB_DIR=... WORK_DIR=... OUTPUT=... NCPUS=... \
#   CHECKPOINT=... ZSCALE_PATH=... [PYTHON=...] scripts/run_pipeline.sh
#
# Required environment variables:
#   PDB_DIR      input folder of PDB files
#   WORK_DIR     working dir; all intermediate features live in its subfolders
#   OUTPUT       path to the final prediction CSV
#   NCPUS        number of worker processes for step 3
#   CHECKPOINT   path to the model checkpoint (.pth)
#   ZSCALE_PATH  path to the amino-acid z-scale CSV
#
# Optional:
#   PYTHON       python interpreter to use (defaults to `python`)
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
CONFIG=${CONFIG:-configs/predict.env}
if [[ -f "$CONFIG" ]]; then
    echo "[PIPELINE] loading config: $CONFIG"
    # shellcheck disable=SC1090
    source "$CONFIG"
else
    echo "[PIPELINE] config file not found: $CONFIG (falling back to environment only)"
fi

# -----------------------------------------------------------------------------
# Required parameters
# -----------------------------------------------------------------------------
: "${PDB_DIR:?Set PDB_DIR (input folder of PDB files)}"
: "${WORK_DIR:?Set WORK_DIR (working dir for intermediate features)}"
: "${OUTPUT:?Set OUTPUT (path to the final prediction CSV)}"
: "${NCPUS:?Set NCPUS (number of worker processes for step 3)}"
: "${CHECKPOINT:?Set CHECKPOINT (path to the model checkpoint .pth)}"
: "${ZSCALE_PATH:?Set ZSCALE_PATH (path to the amino acid z-scale CSV)}"
PYTHON=${PYTHON:-python}

# -----------------------------------------------------------------------------
# Derived paths (stored under WORK_DIR)
# -----------------------------------------------------------------------------
FASTA_DIR=$WORK_DIR/fasta
INFO_DIR=$WORK_DIR/info
ESMIF_DIR=$WORK_DIR/esmif_embedding
ESM_DIR=$WORK_DIR/esm_embedding
FEATURE_DIR=$WORK_DIR/features
GRAPH_DIR=$FEATURE_DIR/graph
LOG_DIR=$WORK_DIR/log
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Step 1 — ESM-IF inverse-folding features (GPU)
# -----------------------------------------------------------------------------
echo "[PIPELINE] Step 1: running features.esmif ..."
"$PYTHON" -m pepdockq.features.esmif \
    --pdb_dir   "$PDB_DIR" \
    --fasta_dir "$FASTA_DIR" \
    --info_dir  "$INFO_DIR" \
    --esmif_dir "$ESMIF_DIR" \
    2>&1 | tee "$LOG_DIR/step1_esmif.log"
echo "[PIPELINE] Step 1 finished."

# -----------------------------------------------------------------------------
# Step 2 — ESM-2 per-chain embeddings (GPU)
# -----------------------------------------------------------------------------
echo "[PIPELINE] Step 2: running features.esm ..."
"$PYTHON" -m pepdockq.features.esm \
    --fasta_dir "$FASTA_DIR" \
    --esm_dir   "$ESM_DIR" \
    2>&1 | tee "$LOG_DIR/step2_esm.log"
echo "[PIPELINE] Step 2 finished."

# -----------------------------------------------------------------------------
# Step 3 — build graph features (CPU, multi-process)
# -----------------------------------------------------------------------------
echo "[PIPELINE] Step 3: running features.graph ..."
"$PYTHON" -m pepdockq.features.graph \
    --pdb_dir     "$PDB_DIR" \
    --esm_dir     "$ESM_DIR" \
    --esmif_dir   "$ESMIF_DIR" \
    --info_dir    "$INFO_DIR" \
    --feature_dir "$FEATURE_DIR" \
    --zscale_path "$ZSCALE_PATH" \
    --ncpus       "$NCPUS" \
    2>&1 | tee "$LOG_DIR/step3_graph.log"
echo "[PIPELINE] Step 3 finished."

# -----------------------------------------------------------------------------
# Step 4 — HGNN prediction (GPU)
# -----------------------------------------------------------------------------
echo "[PIPELINE] Step 4: running predict ..."
"$PYTHON" -m pepdockq.predict \
    --data_path  "$GRAPH_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUTPUT" \
    2>&1 | tee "$LOG_DIR/step4_predict.log"
echo "[PIPELINE] Step 4 finished. Predictions written to: $OUTPUT"
