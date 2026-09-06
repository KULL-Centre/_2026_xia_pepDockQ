#!/bin/bash
# =============================================================================
# PepQA pipeline driver
#
# Defines all parameters in one place and runs the three stages in order:
#   1. run_esm_on_gpu.slurm        (ESM-IF + ESM embeddings, GPU)
#   2. collect_inputs.py           (build graph features, CPU)
#   3. run_prediction_on_gpu.slurm (prediction, GPU)
#
# The GPU stages are submitted with `sbatch --wait`, so each one fully finishes
# before the next stage starts.
#
# Usage:
#   PDB_DIR=... WORK_DIR=... OUTPUT=... NCPUS=... ./run_pipeline.sh
#
# Required (must be provided):
#   PDB_DIR      input folder of PDB files
#   WORK_DIR     working dir; all intermediate features live in its subfolders
#   OUTPUT       path to the final prediction CSV
#   NCPUS        number of worker processes for step 3
#   CHECKPOINT   path to the model checkpoint (.pth)
#   ZSCALE_PATH  path to the amino acid z-scale CSV file
#   PYTHON       path to the conda env python (exported to the Slurm steps)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# Required parameters
# -----------------------------------------------------------------------------
: "${PDB_DIR:?Set PDB_DIR (input folder of PDB files)}"
: "${WORK_DIR:?Set WORK_DIR (working dir for intermediate features)}"
: "${OUTPUT:?Set OUTPUT (path to the final prediction CSV)}"
: "${NCPUS:?Set NCPUS (number of worker processes for step 3)}"
: "${CHECKPOINT:?Set CHECKPOINT (path to the model checkpoint .pth)}"
: "${ZSCALE_PATH:?Set ZSCALE_PATH (path to the amino acid z-scale CSV)}"
: "${PYTHON:?Set PYTHON (path to the conda env python)}"

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

# Slurm log files (%j is replaced by the job id). Directory is created if missing.
STEP12_OUT=${STEP12_OUT:-$LOG_DIR/step1_2_%j.log}
STEP12_ERR=${STEP12_ERR:-$LOG_DIR/step1_2_%j.err}
STEP4_OUT=${STEP4_OUT:-$LOG_DIR/step4_%j.log}
STEP4_ERR=${STEP4_ERR:-$LOG_DIR/step4_%j.err}
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Step 1 & 2 — ESM-IF + ESM embeddings (GPU, blocking)
# -----------------------------------------------------------------------------
echo "[PIPELINE] Step 1&2: submitting run_esm_on_gpu.slurm ..."
sbatch --wait --export=ALL,PYTHON="$PYTHON" --output="$STEP12_OUT" --error="$STEP12_ERR" run_esm_on_gpu.slurm \
    "$PDB_DIR" "$FASTA_DIR" "$INFO_DIR" "$ESMIF_DIR" "$ESM_DIR"
echo "[PIPELINE] Step 1&2 finished."

# -----------------------------------------------------------------------------
# Step 3 — collect inputs / build graph features (CPU)
# -----------------------------------------------------------------------------
echo "[PIPELINE] Step 3: running collect_inputs.py ..."
"$PYTHON" collect_inputs.py \
    --pdb_dir     "$PDB_DIR" \
    --esm_dir     "$ESM_DIR" \
    --esmif_dir   "$ESMIF_DIR" \
    --info_dir    "$INFO_DIR" \
    --feature_dir "$FEATURE_DIR" \
    --zscale_path "$ZSCALE_PATH" \
    --ncpus       "$NCPUS"
echo "[PIPELINE] Step 3 finished."

# -----------------------------------------------------------------------------
# Step 4 — prediction (GPU, blocking)
# -----------------------------------------------------------------------------
echo "[PIPELINE] Step 4: submitting run_prediction_on_gpu.slurm ..."
sbatch --wait --export=ALL,PYTHON="$PYTHON" --output="$STEP4_OUT" --error="$STEP4_ERR" run_prediction_on_gpu.slurm \
    "$GRAPH_DIR" "$CHECKPOINT" "$OUTPUT"
echo "[PIPELINE] Step 4 finished. Predictions written to: $OUTPUT"
