#!/bin/bash
#SBATCH --job-name=peppiscore

#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --constraint=g6-4xlarge
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --comment="cost"

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

# Usage: sbatch run_prediction_on_gpu.slurm DATA_PATH CHECKPOINT OUTPUT
# All arguments are required (provided by run_pipeline.sh).
DATA_PATH=$1
CHECKPOINT=$2
OUTPUT=$3

"$PYTHON" run_prediction.py \
    --data_path "$DATA_PATH" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT"