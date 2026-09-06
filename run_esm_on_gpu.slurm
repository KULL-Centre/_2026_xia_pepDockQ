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

# Usage: sbatch run_esm_on_gpu.slurm PDB_DIR FASTA_DIR INFO_DIR ESMIF_DIR ESM_DIR
# All arguments are required (provided by run_pipeline.sh).
PDB_DIR=$1
FASTA_DIR=$2
INFO_DIR=$3
ESMIF_DIR=$4
ESM_DIR=$5

"$PYTHON" run_esmif.py \
    --pdb_dir "$PDB_DIR" \
    --fasta_dir "$FASTA_DIR" \
    --info_dir "$INFO_DIR" \
    --esmif_dir "$ESMIF_DIR"

"$PYTHON" run_esm.py \
    --fasta_dir "$FASTA_DIR" \
    --esm_dir "$ESM_DIR"
