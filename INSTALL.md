# PepPIScore Pipeline — Installation Guide

- **Python:** 3.9.25
- **PyTorch:** 2.4.0 + CUDA 12.1
- **PyG (torch-geometric) wheels:** built for `pt24cu121`

## 1. Initialize the conda environment

```bash
conda create -n peppiscore python=3.9 -y
conda activate peppiscore
```

## 2. Install all Python packages

`requirements.txt` pins the full stack — including `torch==2.4.0+cu121` and the
matching PyG wheels (`torch-scatter`, `torch-sparse`, `torch-cluster`,
`torch-spline-conv`, `pyg-lib`). The PyTorch and PyG wheel sources are declared at
the top of the file, so a single command installs everything:

```bash
pip install -r requirements.txt
```

## 3. Verify the installation

```bash
conda activate peppiscore
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected: 2.4.0+cu121 12.1 True

python -c "import torch, esm, torch_geometric, biotite; print('all imports OK')"
```

---

## 4. Run the pipeline

The whole pipeline (step 1&2 → step 3 → step 4) is driven by `run_pipeline.sh`.
You must provide `PDB_DIR`, `WORK_DIR`, `OUTPUT`, `NCPUS`, `CHECKPOINT` and
`ZSCALE_PATH`. All other paths (`fasta`, `info`, `esm_embedding`,
`esmif_embedding`, `features`, `graph`) are stored automatically in the
corresponding subfolders of `WORK_DIR`. The GPU stages are submitted with
`sbatch --wait`, so each stage finishes before the next one starts.

```bash

PDB_DIR=/path/to/pdb \
WORK_DIR=/path/to/workdir \
OUTPUT=/path/to/results.csv \
NCPUS=4 \
CHECKPOINT=/path/to/model.pth \
ZSCALE_PATH=/path/to/amino_acid_zscale.csv \
PYTHON=/path/to/conda/envs/peppiscore/bin/python \
./run_pipeline.sh
```

| Variable      | Required | Description                                             |
| ------------- | -------- | ------------------------------------------------------- |
| `PDB_DIR`     | yes      | Input folder of PDB files                               |
| `WORK_DIR`    | yes      | Working dir; all intermediate features go in subfolders |
| `OUTPUT`      | yes      | Path to the final prediction CSV                        |
| `NCPUS`       | yes      | Number of worker processes for step 3                   |
| `CHECKPOINT`  | yes      | Path to the model checkpoint (.pth)                     |
| `ZSCALE_PATH` | yes      | Path to the amino acid z-scale CSV file                 |
| `PYTHON`      | yes      | Path to the conda env python. Exported to the Slurm steps. |

### Example (sbxw-3 dataset)

```bash
PDB_DIR=/novo/projects/departments/cdd/molecular_ai/yiui/workspace/PepQA/dataset/sbxw/sbxw-3/peppro_structures_cat_pdbfixer \
WORK_DIR=/novo/projects/departments/cdd/molecular_ai/yiui/workspace/PepQA/dataset/sbxw/sbxw-3 \
OUTPUT=/novo/projects/departments/cdd/molecular_ai/yiui/workspace/PepQA/dataset/sbxw/sbxw-3/prediction_results.csv \
NCPUS=4 \
CHECKPOINT=$(pwd)/model.pth \
ZSCALE_PATH=$(pwd)/amino_acid_zscale.csv \
PYTHON=/novo/users/yiui/.conda/envs/peppiscore/bin/python \
./run_pipeline.sh
```
