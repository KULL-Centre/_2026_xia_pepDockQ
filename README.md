# pepDockQ

**Graph-based quality estimation for peptide–protein complexes.**

`pepDockQ` takes a docked peptide–protein PDB structure as input and predicts
a scalar quality score correlated with the DockQ metric of the underlying
interface. Predictions are made by a heterogeneous graph neural network (HGNN)
over per-residue features that combine ESM-2 sequence embeddings, ESM-IF
inverse-folding mutation scores, side-chain geometry, and Z-scale
physico-chemical descriptors.

Given a folder of docked PDB files, the pipeline:

1. Extracts per-chain sequences from each PDB and computes **ESM-IF**
   inverse-folding features.
2. Computes **ESM-2** per-chain embeddings.
3. Builds a peptide/protein heterogeneous graph and packages all per-residue
   features into a `torch_geometric` `HeteroData` object (one `.pt` file
   per PDB).
4. Loads the trained HGNN checkpoint and writes a CSV of predicted scores.

---

## Requirements

- **Python** 3.9
- **PyTorch** 2.4.0 + CUDA 12.1
- **torch-geometric** wheels for `pt24cu121`
- A CUDA-capable GPU for the embedding and inference stages

All exact pins are in [`requirements.txt`](requirements.txt).

---

## Installation

```bash
conda create -n pepdockq python=3.9 -y
conda activate pepdockq

# 1. Runtime dependencies (PyTorch cu121 + PyG accelerator wheels).
pip install -r requirements.txt

# 2. The pepDockQ package itself (editable install for development).
pip install -e .
```

The extra index URLs for the `+cu121` PyTorch build and the matching
`torch-geometric` accelerator wheels are declared at the top of
[`requirements.txt`](requirements.txt), so a single `pip install` resolves the
full stack.

### Fetch the released weights

`assets/model.pth` is **not** stored in the repository — it lives on Zenodo.
Fetch it with:

```bash
scripts/download_weights.sh          # writes assets/model.pth
```

The script verifies the SHA-256 of the downloaded file against the expected
value baked into it, so you can safely re-run it.

### Verify the installation

```bash
python -c "import pepdockq; print(pepdockq.__version__)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected: 2.4.0+cu121 12.1 True
```

---

## Inference — run the full pipeline

The four inference stages are wired together by
[`scripts/run_pipeline.sh`](scripts/run_pipeline.sh). It is a plain shell
script — no Slurm dependency — and runs each stage in-process via
`python -m pepdockq.<...>`.

| Step | Module | Where it runs |
|------|--------|---------------|
| 1    | [`pepdockq.features.esmif`](pepdockq/features/esmif.py) | GPU |
| 2    | [`pepdockq.features.esm`](pepdockq/features/esm.py)     | GPU |
| 3    | [`pepdockq.features.graph`](pepdockq/features/graph.py) | CPU |
| 4    | [`pepdockq.predict`](pepdockq/predict.py)               | GPU |

You must provide `PDB_DIR`, `WORK_DIR`, `OUTPUT`, `NCPUS`, `CHECKPOINT` and
`ZSCALE_PATH`. All intermediate paths (`fasta`, `info`, `esm_embedding`,
`esmif_embedding`, `features`, `graph`, `log`) are created automatically as
subfolders of `WORK_DIR`.

```bash
PDB_DIR=/path/to/pdb \
WORK_DIR=/path/to/workdir \
OUTPUT=/path/to/results.csv \
NCPUS=4 \
CHECKPOINT=$(pwd)/assets/model.pth \
ZSCALE_PATH=$(pwd)/assets/amino_acid_zscale.csv \
scripts/run_pipeline.sh
```

| Variable      | Required | Description                                                     |
| ------------- | -------- | --------------------------------------------------------------- |
| `PDB_DIR`     | yes      | Input folder of docked PDB files                                |
| `WORK_DIR`    | yes      | Working directory; all intermediate features go in subfolders   |
| `OUTPUT`      | yes      | Path to the final prediction CSV                                |
| `NCPUS`       | yes      | Number of worker processes for step 3                           |
| `CHECKPOINT`  | yes      | Path to the model checkpoint (`.pth`)                           |
| `ZSCALE_PATH` | yes      | Path to the amino-acid Z-scale CSV file                         |
| `PYTHON`      | no       | Python interpreter to use (defaults to `python`)                |

The final CSV has two columns:

```csv
decoy_id,pred_score
1abc_model_0,0.712
1abc_model_1,0.483
...
```

A worked 30-second reproduction is in [`examples/`](examples/README.md).

### Running one stage at a time

Each stage is a standalone Python entry point — invoke it with `python -m`:

```bash
python -m pepdockq.features.esmif  --pdb_dir ... --fasta_dir ... --info_dir ... --esmif_dir ...
python -m pepdockq.features.esm    --fasta_dir ... --esm_dir ...
python -m pepdockq.features.graph  --pdb_dir ... --esm_dir ... --esmif_dir ... \
                                    --info_dir ... --feature_dir ... --zscale_path ... --ncpus 4
python -m pepdockq.predict         --data_path ... --checkpoint ... --output ...
```

After `pip install -e .`, `pepdockq-predict` and `pepdockq-train` are also
registered as console scripts.

---

## Training

`pepdockq/train.py` reproduces the training procedure of the released
checkpoint.

**Prerequisites.** Before training you need the pre-computed graph features
for every complex in your training / validation / test splits. Run steps 1–3
of the inference pipeline once on all PDBs you plan to train on. This
produces one `.pt` file per structure under `<feature_dir>/graph`, which is
the `--data_path` argument below.

**Split files.** Their schema is documented in
[`docs/data_format.md`](docs/data_format.md). Example split files ship with
the repo under [`data/`](data/README.md).

```bash
python -m pepdockq.train \
    --data_path  /path/to/features/graph \
    --train_list data/train.txt \
    --valid_list data/valid.tsv \
    --test_list  data/test.tsv \
    --output_dir outputs \
    --epochs 6 \
    --batch_size 32 \
    --lr 5e-5
```

Each run writes a timestamped subdirectory under `--output_dir` containing:

- `config.json` — a dump of all CLI arguments
- `log.txt` — mirrored stdout
- `checkpoints/model_epoch{N}.pth` — one checkpoint per epoch
- `checkpoints/prediction_epoch{N}.pkl` — per-complex predictions on valid + test
- TensorBoard event files (`tensorboard --logdir outputs/<timestamp>`)

The training objective (see [`pepdockq/models/losses.py`](pepdockq/models/losses.py)):

- **`DockQPairwiseThresholdLoss`** — Huber regression on the global DockQ
  target plus a threshold-aware pairwise ranking term.
- **`EdgeBinnedSoftTargetLoss`** — soft-target cross-entropy over binned
  per-edge distance errors (32 bins over `[-5 Å, 30 Å]`), providing an
  auxiliary local supervision signal on the interface geometry.

Full documentation of the architecture and the feature stack is in
[`docs/architecture.md`](docs/architecture.md).

---

## Repository layout

```
pepDockQ/
├── README.md · LICENSE · CITATION.cff · pyproject.toml · requirements.txt
│
├── assets/
│   ├── amino_acid_zscale.csv          static Z-scale descriptors
│   └── model.pth                       ← fetched by scripts/download_weights.sh
│
├── data/                               training / valid / test split files
│   ├── train.txt
│   ├── valid.tsv
│   └── test.tsv
│
├── docs/
│   ├── architecture.md
│   ├── data_format.md
│   └── figures/
│
├── examples/                           30-second end-to-end reproduction
│   ├── inputs/
│   └── expected_outputs/
│
├── scripts/
│   ├── run_pipeline.sh                 end-to-end inference driver
│   └── download_weights.sh             fetch model.pth from Zenodo
│
└── pepdockq/                           the installable Python package
    ├── __init__.py
    ├── features/                       PDB → HeteroData
    │   ├── esm.py                      step 2: ESM-2 embeddings
    │   ├── esmif.py                    step 1: ESM-IF features
    │   └── graph.py                    step 3: graph construction
    ├── models/
    │   ├── hgnn.py                     HGNN architecture
    │   └── losses.py                   training losses
    ├── data.py                         training-time datasets & pair samplers
    ├── metrics.py                      evaluation metrics
    ├── predict.py                      step 4: inference → CSV
    └── train.py                        training entry point
```

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation

If you use `pepDockQ` in your work, please cite:

```bibtex
@software{pepdockq,
  title  = {pepDockQ: Graph-based quality estimation for peptide-protein complexes},
  author = {<authors>},
  year   = {2025},
  url    = {https://github.com/<org>/pepDockQ},
}
```

*(Publication citation will be added once the manuscript is out.)*
