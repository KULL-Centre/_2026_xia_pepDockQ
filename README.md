# pepDockQ

**Graph-based quality estimation for peptide–protein complexes.**

`pepDockQ` takes a docked peptide–protein PDB structure as input and predicts
a scalar quality score correlated with the DockQ metric of the underlying
interface. Predictions are made by a heterogeneous graph neural network (HGNN)
over per-residue features that combine ESM-2 sequence embeddings, ESM-IF
inverse-folding mutation scores, side-chain geometry, and Z-scale
physico-chemical descriptors.

![pepDockQ overview](docs/figures/pepDockQ.png)

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

pip install -r requirements.txt   # PyTorch cu121 + PyG accelerator wheels
pip install -e .                   # the pepDockQ package (editable)
```

The extra index URLs for the `+cu121` PyTorch build and matching
`torch-geometric` accelerator wheels are declared at the top of
[`requirements.txt`](requirements.txt), so a single `pip install` resolves the
full stack.

---

## Inference

The four inference stages are wired together by
[`scripts/run_prediction.sh`](scripts/run_prediction.sh) and configured by
[`configs/predict.env`](configs/predict.env). Every knob has a default that
points at the frozen example under [`examples/`](examples/) and the released
checkpoint under [`assets/`](assets/), so out of the box:

```bash
bash scripts/run_prediction.sh
```

runs the two PDB files under [`examples/pdb/`](examples/pdb/) end-to-end and
writes predictions to `examples/prediction_results.csv`.

| Step | Module | Where it runs |
|------|--------|---------------|
| 1    | [`pepdockq.features.esmif`](pepdockq/features/esmif.py) | GPU |
| 2    | [`pepdockq.features.esm`](pepdockq/features/esm.py)     | GPU |
| 3    | [`pepdockq.features.graph`](pepdockq/features/graph.py) | CPU |
| 4    | [`pepdockq.predict`](pepdockq/predict.py)               | GPU |

To predict on your own data, edit [`configs/predict.env`](configs/predict.env)
— it exposes the input folder, working directory, output CSV path, checkpoint,
Z-scale asset, and stage-3 worker count. Any variable already set in the
environment overrides the config, so a one-off run without editing the file
looks like:

```bash
PDB_DIR=/path/to/pdbs OUTPUT=/tmp/pred.csv bash scripts/run_prediction.sh
```

Intermediate features (`fasta`, `info`, `esm_embedding`, `esmif_embedding`,
`features`, `log`) are created as subfolders of `WORK_DIR`.

The final CSV has two columns:

```csv
decoy_id,pred_score
1abc_model_0,0.712
1abc_model_1,0.483
```

### Running one stage at a time

Each stage is a standalone entry point:

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

Training is driven by [`scripts/run_training.sh`](scripts/run_training.sh),
configured by [`configs/train.env`](configs/train.env). Unlike the old workflow,
the training pipeline now runs the full feature-generation stack in one place:

1. `pepdockq.features.esmif_train` — extracts native + decoy residue metadata,
   writes per-complex FASTA files, and computes ESM-IF embeddings for decoys.
2. `pepdockq.features.esm` — computes ESM-2 per-chain embeddings.
3. `pepdockq.features.dockq` — computes ground-truth DockQ values for each
   native/decoy pair.
4. `pepdockq.features.graph_train` — builds HeteroData training graphs and
   writes `.pt` files under `<WORK_DIR>/features/graph`.
5. `pepdockq.train` — trains the HGNN with the generated graph dataset.

This matches the style of the inference script: all intermediate artifacts live
under a shared working directory, with `fasta`, `info`, `esmif_embedding`,
`esm_embedding`, `dockq`, `features`, and `log` created automatically.

For the bundled demo dataset, the defaults in [`configs/train.env`](configs/train.env)
point at:

```bash
PDB_DIR=train_data/pdb
TRAIN_LIST=train_data/split/train.txt
VALID_LIST=train_data/split/valid.tsv
TEST_LIST=train_data/split/test.tsv
WORK_DIR=train_data/work
OUTPUT_DIR=outputs
ZSCALE_PATH=assets/amino_acid_zscale.csv
```

A typical training run is therefore:

```bash
bash scripts/run_training.sh
```

The repo includes a native/decoy example at `train_data/pdb/1a0n_A`, where:

- `train_data/pdb/1a0n_A/1a0n_A.pdb` is the native structure;
- `train_data/pdb/1a0n_A/decoys/*.pdb` contains the decoy structures.

Because native PDB residue indices are not guaranteed to start at 1 or be
contiguous, residue metadata is written separately for each source:

- `<WORK_DIR>/info/1a0n_A.json` — native structure residue mapping
- `<WORK_DIR>/info/1a0n_A_decoy.json` — decoy residue mapping

These files are consumed by the graph builder so that native and decoy
coordinates are matched by the correct residue index set instead of assuming the
same numbering scheme.

Any environment variable already set in the shell overrides the config, and any
extra CLI arguments are forwarded to `pepdockq.train`:

```bash
PDB_DIR=/path/to/pdbs WORK_DIR=/tmp/pepdockq_run bash scripts/run_training.sh
```

Each run writes a timestamped subdirectory under `OUTPUT_DIR` containing:

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
│   └── model.pth                       ← downloaded manually from Zenodo
│
├── configs/
│   ├── predict.env                     inference pipeline config
│   └── train.env                       training pipeline config
│
├── data/                               example split files for training and validation
│   ├── train.txt
│   ├── valid.tsv
│   └── test.tsv
│
├── docs/
│   ├── architecture.md
│   ├── data_format.md
│   └── figures/
│
├── examples/                           frozen inference demo
│   ├── pdb/
│   └── prediction_results.csv
│
├── outputs/                            training experiment outputs (timestamped folders)
│
├── scripts/
│   ├── run_prediction.sh               end-to-end inference driver
│   └── run_training.sh                 end-to-end training driver
│
├── train_data/                         demo training data for native + decoy complexes
│   ├── pdb/
│   │   ├── 1a08_C/
│   │   ├── 1a0n_A/
│   │   └── ...
│   ├── split/
│   │   ├── train.txt
│   │   ├── valid.tsv
│   │   └── test.tsv
│   └── work/                          generated intermediate features
│       ├── fasta/
│       ├── info/
│       ├── esmif_embedding/
│       ├── esm_embedding/
│       ├── dockq/
│       ├── features/
│       │   ├── coordinate/
│       │   ├── distance_map/
│       │   └── graph/
│       └── log/
│
└── pepdockq/                           the installable Python package
    ├── __init__.py
    ├── data.py                         training dataset utilities
    ├── metrics.py                      evaluation metrics
    ├── predict.py                      inference entry point
    ├── train.py                        training entry point
    ├── features/                       per-stage feature-generation modules
    │   ├── __init__.py
    │   ├── esm.py                      step 2: ESM-2 embeddings
    │   ├── esmif.py                    step 1: ESM-IF utilities
    │   ├── esmif_train.py              step 1: native/decoy training feature extraction
    │   ├── dockq.py                    step 3: DockQ ground-truth computation
    │   ├── graph.py                    inference graph builder
    │   ├── graph_train.py              training graph builder with native/decoy residue maps
    │   └── ...
    └── models/
        ├── hgnn.py                     HGNN architecture
        ├── losses.py                   training losses
        └── __init__.py
```
---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation

Publication forthcoming. Citation metadata is tracked in
[`CITATION.cff`](CITATION.cff).
