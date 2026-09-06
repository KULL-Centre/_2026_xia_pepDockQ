# Input & split file formats

This page documents the on-disk layout the pipeline expects. All paths are
relative to the caller-provided `PDB_DIR` / `WORK_DIR` / split-file arguments
described in the top-level [README](../README.md).

## Input PDB files

- One `.pdb` file per docked complex, placed directly under `PDB_DIR`.
- Each file must contain both the peptide and the receptor protein, with
  distinct chain identifiers.
- **TODO(user):** document the chain-ID convention actually enforced by
  `pepdockq/features/graph.py` (peptide chain must be the shortest?
  labeled `B`?). Cross-check `GraphPepFeature.get_chain_list` and fill in.
- Maximum peptide length is **53 residues** (`MAX_PEPTIDE_LENGTH` in
  `pepdockq/features/graph.py`). Longer peptides are silently skipped.
- Non-standard residues MSE and SEP are mapped to M and S respectively; all
  other non-standard residues are rejected.

## Intermediate `WORK_DIR` layout

After `scripts/run_pipeline.sh` finishes, `WORK_DIR` looks like:

```
WORK_DIR/
├── fasta/                       # step 1 output — per-PDB FASTA
│   └── <pdb_id>.fasta
├── info/                        # step 1 output — per-residue metadata JSON
│   └── <pdb_id>.json
├── esmif_embedding/             # step 1 output — pickled per-chain ESM-IF tensors
│   └── <pdb_id>.pkl
├── esm_embedding/               # step 2 output — pickled per-chain ESM-2 tensors
│   └── <pdb_id>.pkl
├── features/
│   ├── coordinate/              # step 3 intermediate — Cα / SC coordinates
│   ├── distance_map/            # step 3 intermediate — pairwise distances
│   └── graph/                   # step 3 final output — HeteroData .pt files
│       └── <pdb_id>.pt
└── log/                         # per-stage stdout/stderr from run_pipeline.sh
```

Only `features/graph/` is consumed by the prediction step (step 4); the other
subfolders are safe to delete afterwards if disk space is tight.

## Training split files

`pepdockq/train.py` takes three split files:

### `--train_list train.txt`

One PDB id per line, one entry per training complex. Each PDB id must have a
matching `<pdb_id>.pt` file under `--data_path` (= `features/graph/`).

```
1abc_model_0
1abc_model_1
2def_model_0
...
```

### `--valid_list` / `--test_list`

Tab-separated files with (at minimum) these columns:

| Column | Type | Description |
|---|---|---|
| `pep_id` | str | Complex identifier — matches `<pdb_id>.pt` under `--data_path` |
| `max_pdb_receptor_seq_identity`    | float in [0, 1] | Max receptor-side identity to any complex in the PDB (used for stratified evaluation) |

Evaluation is stratified into three bins on `max_pdb_receptor_seq_identity`:
`[0.0, 0.4)`, `[0.4, 0.8)`, `[0.8, 1.0]`. Extra columns are ignored.

Example split TSVs live in [`data/`](../data/):

- `data/train.txt`
- `data/valid.tsv`
- `data/test.tsv`

**TODO(user):** if the released splits derive from a specific public dataset
(e.g. Boltz-2's peptide split, a PPI benchmark), record the source, download
URL, and any filtering criteria in a `data/README.md`.
