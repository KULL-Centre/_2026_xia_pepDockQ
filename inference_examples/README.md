# Examples — 30-second reproduction

This directory contains a **minimal frozen test case** for pepDockQ: one or
two small peptide–protein complexes, the expected prediction CSV they produce
with the released checkpoint, and enough documentation to reproduce it in a
single shell command.

## Layout

```
examples/
├── inputs/
│   └── *.pdb                    ← input PDB files (peptide + protein per file)
└── expected_outputs/
    └── predictions.csv          ← frozen output produced by scripts/run_pipeline.sh
                                    at the released commit + weights
```

## How to run it

From the repository root:

```bash
# 1. Fetch the released weights (writes assets/model.pth). One-time setup.
scripts/download_weights.sh

# 2. Run the pipeline against the fixture PDBs.
PDB_DIR=$(pwd)/examples/inputs \
WORK_DIR=/tmp/pepdockq_example \
OUTPUT=/tmp/pepdockq_example/predictions.csv \
NCPUS=2 \
CHECKPOINT=$(pwd)/assets/model.pth \
ZSCALE_PATH=$(pwd)/assets/amino_acid_zscale.csv \
scripts/run_pipeline.sh

# 3. Compare against the frozen reference.
diff /tmp/pepdockq_example/predictions.csv examples/expected_outputs/predictions.csv
```

Small numerical differences (< 0.05 in the `pred_score` column) are expected
across GPUs and CUDA versions.

## Choosing the fixture PDBs

<!-- TODO(user): drop 1–2 small public PDBs here and remove this notice. -->

The `inputs/` directory is intentionally empty at first commit — the maintainer
must:

1. Pick 1–2 small public peptide–protein complexes from the RCSB PDB (or your
   own docked models). Anything under ~5 kB works and keeps the repo lean.
2. If the file was produced by AlphaFold / docking, keep the version used for
   the released benchmark so numbers are traceable.
3. Freeze the corresponding `predictions.csv` produced by `scripts/run_pipeline.sh`
   at the release commit and check it into `expected_outputs/`.
4. Record the PDB provenance (source URL, release date, license) in a short
   `inputs/PROVENANCE.md`.
