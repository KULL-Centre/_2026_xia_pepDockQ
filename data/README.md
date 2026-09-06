# Training splits

Split files consumed by `pepdockq/train.py`. See
[`docs/data_format.md`](../docs/data_format.md) for the schema of each file.

| File | Used as |
|---|---|
| `train.txt`                        | `--train_list` |
| `valid.tsv`                        | `--valid_list` |
| `test.tsv`                         | `--test_list`  |

## Provenance

<!-- TODO(user): document where each split came from — source dataset,
     release date, filtering criteria, redundancy removal, etc. — so a
     reader of the paper can reproduce the split assignment. -->

- Source dataset: **REPLACE_ME**
- Redundancy filter: **REPLACE_ME**  (e.g. MMseqs2 30% seq identity on receptor)
- Number of complexes: train=REPLACE_ME · valid=REPLACE_ME · test=REPLACE_ME
- Split date / cutoff: **REPLACE_ME**

## Getting the graph features (`.pt` files)

The `.pt` files that the split entries refer to are **not** in the repository —
they are the per-complex output of the feature-extraction stages of the
inference pipeline. To regenerate them for all splits at once:

```bash
PDB_DIR=/path/to/all_pdbs \
WORK_DIR=/path/to/features_workdir \
OUTPUT=/tmp/discard.csv \
NCPUS=8 \
CHECKPOINT=$(pwd)/assets/model.pth \
ZSCALE_PATH=$(pwd)/assets/amino_acid_zscale.csv \
scripts/run_pipeline.sh
```

Then point `pepdockq/train.py --data_path` at `${WORK_DIR}/features/graph`.
