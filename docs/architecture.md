# pepDockQ architecture

<!-- TODO(user): replace every "REPLACE_ME" / "TODO" block below with the
     content from the manuscript. This page is the one that reviewers will
     read alongside the code, so keep it precise and mostly cross-links to
     specific functions/classes rather than paraphrasing them. -->

## Overview

pepDockQ estimates the DockQ score of a docked peptide–protein complex from
its 3D structure and sequence, using a heterogeneous graph neural network
(HGNN) over per-residue features.

![Model overview](figures/overview.png)

<!-- TODO(user): add figures/overview.png — the same schematic as Figure 1
     of the manuscript, exported at ~1200 px wide. -->

## Feature stack

Each residue node carries four feature groups:

| Feature | Source | Dim | Provenance in code |
|---|---|---|---|
| ESM-2 embedding             | `esm2_t33_650M_UR50D` per-chain forward | 1280 | `pepdockq/features/esm.py` |
| ESM-IF mutation score       | Inverse-folding logits, wildtype logit  | 1    | `pepdockq/features/esmif.py` |
| Backbone / side-chain geometry | Dihedrals + Cα / SC positions        | 6    | `pepdockq/features/graph.py` |
| Z-scale physico-chemical    | Static lookup table                     | 5    | `assets/amino_acid_zscale.csv` |

The concrete tensor keys on the `HeteroData` object are `esm`, `zscale`,
`dihedrals`, and `esmif_mutation_score`; keep these names in sync between
`pepdockq/features/graph.py` (writer) and `pepdockq/models/hgnn.py` (reader).

## Graph construction

<!-- TODO(user): a short paragraph on the radius cutoff (default 10 Å in
     `pepdockq/features/graph.py:GraphPepFeature.edge_th`), the four edge
     types (`peptide-peptide`, `protein-protein`, `peptide-protein`,
     `protein-peptide`), and the representative-atom choice
     (`rep_atom = 'sc'`). Cite the section in the manuscript. -->

## HGNN

Defined in `pepdockq/models/hgnn.py`. Key layers:

- `HGNNlayer` — a per-edge-type message-passing block with multi-head
  attention, edge-attribute update, and stochastic depth on the residual.
- `HGNN` — stacks `num_layers` of `HGNNlayer`, followed by global mean +
  max pool per node type and an MLP head that outputs the scalar
  quality score.

## Training objective

<!-- TODO(user): match the wording used in the manuscript. -->

Sum of:

- `DockQPairwiseThresholdLoss` — Huber regression on the global DockQ target
  plus a threshold-aware pairwise ranking term (weight `lambda_pair`).
- `EdgeBinnedSoftTargetLoss` — soft-target cross-entropy over 32 bins of
  per-edge distance error in `[-5 Å, 30 Å]`.

Both are in `pepdockq/models/losses.py`.
