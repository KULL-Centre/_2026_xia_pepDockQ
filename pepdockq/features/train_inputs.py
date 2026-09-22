import csv
import os


ROLE_COLUMNS = {"pdb_id", "receptor_chains", "peptide_chains"}
SIMILARITY_COLUMN = "max_pdb_receptor_seq_identity"


def split_chain_ids(value):
    return [chain_id.strip() for chain_id in value.split("_") if chain_id.strip()]


def read_training_inputs(manifest_paths, pdb_dir):
    """Read all training split manifests and validate their common contract."""
    inputs = {}
    for manifest_path in manifest_paths:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not ROLE_COLUMNS.issubset(reader.fieldnames):
                raise ValueError(
                    f"{manifest_path} must contain columns: pdb_id,receptor_chains,peptide_chains"
                )
            for row_number, row in enumerate(reader, start=2):
                pdb_id = row["pdb_id"].strip()
                receptor_chains = split_chain_ids(row["receptor_chains"])
                peptide_chains = split_chain_ids(row["peptide_chains"])
                if not pdb_id or not receptor_chains or not peptide_chains:
                    raise ValueError(
                        f"{manifest_path}:{row_number} requires pdb_id, receptor_chains, and peptide_chains"
                    )
                if pdb_id in inputs:
                    raise ValueError(f"{manifest_path}:{row_number} duplicates pdb_id '{pdb_id}'")
                if os.path.basename(pdb_id) != pdb_id:
                    raise ValueError(f"{manifest_path}:{row_number} pdb_id must not be a path")
                overlap = set(receptor_chains) & set(peptide_chains)
                if overlap:
                    raise ValueError(
                        f"{manifest_path}:{row_number} assigns chain(s) {sorted(overlap)} to both receptor and peptide"
                    )
                complex_dir = os.path.join(pdb_dir, pdb_id)
                if not os.path.isdir(complex_dir):
                    raise FileNotFoundError(
                        f"{manifest_path}:{row_number} PDB directory not found: {complex_dir}"
                    )
                inputs[pdb_id] = {
                    "receptor": receptor_chains,
                    "peptide": peptide_chains,
                    "manifest_path": manifest_path,
                    "row_number": row_number,
                }
    if not inputs:
        raise ValueError("Training manifests do not contain any PDB IDs")
    return inputs


def read_split_pdb_ids(manifest_path, require_similarity=False):
    """Read a train/validation/test split for model training."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = set(ROLE_COLUMNS)
        if require_similarity:
            required_columns.add(SIMILARITY_COLUMN)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            required = ",".join(sorted(required_columns))
            raise ValueError(f"{manifest_path} must contain columns: {required}")
        return list(reader)