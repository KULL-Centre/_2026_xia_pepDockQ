import argparse
import glob
import json
import os
import pickle
import tempfile
from tqdm import tqdm

from pepdockq.features.esmif import (
    load_esmif_model,
    load_structure,
    load_residue_info,
    write_fasta,
    ext_complex_reps,
    ext_single_chain_rep,
    modified_occupancy,
)
from pepdockq.features.train_inputs import read_training_inputs


def _find_native_pdb(complex_dir, pdb_id):
    candidates = [
        os.path.join(complex_dir, f"{pdb_id}.pdb"),
        os.path.join(complex_dir, "complex.pdb"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Native PDB not found for {pdb_id}: expected {candidates}")


def _validate_chain_roles(pdb_id, sequences, chain_roles):
    requested_chains = chain_roles["receptor"] + chain_roles["peptide"]
    missing_chains = [chain_id for chain_id in requested_chains if chain_id not in sequences]
    if missing_chains:
        raise ValueError(
            f"{pdb_id}: requested chain(s) {missing_chains} are absent; available chains: {sorted(sequences)}"
        )


def extract_train_esmif(pdb_dir, fasta_dir, info_dir, esmif_dir, input_csvs):
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(esmif_dir, exist_ok=True)

    manifest_inputs = read_training_inputs(input_csvs, pdb_dir)
    complex_ids = sorted(manifest_inputs)
    print(f"[INFO] Found {len(complex_ids)} training structures in split manifests")

    # Validate native PDB chains before allocating the ESM-IF model on the GPU.
    for complex_id in complex_ids:
        native_pdb = _find_native_pdb(os.path.join(pdb_dir, complex_id), complex_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            prep_pdb = os.path.join(tmpdir, "prep_native.pdb")
            modified_occupancy(native_pdb, prep_pdb)
            _, sequences = load_residue_info(load_structure(prep_pdb), complex_id)
        _validate_chain_roles(complex_id, sequences, manifest_inputs[complex_id])

    model, alphabet = load_esmif_model()

    for complex_id in tqdm(complex_ids, desc="ESM-IF features (native + decoys)"):
        complex_dir = os.path.join(pdb_dir, complex_id)
        native_pdb = _find_native_pdb(complex_dir, complex_id)
        chain_roles = manifest_inputs[complex_id]

        fasta_path = os.path.join(fasta_dir, f"{complex_id}.fasta")
        info_path = os.path.join(info_dir, f"{complex_id}.json")
        decoy_info_path = os.path.join(info_dir, f"{complex_id}_decoy.json")

        # 1. Process native complex structure to extract sequences, residue info, and chain roles.
        with tempfile.TemporaryDirectory() as tmpdir:
            prep_pdb = os.path.join(tmpdir, "prep_native.pdb")
            modified_occupancy(native_pdb, prep_pdb)
            structure = load_structure(prep_pdb)
            json_data, sequences = load_residue_info(structure, complex_id)
        _validate_chain_roles(complex_id, sequences, chain_roles)
        json_data["chain_roles"] = {
            "receptor": chain_roles["receptor"],
            "peptide": chain_roles["peptide"],
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        write_fasta(sequences, fasta_path, complex_id)
        chain_ids = list(sequences.keys())

        # 2. Process decoy structures under decoys/
        decoy_dir = os.path.join(complex_dir, "decoys")
        decoy_files = sorted(glob.glob(os.path.join(decoy_dir, "*.pdb"))) if os.path.isdir(decoy_dir) else []

        # Extract decoy-specific residue info (e.g. for res_idx mapping in graph build)
        if decoy_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                prep_decoy = os.path.join(tmpdir, "prep_decoy_sample.pdb")
                modified_occupancy(decoy_files[0], prep_decoy)
                decoy_json_data, decoy_sequences = load_residue_info(
                    load_structure(prep_decoy), f"{complex_id}_decoy"
                )
            _validate_chain_roles(f"{complex_id} decoy", decoy_sequences, chain_roles)
            decoy_json_data["chain_roles"] = {
                "receptor": chain_roles["receptor"],
                "peptide": chain_roles["peptide"],
            }
            with open(decoy_info_path, "w", encoding="utf-8") as f:
                json.dump(decoy_json_data, f, indent=2)

        # Extract ESM-IF embeddings for all decoy structures under decoys/
        for decoy_pdb in decoy_files:
            decoy_name = os.path.splitext(os.path.basename(decoy_pdb))[0]
            esmif_out = os.path.join(esmif_dir, f"{decoy_name}.pkl")
            if os.path.exists(esmif_out):
                continue
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    prep_decoy = os.path.join(tmpdir, "prep_decoy.pdb")
                    modified_occupancy(decoy_pdb, prep_decoy)
                    complex_reps = ext_complex_reps(model, alphabet, prep_decoy, chain_ids, sequences)
                    monomer_reps = {
                        cid: ext_single_chain_rep(model, alphabet, prep_decoy, cid, sequences)
                        for cid in chain_ids
                    }
                with open(esmif_out, "wb") as f:
                    pickle.dump({"multimer": complex_reps, "monomer": monomer_reps}, f)
            except Exception as e:
                print(f"[WARN] Error extracting ESM-IF for {decoy_pdb}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-IF features for training complexes and their decoys.")
    parser.add_argument("--pdb_dir", required=True, help="Input directory containing per-complex folders.")
    parser.add_argument("--fasta_dir", required=True, help="Output directory for FASTA files.")
    parser.add_argument("--info_dir", required=True, help="Output directory for residue info JSON files.")
    parser.add_argument("--esmif_dir", required=True, help="Output directory for ESM-IF .pkl files.")
    parser.add_argument("--input_csvs", required=True, nargs="+", help="Train, validation, and test split CSV files.")
    args = parser.parse_args()

    extract_train_esmif(
        pdb_dir=args.pdb_dir,
        fasta_dir=args.fasta_dir,
        info_dir=args.info_dir,
        esmif_dir=args.esmif_dir,
        input_csvs=args.input_csvs,
    )


if __name__ == "__main__":
    main()
