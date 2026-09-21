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


def extract_train_esmif(pdb_dir, fasta_dir, info_dir, esmif_dir):
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(esmif_dir, exist_ok=True)

    model, alphabet = load_esmif_model()
    complex_ids = sorted([d for d in os.listdir(pdb_dir) if os.path.isdir(os.path.join(pdb_dir, d))])
    print(f"[INFO] Found {len(complex_ids)} complex directories in {pdb_dir}")

    for complex_id in tqdm(complex_ids, desc="ESM-IF features (native + decoys)"):
        complex_dir = os.path.join(pdb_dir, complex_id)
        native_pdb = os.path.join(complex_dir, f"{complex_id}.pdb")
        if not os.path.exists(native_pdb):
            native_pdb = os.path.join(complex_dir, "complex.pdb")

        fasta_path = os.path.join(fasta_dir, f"{complex_id}.fasta")
        info_path = os.path.join(info_dir, f"{complex_id}.json")
        decoy_info_path = os.path.join(info_dir, f"{complex_id}_decoy.json")

        # 1. Process native complex structure to extract sequences and residue info
        if os.path.exists(native_pdb) and (not os.path.exists(fasta_path) or not os.path.exists(info_path)):
            with tempfile.TemporaryDirectory() as tmpdir:
                prep_pdb = os.path.join(tmpdir, "prep_native.pdb")
                modified_occupancy(native_pdb, prep_pdb)
                structure = load_structure(prep_pdb)
                json_data, sequences = load_residue_info(structure, complex_id)
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)
                write_fasta(sequences, fasta_path, complex_id)

        if not os.path.exists(info_path):
            print(f"[WARN] Residue info missing for {complex_id}, skipping decoys...")
            continue

        with open(info_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        sequences = {cid: cdata["sequence"] for cid, cdata in json_data["chains"].items()}
        chain_ids = list(sequences.keys())

        # 2. Process decoy structures under decoys/
        decoy_dir = os.path.join(complex_dir, "decoys")
        decoy_files = sorted(glob.glob(os.path.join(decoy_dir, "*.pdb"))) if os.path.isdir(decoy_dir) else []

        # Extract decoy-specific residue info (e.g. for res_idx mapping in graph build)
        if decoy_files and not os.path.exists(decoy_info_path):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    prep_decoy = os.path.join(tmpdir, "prep_decoy_sample.pdb")
                    modified_occupancy(decoy_files[0], prep_decoy)
                    decoy_structure = load_structure(prep_decoy)
                    decoy_json_data, _ = load_residue_info(decoy_structure, f"{complex_id}_decoy")
                    with open(decoy_info_path, "w", encoding="utf-8") as f:
                        json.dump(decoy_json_data, f, indent=2)
            except Exception as e:
                print(f"[WARN] Error extracting decoy residue info for {complex_id}: {e}")

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
    args = parser.parse_args()

    extract_train_esmif(
        pdb_dir=args.pdb_dir,
        fasta_dir=args.fasta_dir,
        info_dir=args.info_dir,
        esmif_dir=args.esmif_dir,
    )


if __name__ == "__main__":
    main()
