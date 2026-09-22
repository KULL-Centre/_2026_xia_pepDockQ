import argparse
import csv
import glob
import json
import os
import subprocess
from tqdm import tqdm


def compute_all_dockq(pdb_dir, output_dir, input_csvs):
    os.makedirs(output_dir, exist_ok=True)
    complex_ids = set()
    for input_csv in input_csvs:
        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "pdb_id" not in reader.fieldnames:
                raise ValueError(f"{input_csv} must contain a pdb_id column")
            complex_ids.update(row["pdb_id"].strip() for row in reader)
    complex_ids = sorted(complex_ids)
    for complex_id in tqdm(complex_ids, desc="Computing ground-truth DockQ"):
        complex_dir = os.path.join(pdb_dir, complex_id)
        native_pdb = os.path.join(complex_dir, f"{complex_id}.pdb")
        if not os.path.exists(native_pdb):
            native_pdb = os.path.join(complex_dir, "complex.pdb")
        if not os.path.exists(native_pdb):
            print(f"[WARN] Native PDB not found for {complex_id}, skipping...")
            continue

        decoy_dir = os.path.join(complex_dir, "decoys")
        decoy_files = sorted(glob.glob(os.path.join(decoy_dir, "*.pdb"))) if os.path.isdir(decoy_dir) else []
        complex_dockq = {}

        for decoy_pdb in decoy_files:
            fname = os.path.basename(decoy_pdb)
            # Extract model index from <complex_id>_model_<idx>.pdb
            model_part = fname.replace(f"{complex_id}_model_", "").replace(".pdb", "")
            if not model_part.isdigit():
                continue
            model_i = int(model_part)

            result = subprocess.run(
                ["DockQ", decoy_pdb, native_pdb, "--short"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"DockQ failed for {decoy_pdb}: {result.stderr.strip() or result.stdout.strip()}"
                )

            dockq_score = None
            for line in result.stdout.splitlines():
                if line.startswith("DockQ "):
                    dockq_score = float(line.split()[1])
                    break
            if dockq_score is None:
                raise RuntimeError(f"Could not parse DockQ score for {decoy_pdb}: {result.stdout.strip()}")

            complex_dockq[model_i] = {"dockq": dockq_score}

        output_json = os.path.join(output_dir, f"{complex_id}.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(complex_dockq, f, indent=2)
        print(f"[INFO] DockQ scores saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Compute DockQ scores for all decoy PDBs against native PDBs.")
    parser.add_argument("--pdb_dir", required=True, help="Input directory containing complex PDB folders.")
    parser.add_argument("--output_dir", required=True, help="Output directory for per-complex DockQ JSON files.")
    parser.add_argument("--input_csvs", required=True, nargs="+", help="Train, validation, and test split CSV files.")
    args = parser.parse_args()

    compute_all_dockq(args.pdb_dir, args.output_dir, args.input_csvs)


if __name__ == "__main__":
    main()
