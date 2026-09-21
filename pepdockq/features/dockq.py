import argparse
import glob
import json
import os
import subprocess
from tqdm import tqdm


def compute_all_dockq(pdb_dir, output_json):
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    complex_ids = sorted([d for d in os.listdir(pdb_dir) if os.path.isdir(os.path.join(pdb_dir, d))])
    all_dockq = {}

    try:
        from DockQ.DockQ import load_PDB, calc_DockQ
        has_dockq_pkg = True
    except ImportError:
        has_dockq_pkg = False

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
        all_dockq[complex_id] = {}

        for decoy_pdb in decoy_files:
            fname = os.path.basename(decoy_pdb)
            # Extract model index from <complex_id>_model_<idx>.pdb
            model_part = fname.replace(f"{complex_id}_model_", "").replace(".pdb", "")
            if not model_part.isdigit():
                continue
            model_i = int(model_part)

            dockq_score = 0.0
            if has_dockq_pkg:
                try:
                    model_struct = load_PDB(decoy_pdb)
                    native_struct = load_PDB(native_pdb)
                    dockq_score = float(calc_DockQ(model_struct, native_struct)["DockQ"])
                except Exception:
                    dockq_score = 0.0
            else:
                cmd = f"DockQ {decoy_pdb} {native_pdb} --short"
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        if "DockQ" in line:
                            dockq_score = float(line.strip().split()[-1])
                            break

            all_dockq[complex_id][model_i] = {"dockq": dockq_score}

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_dockq, f, indent=2)
    print(f"[INFO] DockQ scores saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Compute DockQ scores for all decoy PDBs against native PDBs.")
    parser.add_argument("--pdb_dir", required=True, help="Input directory containing complex PDB folders.")
    parser.add_argument("--output_json", required=True, help="Output JSON path for DockQ ground truths.")
    args = parser.parse_args()

    compute_all_dockq(args.pdb_dir, args.output_json)


if __name__ == "__main__":
    main()
