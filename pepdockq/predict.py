import os
import torch
import argparse
from torch_geometric.loader import DataLoader
from pepdockq.models.hgnn import HGNN
from tqdm import tqdm
import csv


class PredPepDataset:
    """Eval-only dataset. Each sample carries pdb_id and decoy_id."""

    def __init__(self, pdb_list, data_path):
        self.data_path = data_path
        self.samples = []  # [(pdb_id, decoy_id, path), ...]
        self._build(pdb_list)

    def _build(self, pdb_list):
        pdb_set = set(pdb_list)
        for entry in sorted(os.scandir(self.data_path), key=lambda e: e.name):
            if not entry.is_file() or not entry.name.endswith('.pt'):
                continue
            stem = os.path.splitext(entry.name)[0]
            if stem in pdb_set:
                self.samples.append((stem, stem, entry.path))
        print(f'[PredPepDataset] {len(pdb_set)} PDBs -> {len(self.samples)} decoys')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pdb_id, decoy_id, path = self.samples[idx]
        return torch.load(path, weights_only=False)


def run_prediction(model, val_list, args, mode):
    """Run model forward pass and collect predictions. Returns list of {decoy_id, pred_score}."""
    model.eval()

    val_data = PredPepDataset(val_list, args.data_path)
    all_decoy_ids = [s[1] for s in val_data.samples]
    val_loader = DataLoader(
        val_data,
        args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    decoy_results = []
    cursor = 0
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Predicting {mode}"):
            data = data.to(args.device)
            _, _, global_pred = model(data)
            global_pred = global_pred.detach().reshape(-1).cpu().numpy()
            batch_size = len(global_pred)
            batch_decoy_ids = all_decoy_ids[cursor: cursor + batch_size]
            cursor += batch_size
            for decoy_id, score in zip(batch_decoy_ids, global_pred):
                decoy_results.append({'decoy_id': decoy_id, 'pred_score': float(score)})

    return decoy_results




def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str, required=True, help="Path to directory containing .pt files for prediction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with the decoy_id values to predict")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")

    args = parser.parse_args()
    

    return args



def main():
    args = parse_argument()

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "decoy_id" not in reader.fieldnames:
            raise ValueError(f"{args.input_csv} must contain a decoy_id column")
        all_pdb_ids = [row["decoy_id"].strip() for row in reader]
    if not all_pdb_ids or any(not pdb_id for pdb_id in all_pdb_ids):
        raise ValueError(f"{args.input_csv} must contain non-empty decoy_id values")

    missing_graphs = [pdb_id for pdb_id in all_pdb_ids if not os.path.isfile(os.path.join(args.data_path, f"{pdb_id}.pt"))]
    if missing_graphs:
        raise FileNotFoundError(f"Missing graph feature file(s): {', '.join(missing_graphs)}")

    model = HGNN(num_layers=4, hidden_channels=256,
                   features=["esm", 'zscale', 'dihedrals', 'esmif_mutation_score'],
                   heads=4, num_error_bins=32)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.to(args.device)

    print(f"[INFO] Predicting {len(all_pdb_ids)} PDB IDs from {args.input_csv}")

    decoy_results = run_prediction(model, all_pdb_ids, args, 'all')

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['decoy_id', 'pred_score'])
        writer.writeheader()
        writer.writerows(decoy_results)
    print(f"[INFO] Predictions saved to {args.output} ({len(decoy_results)} decoys)")


if __name__ == "__main__":
    main()

