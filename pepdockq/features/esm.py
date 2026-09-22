import argparse
import csv
import os
import esm
import torch
import pickle

def load_fasta(fpath):
    sequences = {}
    current_id = None
    current_seq = []

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Compute per-chain ESM2 embeddings for all FASTAs in a directory.")
    parser.add_argument("--fasta_dir", required=True, help="Directory containing .fasta files.")
    parser.add_argument("--esm_dir", required=True, help="Output directory for per-chain .pkl files.")
    parser.add_argument("--input_csv", required=True, nargs="+", help="CSV manifest file(s) listing decoy_id or pdb_id values to process.")
    args = parser.parse_args()

    os.makedirs(args.esm_dir, exist_ok=True)

    pdb_ids = []
    for input_csv in args.input_csv:
        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{input_csv} must contain a header row")
            id_column = "decoy_id" if "decoy_id" in reader.fieldnames else "pdb_id"
            if id_column not in reader.fieldnames:
                raise ValueError(f"{input_csv} must contain a decoy_id or pdb_id column")
            pdb_ids.extend(row[id_column].strip() for row in reader)

    fasta_files = [f"{pdb_id}.fasta" for pdb_id in pdb_ids]
    missing_fastas = [fasta_file for fasta_file in fasta_files if not os.path.isfile(os.path.join(args.fasta_dir, fasta_file))]
    if missing_fastas:
        raise FileNotFoundError(f"Missing FASTA file(s): {', '.join(missing_fastas)}")
    print(f"Found {len(fasta_files)} fasta files", flush=True)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.cuda()
    model.eval()

    for fasta_file in fasta_files:
        pdb_id = fasta_file[:-6]  # Remove .fasta
        esm_output = os.path.join(args.esm_dir, f"{pdb_id}.pkl")
        if os.path.exists(esm_output):
            continue

        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        sequences = load_fasta(fasta_path)

        seq_emb_dict = {}
        for chain_id, sequence in sequences.items():

            data = [(chain_id, sequence)]
            _, _, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            seq_emb = token_representations[0, 1 : batch_lens[0] - 1].cpu()
            assert len(seq_emb) == len(sequence), (
                f"Embedding length {len(seq_emb)} != sequence length {len(sequence)} for {chain_id}"
            )

            seq_emb_dict[chain_id] = seq_emb
        
        with open(esm_output, "wb") as f:
            pickle.dump(seq_emb_dict, f)
        
        print(f"Done: {fasta_file}", flush=True)


if __name__ == "__main__":
    main()
