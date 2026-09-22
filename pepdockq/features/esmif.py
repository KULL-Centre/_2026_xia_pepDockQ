import argparse
import csv
import json
import os
import pickle
import subprocess
import tempfile
from collections import defaultdict

import esm
import esm.inverse_folding
import numpy as np
import torch
from biotite.structure import filter_backbone, get_chains
from biotite.structure.io import pdb, pdbx
import biotite.structure as struc
from tqdm import tqdm


def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)  #  "N", "CA", "C" 
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure

def load_residue_info(structure, pdb_id):

    # structure: AtomArray
    res_starts = struc.get_residue_starts(structure)

    res_ids = structure.res_id[res_starts]
    ins_codes = structure.ins_code[res_starts]

    res_names = structure.res_name[res_starts]
    chain_ids = structure.chain_id[res_starts]

    res_names_1letter = []

    for resname in res_names:  
        if resname in three_to_one:
            aa = three_to_one[resname]
        else:
            aa = 'X'
        res_names_1letter.append(aa)
    
    residue_info = list(zip(
                                chain_ids,
                                res_ids,
                                ins_codes,
                                res_names_1letter
                            ))
                            
    chains = defaultdict(lambda: {"sequence": [], "res_idx": []})

    for chain, res_id, ins_code, aa in residue_info:
        chains[chain]["sequence"].append(aa)
        res_label = f"{res_id}{ins_code}" if ins_code else str(res_id)
        chains[chain]["res_idx"].append(res_label)

    json_data = {
                    "pdb_id": pdb_id,
                    "chains": {
                        chain_id: {
                            "sequence": "".join(info["sequence"]),
                            "res_idx": info["res_idx"],
                        }
                        for chain_id, info in chains.items()
                    }
                }
    
    sequences = {}
    for chain_id in json_data["chains"]:
        sequences[chain_id] = json_data["chains"][chain_id]["sequence"]

    return json_data, sequences

three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}


def ext_single_chain_rep(model, alphabet, fpath, chain_id, sequence):
    # for single chain
    structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
    coords, chain_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

    assert chain_seq == sequence[chain_id]

    prob_tokens, alphabet, hidden_representations = run_model(coords, chain_seq, save_hiddens=False)
    mutation_score_matrix = score_variants(chain_seq, prob_tokens, alphabet)
    return mutation_score_matrix


def ext_complex_reps(model, alphabet, fpath, chain_ids, sequences):
    # for multiple chains
    structure = esm.inverse_folding.util.load_structure(fpath, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    reps = {}

    for target_chain_id in chain_ids:
        assert native_seqs[target_chain_id] == sequences[target_chain_id], (
                f"Mismatch!: {fpath}\n"
                f"native: {native_seqs[target_chain_id]}\n"
                f"seq   : {sequences[target_chain_id]}"
            )

        all_coords = esm.inverse_folding.multichain_util._concatenate_coords(coords, target_chain_id)
        chain_seq = native_seqs[target_chain_id]

        prob_tokens, alphabet, hidden_representations = run_model(all_coords, chain_seq, save_hiddens=False)
        mutation_score_matrix = score_variants(chain_seq, prob_tokens, alphabet)

        assert len(mutation_score_matrix) == len(sequences[target_chain_id])
        reps[target_chain_id] = mutation_score_matrix

    return reps


def modified_occupancy(input_file, output_file):
    awk_script = r'''
    BEGIN {
        split("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR ASX GLX UNK", a)
        for (i in a) valid[a[i]] = 1
    }

    /^(ATOM|HETATM)/ {
        # --- residue name: columns 18-20 (1-based) ---
        res = substr($0, 18, 3)
        if (!(res in valid)) {
            # replace residue name with UNK, keep alignment
            $0 = substr($0, 1, 17) "UNK" substr($0, 21)
        }

        # --- set occupancy to 1.00 (columns 55-60) ---
        printf "%-54s%6.2f%s\n", substr($0,1,54), 1.00, substr($0,61)
        next
    }

    { print }
    '''

    with open(output_file, "w") as fout:
        subprocess.run(
            ["awk", awk_script, input_file],
            stdout=fout,
            check=True
        )

    return

def masked_marginals(wt, idx, mt, sequence, token_probs, alphabet):
    wt, idx, mt = wt, idx, mt
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # no BOS here (apparently)
    score = token_probs[0,idx, mt_encoded] - token_probs[0, idx, wt_encoded]
    return score.item()


def run_model(coords, sequence, save_hiddens=False):
    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
    batch = [(coords, None, sequence)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device='cuda')

    prev_output_tokens = tokens[:, :-1].cuda()

    logits, extra = model.forward(coords, padding_mask, confidence, prev_output_tokens, return_all_hiddens=save_hiddens)
    if save_hiddens:
        hidden_representations=extra['inner_states'][-1].cpu().detach().numpy()
        hidden_representations=np.swapaxes(hidden_representations,0,1)
    else:
        hidden_representations=None
    
    logits_swapped=torch.swapaxes(logits,1,2) 
    token_probs = torch.log_softmax(logits_swapped, dim=-1)
    
    return token_probs, alphabet, hidden_representations


def score_variants(sequence, token_probs, alphabet):
    score_matrix = torch.zeros((len(sequence), 20))

    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}

    for i, wt in enumerate(sequence):
        for j in range(1, 21):
            if alphabetAA_D_L[j] != wt:
                score = masked_marginals(wt, i, alphabetAA_D_L[j], sequence, token_probs, alphabet)
                score_matrix[i, j-1] = score

    return score_matrix


def write_fasta(sequences, fasta_output, pdb_id):
    """Write chain sequences to a FASTA file. Non-standard AAs are already encoded as X."""
    os.makedirs(os.path.dirname(fasta_output) or ".", exist_ok=True)
    with open(fasta_output, "w") as f:
        for chain_id, seq in sequences.items():
            f.write(f">{chain_id}\n{seq}\n")


def read_prediction_inputs(input_csv, pdb_dir):
    """Validate the prediction manifest before loading an ESM model."""
    required_columns = {"decoy_id", "receptor_chains", "peptide_chains"}
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{input_csv} must contain columns: decoy_id,receptor_chains,peptide_chains"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{input_csv} does not contain any prediction inputs")

    inputs = []
    seen_ids = set()
    for row_number, row in enumerate(rows, start=2):
        pdb_id = row["decoy_id"].strip()
        receptor_chains = [chain.strip() for chain in row["receptor_chains"].split("_") if chain.strip()]
        peptide_chains = [chain.strip() for chain in row["peptide_chains"].split("_") if chain.strip()]
        if not pdb_id or not receptor_chains or not peptide_chains:
            raise ValueError(f"{input_csv}:{row_number} requires decoy_id, receptor_chains, and peptide_chains")
        if pdb_id in seen_ids:
            raise ValueError(f"{input_csv}:{row_number} duplicates decoy_id '{pdb_id}'")
        if os.path.basename(pdb_id) != pdb_id or pdb_id.endswith(".pdb"):
            raise ValueError(f"{input_csv}:{row_number} decoy_id must be a PDB filename stem, not a path")
        overlap = set(receptor_chains) & set(peptide_chains)
        if overlap:
            raise ValueError(f"{input_csv}:{row_number} assigns chain(s) {sorted(overlap)} to both receptor and peptide")
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"{input_csv}:{row_number} PDB file not found: {pdb_path}")
        inputs.append({
            "pdb_id": pdb_id,
            "pdb_path": pdb_path,
            "receptor": receptor_chains,
            "peptide": peptide_chains,
        })
        seen_ids.add(pdb_id)
    return inputs


def batch_process_pdb_folder(
    pdb_dir,
    fasta_dir,
    info_dir,
    esmif_dir,
    input_csv,
):
    prediction_inputs = read_prediction_inputs(input_csv, pdb_dir)

    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(esmif_dir, exist_ok=True)

    pending_inputs = []
    for prediction_input in prediction_inputs:
        pdb_id = prediction_input["pdb_id"]
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessed = os.path.join(tmpdir, "preprocessed.pdb")
            modified_occupancy(prediction_input["pdb_path"], preprocessed)
            structure = load_structure(preprocessed)
            json_data, sequences = load_residue_info(structure, f"{pdb_id}.pdb")

        available_chains = set(sequences)
        requested_chains = prediction_input["receptor"] + prediction_input["peptide"]
        missing_chains = [chain for chain in requested_chains if chain not in available_chains]
        if missing_chains:
            raise ValueError(
                f"{pdb_id}: requested chain(s) {missing_chains} are absent; available chains: {sorted(available_chains)}"
            )

        json_data["chain_roles"] = {
            "receptor": prediction_input["receptor"],
            "peptide": prediction_input["peptide"],
        }
        with open(os.path.join(info_dir, f"{pdb_id}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        write_fasta(sequences, os.path.join(fasta_dir, f"{pdb_id}.fasta"), pdb_id)

        esmif_output = os.path.join(esmif_dir, f"{pdb_id}.pkl")
        if not os.path.exists(esmif_output):
            prediction_input["sequences"] = sequences
            pending_inputs.append(prediction_input)

    if not pending_inputs:
        print("All requested ESM-IF features already exist.")
        return

    model, alphabet = load_esmif_model()

    for prediction_input in tqdm(pending_inputs, desc="ESM-IF features"):
        pdb_id = prediction_input["pdb_id"]
        print(f"Processing {pdb_id}.pdb ...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                preprocessed = os.path.join(tmpdir, "preprocessed.pdb")
                modified_occupancy(prediction_input["pdb_path"], preprocessed)
                sequences = prediction_input["sequences"]
                chain_ids = list(sequences)
                complex_reps = ext_complex_reps(model, alphabet, preprocessed, chain_ids, sequences)
                monomer_reps = {
                    chain_id: ext_single_chain_rep(model, alphabet, preprocessed, chain_id, sequences)
                    for chain_id in chain_ids
                }

            with open(os.path.join(esmif_dir, f"{pdb_id}.pkl"), "wb") as f:
                pickle.dump({"multimer": complex_reps, "monomer": monomer_reps}, f)
        except Exception as e:
            raise RuntimeError(f"Error processing {pdb_id}.pdb: {e}") from e


def load_esmif_model():
    global model, alphabet
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.cuda()
    model = model.eval()
    return model, alphabet


def extract_esmif(
    raw_complex_pdb_path,
    sequences,
    out_path,
    json_output=None,
    model_arg=None,
    alphabet_arg=None,
):
    global model, alphabet

    if model_arg is None or alphabet_arg is None:
        model, alphabet = load_esmif_model()
    else:
        model = model_arg
        alphabet = alphabet_arg

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        complex_pdb_path = os.path.join(tmpdir, "complex_tmp.pdb")
        modified_occupancy(raw_complex_pdb_path, complex_pdb_path)

        if json_output:
            structure = load_structure(complex_pdb_path)
            json_data, _ = load_residue_info(structure, os.path.basename(raw_complex_pdb_path))
            os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)

        chain_ids = list(sequences.keys())

        complex_reps_dict = ext_complex_reps(model, alphabet, complex_pdb_path, chain_ids, sequences)

        single_chain_reps_dict = {}
        for chain_id in chain_ids:
            rep = ext_single_chain_rep(model, alphabet, complex_pdb_path, chain_id, sequences)
            single_chain_reps_dict[chain_id] = rep

        with open(out_path, "wb") as f:
            pickle.dump({"multimer": complex_reps_dict, "monomer": single_chain_reps_dict}, f)


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-IF features for a folder of PDB files.")
    parser.add_argument("--pdb_dir", required=True, help="Input folder of PDB files.")
    parser.add_argument("--fasta_dir", required=True, help="Output folder for FASTA files.")
    parser.add_argument("--info_dir", required=True, help="Output folder for residue_info JSON files.")
    parser.add_argument("--esmif_dir", required=True, help="Output folder for ESM-IF .pkl files.")
    parser.add_argument("--input_csv", required=True, help="CSV with decoy_id, receptor_chains, and peptide_chains.")

    args = parser.parse_args()

    batch_process_pdb_folder(
        pdb_dir=args.pdb_dir,
        fasta_dir=args.fasta_dir,
        info_dir=args.info_dir,
        esmif_dir=args.esmif_dir,
        input_csv=args.input_csv,
    )


if __name__ == "__main__":
    main()
