# source /opt/software/miniconda/py39_25.9.1-1/bin/activate esm-if

import argparse
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


def parse_chain_ids(chain_ids_value):
    if chain_ids_value is None:
        return None
    chain_ids = [item.strip() for item in chain_ids_value.split(",") if item.strip()]
    return chain_ids or None


def write_fasta(sequences, fasta_output, pdb_id):
    """Write chain sequences to a FASTA file. Non-standard AAs are already encoded as X."""
    os.makedirs(os.path.dirname(fasta_output) or ".", exist_ok=True)
    with open(fasta_output, "w") as f:
        for chain_id, seq in sequences.items():
            f.write(f">{chain_id}\n{seq}\n")


def batch_process_pdb_folder(
    pdb_dir,
    fasta_dir,
    info_dir,
    esmif_dir,
):
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}")
        return

    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(esmif_dir, exist_ok=True)

    model, alphabet = load_esmif_model()

    for pdb_fname in tqdm(pdb_files):
        pdb_id = os.path.splitext(pdb_fname)[0]
        raw_pdb_path = os.path.join(pdb_dir, pdb_fname)
        fasta_output = os.path.join(fasta_dir, pdb_id + ".fasta")
        json_output = os.path.join(info_dir, pdb_id + ".json")
        esmif_output = os.path.join(esmif_dir, pdb_id + ".pkl")

        if os.path.exists(esmif_output):
            print(f"Skipping {pdb_fname} since ESM-IF output already exists.")
            continue

        print(f"Processing {pdb_fname} ...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                preprocessed = os.path.join(tmpdir, "preprocessed.pdb")
                modified_occupancy(raw_pdb_path, preprocessed)

                structure = load_structure(preprocessed)
                json_data, sequences = load_residue_info(structure, pdb_fname)

                with open(json_output, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

                write_fasta(sequences, fasta_output, pdb_id)

                chain_ids = list(sequences.keys())
                complex_reps = ext_complex_reps(model, alphabet, preprocessed, chain_ids, sequences)
                monomer_reps = {
                    chain_id: ext_single_chain_rep(model, alphabet, preprocessed, chain_id, sequences)
                    for chain_id in chain_ids
                }

            with open(esmif_output, "wb") as f:
                pickle.dump({"multimer": complex_reps, "monomer": monomer_reps}, f)
        except Exception as e:
            print(f"  ERROR processing {pdb_fname}: {e}")
            continue


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


def extract_esmif_batch(
    pdb_paths,
    sequences,
    out_paths,
    json_outputs=None,
):
    if len(pdb_paths) != len(out_paths):
        raise ValueError("pdb_paths and out_paths must be the same length.")
    if json_outputs is not None and len(json_outputs) != len(pdb_paths):
        raise ValueError("json_outputs must be the same length as pdb_paths.")

    model, alphabet = load_esmif_model()
    for idx, raw_complex_pdb_path in enumerate(pdb_paths):
        json_output = json_outputs[idx] if json_outputs is not None else None
        extract_esmif(
            raw_complex_pdb_path=raw_complex_pdb_path,
            sequences=sequences,
            out_path=out_paths[idx],
            json_output=json_output,
            model_arg=model,
            alphabet_arg=alphabet,
        )


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-IF features for a folder of PDB files.")
    parser.add_argument("--pdb_dir", required=True, help="Input folder of PDB files.")
    parser.add_argument("--fasta_dir", required=True, help="Output folder for FASTA files.")
    parser.add_argument("--info_dir", required=True, help="Output folder for residue_info JSON files.")
    parser.add_argument("--esmif_dir", required=True, help="Output folder for ESM-IF .pkl files.")

    args = parser.parse_args()

    batch_process_pdb_folder(
        pdb_dir=args.pdb_dir,
        fasta_dir=args.fasta_dir,
        info_dir=args.info_dir,
        esmif_dir=args.esmif_dir,
    )


if __name__ == "__main__":
    main()
