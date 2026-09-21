import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import pickle
import re
import warnings

from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBParser
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData
from tqdm import tqdm

warnings.simplefilter("ignore", BiopythonWarning)

three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEP": "S",
}

MAX_PEPTIDE_LENGTH = 53
AA_LIST = list("ACDEFGHIKLMNPQRSTVWYX")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


class PepDockQTrainFeature:
    """Feature extractor for training-set complexes.

    Builds HeteroData graphs with ground-truth DockQ (data.y) and interface
    distance labels (data['peptide', 'near', 'protein'].edge_label) for computing
    local and global losses.
    """

    def __init__(
        self,
        decoy_dir: str,
        feature_dir: str,
        dockq_json: str,
        zscale_path: str,
        ncpus: int = 8,
        info_dir: str = None,
        esm_dir: str = None,
        esmif_dir: str = None,
        edge_th: float = 10.0,
        sidechain_type: str = "sc",
        rep_atom: str = "sc",
    ):
        self.decoy_dir = decoy_dir
        self.feature_dir = feature_dir
        self.ncpus = ncpus
        self.edge_th = edge_th
        self.edge_type = "radius"
        self.sidechain_type = sidechain_type
        self.rep_atom = rep_atom

        self.info_dir = info_dir or os.path.join(feature_dir, "info")
        self.esm_dir = esm_dir or os.path.join(feature_dir, "esm_embedding")
        self.esmif_dir = esmif_dir or os.path.join(feature_dir, "esmif_mutation_score")
        self.coord_path = os.path.join(feature_dir, "coordinate")
        self.distance_map_path = os.path.join(feature_dir, "distance_map")
        self.data_path = os.path.join(feature_dir, "graph")

        os.makedirs(self.coord_path, exist_ok=True)
        os.makedirs(self.distance_map_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

        self.AA_feat_dict = self.load_AA_df(zscale_path)
        self.dockq_dict = self.load_all_dockq(dockq_json)

        self.pdb_list = sorted(list(self.dockq_dict.keys()))
        self.chain_id_dict, self.chain_fasta_dict, self.pdb_list = self.get_chain_list()
        print(f"[INFO] Valid structures for training: {len(self.pdb_list)}")

        print("[RUN] Step 1/3: Computing coordinates (exp & decoys)...")
        self.cal_multiprocess(self.process_coordinates, max_workers=self.ncpus)

        print("[RUN] Step 2/3: Computing distance maps (exp & decoys)...")
        self.cal_multiprocess(self.process_distance_maps, max_workers=self.ncpus)

        print("[RUN] Step 3/3: Building HeteroData training graphs...")
        self.cal_multiprocess(self.process_graphs, max_workers=self.ncpus)
        print("[INFO] Finished processing all training structures.")

    def load_all_dockq(self, dockq_path: str) -> dict:
        with open(dockq_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        dockq_dict = {}
        for pdb_id, model_dict in raw.items():
            dockq_dict[pdb_id] = {}
            for model_i, info in model_dict.items():
                if isinstance(info, dict):
                    dockq_dict[pdb_id][int(model_i)] = float(info.get("dockq", info.get("GlobalDockQ", 0.0)))
                else:
                    dockq_dict[pdb_id][int(model_i)] = float(info)
        return dockq_dict

    def load_AA_df(self, zscale_path: str) -> pd.DataFrame:
        df = pd.read_csv(zscale_path)
        return df.set_index("aa")[["gz1", "gz2", "gz3", "gz4", "gz5", "onehot"]]

    def get_AA_feature(self, seq: str) -> torch.Tensor:
        aa_list = list(seq)
        feat = self.AA_feat_dict.reindex(aa_list).fillna(0).to_numpy()
        return torch.tensor(feat, dtype=torch.float32)

    def get_chain_list(self):
        chain_id_dict = {}
        chain_fasta_dict = {}
        selected_pdb_list = []

        for pdb_id in self.pdb_list:
            exp_res_info_fpath = os.path.join(self.info_dir, f"{pdb_id}.json")
            if not os.path.exists(exp_res_info_fpath):
                exp_res_info_fpath = os.path.join(self.decoy_dir, pdb_id, "complex_residue_info.json")
            if not os.path.exists(exp_res_info_fpath):
                exp_res_info_fpath = os.path.join(self.decoy_dir, f"{pdb_id}.json")

            pred_res_info_fpath = os.path.join(self.info_dir, f"{pdb_id}_decoy.json")
            if not os.path.exists(pred_res_info_fpath):
                pred_res_info_fpath = os.path.join(self.decoy_dir, pdb_id, f"{pdb_id}_model_0_residue_info.json")
            if not os.path.exists(pred_res_info_fpath):
                pred_res_info_fpath = exp_res_info_fpath

            if not os.path.exists(exp_res_info_fpath):
                print(f"[WARN] Residue info not found for {pdb_id}, skipping...")
                continue

            with open(exp_res_info_fpath, "r", encoding="utf-8") as f:
                exp_res_info = json.load(f)
            if os.path.exists(pred_res_info_fpath):
                with open(pred_res_info_fpath, "r", encoding="utf-8") as f:
                    pred_res_info = json.load(f)
            else:
                pred_res_info = exp_res_info

            if "_" in pdb_id:
                pep_chain_list = [pdb_id.split("_")[1]]
            else:
                pep_chain_list = [
                    cid for cid, cinfo in pred_res_info["chains"].items()
                    if len(cinfo["sequence"]) <= MAX_PEPTIDE_LENGTH
                ]
            chain_ids = list(pred_res_info["chains"].keys())
            rec_chain_list = [cid for cid in chain_ids if cid not in pep_chain_list]

            if not pep_chain_list or not rec_chain_list:
                print(f"[WARN] No valid peptide/receptor chains for {pdb_id}, skipping...")
                continue

            match_flag = True
            for chain_id in pep_chain_list + rec_chain_list:
                exp_seq = exp_res_info["chains"].get(chain_id, {}).get("sequence", "")
                pred_seq = pred_res_info["chains"].get(chain_id, {}).get("sequence", "")
                if exp_seq and pred_seq and len(exp_seq) != len(pred_seq):
                    match_flag = False
                    break

            if match_flag:
                selected_pdb_list.append(pdb_id)
                chain_id_dict[pdb_id] = {"peptide": pep_chain_list, "receptor": rec_chain_list}
                chain_fasta_dict[pdb_id] = {}
                for cid in pep_chain_list + rec_chain_list:
                    chain_fasta_dict[pdb_id][cid] = pred_res_info["chains"][cid]["sequence"]

        return chain_id_dict, chain_fasta_dict, selected_pdb_list

    def process_coordinates(self, pdb_id: str):
        # 1. Experimental structure
        exp_pdb = os.path.join(self.decoy_dir, pdb_id, f"{pdb_id}.pdb")
        if not os.path.exists(exp_pdb):
            exp_pdb = os.path.join(self.decoy_dir, pdb_id, "complex.pdb")
        if not os.path.exists(exp_pdb):
            exp_pdb = os.path.join(self.decoy_dir, f"{pdb_id}_exp.pdb")

        exp_res_info = os.path.join(self.info_dir, f"{pdb_id}.json")
        if not os.path.exists(exp_res_info):
            exp_res_info = os.path.join(self.decoy_dir, pdb_id, "complex_residue_info.json")
        if not os.path.exists(exp_res_info):
            exp_res_info = os.path.join(self.decoy_dir, f"{pdb_id}.json")

        exp_coord_file = os.path.join(self.coord_path, f"{pdb_id}_exp.pkl")
        if os.path.exists(exp_pdb) and os.path.exists(exp_res_info) and not os.path.exists(exp_coord_file):
            exp_coords = get_protein_coordinates(exp_pdb, exp_res_info)
            with open(exp_coord_file, "wb") as f:
                pickle.dump(exp_coords, f)

        # 2. Decoy structures
        pred_res_info = os.path.join(self.info_dir, f"{pdb_id}_decoy.json")
        if not os.path.exists(pred_res_info):
            pred_res_info = os.path.join(self.decoy_dir, pdb_id, f"{pdb_id}_model_0_residue_info.json")
        if not os.path.exists(pred_res_info):
            pred_res_info = exp_res_info

        for model_i in self.dockq_dict.get(pdb_id, {}):
            pred_pdb = os.path.join(self.decoy_dir, pdb_id, "decoys", f"{pdb_id}_model_{model_i}.pdb")
            if not os.path.exists(pred_pdb):
                pred_pdb = os.path.join(self.decoy_dir, pdb_id, f"{pdb_id}_model_{model_i}.pdb")
            if not os.path.exists(pred_pdb):
                pred_pdb = os.path.join(self.decoy_dir, f"{pdb_id}_model_{model_i}.pdb")
            if not os.path.exists(pred_pdb):
                continue

            pred_coord_file = os.path.join(self.coord_path, f"{pdb_id}_model_{model_i}.pkl")
            if not os.path.exists(pred_coord_file):
                pred_coords = get_protein_coordinates(pred_pdb, pred_res_info)
                with open(pred_coord_file, "wb") as f:
                    pickle.dump(pred_coords, f)

    def process_distance_maps(self, pdb_id: str):
        peptide_chains = self.chain_id_dict[pdb_id]["peptide"]
        receptor_chains = self.chain_id_dict[pdb_id]["receptor"]

        # 1. Experimental distance map
        exp_coord_file = os.path.join(self.coord_path, f"{pdb_id}_exp.pkl")
        exp_dist_file = os.path.join(self.distance_map_path, f"{pdb_id}_exp.pkl")

        if os.path.exists(exp_coord_file) and not os.path.exists(exp_dist_file):
            with open(exp_coord_file, "rb") as f:
                chain_coords = pickle.load(f)
            pep_coords = self.concat_all_atom_coordinates(pdb_id, peptide_chains, chain_coords)
            rec_coords = self.concat_all_atom_coordinates(pdb_id, receptor_chains, chain_coords)
            inter_dist = cal_minimize_atom_distance_map(pep_coords, rec_coords)
            pep_dist = cal_minimize_atom_distance_map(pep_coords, pep_coords)
            rec_dist = cal_minimize_atom_distance_map(rec_coords, rec_coords)
            dist_data = {"peptide-protein": inter_dist, "peptide": pep_dist, "protein": rec_dist}
            with open(exp_dist_file, "wb") as f:
                pickle.dump(dist_data, f)

        # 2. Decoy distance maps
        for model_i in self.dockq_dict.get(pdb_id, {}):
            coord_file = os.path.join(self.coord_path, f"{pdb_id}_model_{model_i}.pkl")
            dist_file = os.path.join(self.distance_map_path, f"{pdb_id}_model_{model_i}.pkl")

            if os.path.exists(coord_file) and not os.path.exists(dist_file):
                with open(coord_file, "rb") as f:
                    chain_coords = pickle.load(f)
                pep_coords = self.concat_all_atom_coordinates(pdb_id, peptide_chains, chain_coords)
                rec_coords = self.concat_all_atom_coordinates(pdb_id, receptor_chains, chain_coords)
                inter_dist = cal_minimize_atom_distance_map(pep_coords, rec_coords)
                pep_dist = cal_minimize_atom_distance_map(pep_coords, pep_coords)
                rec_dist = cal_minimize_atom_distance_map(rec_coords, rec_coords)
                dist_data = {"peptide-protein": inter_dist, "peptide": pep_dist, "protein": rec_dist}
                with open(dist_file, "wb") as f:
                    pickle.dump(dist_data, f)

    def concat_all_atom_coordinates(self, pdb_id: str, chain_list: list, chain_coords: dict):
        mol_coords = []
        for cid in chain_list:
            mol_coords += chain_coords[cid]["all_atom_coords"]
        return mol_coords

    def get_sequence_feature(self, pdb_id: str, chain_list: list, esm_file: str) -> torch.Tensor:
        if esm_file.endswith(".pt"):
            esm_emb_dict = torch.load(esm_file, map_location="cpu", weights_only=False)
        else:
            with open(esm_file, "rb") as f:
                esm_emb_dict = pickle.load(f)

        mol_feat = []
        for chain_id in chain_list:
            sequence = self.chain_fasta_dict[pdb_id][chain_id]
            esm_emb = esm_emb_dict[chain_id]
            if isinstance(esm_emb, np.ndarray):
                esm_emb = torch.from_numpy(esm_emb)

            assert len(esm_emb) == len(sequence), f"ESM length {len(esm_emb)} != seq length {len(sequence)}"
            aa_feat = self.get_AA_feature(sequence)
            chain_feat = torch.hstack([esm_emb, aa_feat])
            assert chain_feat.shape[0] == len(sequence)
            assert chain_feat.shape[1] == 1286
            mol_feat.append(chain_feat)

        return torch.vstack(mol_feat).float()

    def process_graphs(self, pdb_id: str):
        peptide_chains = self.chain_id_dict[pdb_id]["peptide"]
        receptor_chains = self.chain_id_dict[pdb_id]["receptor"]

        exp_dist_file = os.path.join(self.distance_map_path, f"{pdb_id}_exp.pkl")
        if not os.path.exists(exp_dist_file):
            print(f"[WARN] No experimental distance map for {pdb_id}")
            return

        with open(exp_dist_file, "rb") as f:
            exp_dist_maps_dict = pickle.load(f)

        # ESM-2 embedding is sequence-based and shared across decoys
        esm_file = os.path.join(self.esm_dir, f"{pdb_id}.pkl")
        if not os.path.exists(esm_file):
            esm_file = os.path.join(self.esm_dir, f"{pdb_id}.pt")
        if not os.path.exists(esm_file):
            print(f"[WARN] No ESM embedding for {pdb_id}")
            return

        pep_seq_feat = self.get_sequence_feature(pdb_id, peptide_chains, esm_file)
        rec_seq_feat = self.get_sequence_feature(pdb_id, receptor_chains, esm_file)

        for model_i, dockq_val in self.dockq_dict.get(pdb_id, {}).items():
            saved_path = os.path.join(self.data_path, f"{pdb_id}_model_{model_i}.pt")
            if os.path.exists(saved_path):
                continue

            decoy_dist_file = os.path.join(self.distance_map_path, f"{pdb_id}_model_{model_i}.pkl")
            pred_coord_file = os.path.join(self.coord_path, f"{pdb_id}_model_{model_i}.pkl")
            pred_esmif_file = os.path.join(self.esmif_dir, f"{pdb_id}_model_{model_i}.pkl")

            if not os.path.exists(decoy_dist_file) or not os.path.exists(pred_coord_file) or not os.path.exists(pred_esmif_file):
                continue

            with open(decoy_dist_file, "rb") as f:
                decoy_dist_maps_dict = pickle.load(f)

            pep_esmif_emb, rec_esmif_emb = get_esmif_feat(pred_esmif_file, peptide_chains, receptor_chains)
            chain_backbone_coord_dict = get_atom_coordinates(pred_coord_file)

            pep_backbone_coord = self.concat_coord(chain_backbone_coord_dict, peptide_chains)
            rec_backbone_coord = self.concat_coord(chain_backbone_coord_dict, receptor_chains)

            if len(rec_backbone_coord) == 0 or len(pep_backbone_coord) == 0:
                continue

            pep_feat = torch.cat([pep_seq_feat, pep_esmif_emb], dim=1)
            rec_feat = torch.cat([rec_seq_feat, rec_esmif_emb], dim=1)

            pep_mask = torch.isfinite(pep_backbone_coord.sum(dim=(1, 2)))
            pep_backbone_coord[~pep_mask] = np.inf
            rec_mask = torch.isfinite(rec_backbone_coord.sum(dim=(1, 2)))
            rec_backbone_coord[~rec_mask] = np.inf

            pep_node_s = self.get_node_attr(pep_backbone_coord)
            rec_node_s = self.get_node_attr(rec_backbone_coord)

            pep_edge_s, pep_edge_index = self.get_hetero_edge_attr(True, self.edge_type, pep_backbone_coord)
            rec_edge_s, rec_edge_index = self.get_hetero_edge_attr(True, self.edge_type, rec_backbone_coord)
            pep_rec_edge_s, pep_rec_edge_index, rec_pep_edge_s, rec_pep_edge_index = self.get_hetero_edge_attr(
                False, self.edge_type, pep_backbone_coord, rec_backbone_coord
            )

            data = HeteroData()
            data["peptide"].seq_feat = pep_feat.float()
            data["peptide"].node_s = pep_node_s

            data["protein"].seq_feat = rec_feat.float()
            data["protein"].node_s = rec_node_s

            data["peptide", "near", "peptide"].edge_index = pep_edge_index
            data["peptide", "near", "peptide"].edge_s = pep_edge_s

            data["protein", "near", "protein"].edge_index = rec_edge_index
            data["protein", "near", "protein"].edge_s = rec_edge_s

            data["peptide", "near", "protein"].edge_index = pep_rec_edge_index
            data["peptide", "near", "protein"].edge_s = pep_rec_edge_s

            data["protein", "near", "peptide"].edge_index = rec_pep_edge_index
            data["protein", "near", "peptide"].edge_s = rec_pep_edge_s

            # Ground truth interface distance labels & DockQ
            exp_inter_dist = torch.tensor(
                exp_dist_maps_dict["peptide-protein"][pep_rec_edge_index[0], pep_rec_edge_index[1]],
                dtype=torch.float32,
            )
            decoy_inter_dist = torch.tensor(
                decoy_dist_maps_dict["peptide-protein"][pep_rec_edge_index[0], pep_rec_edge_index[1]],
                dtype=torch.float32,
            )
            if exp_inter_dist.ndim == 0:
                exp_inter_dist = exp_inter_dist.unsqueeze(0)
                decoy_inter_dist = decoy_inter_dist.unsqueeze(0)

            data["peptide", "near", "protein"].edge_label = torch.stack([exp_inter_dist, decoy_inter_dist], dim=1)
            data.y = torch.tensor([dockq_val], dtype=torch.float32)

            torch.save(data, saved_path)

    def cal_rep_atom_coord(self, atoms_coord):
        rep_atom_col_dict = {"ca": 1, "cb": 4, "sc": 5, "center": 6}
        return atoms_coord[:, rep_atom_col_dict[self.rep_atom], :]

    def cal_sidechain_coord(self, atoms_coord):
        if self.sidechain_type == "imputed_cb":
            return self.cal_imputed_cb(atoms_coord)
        sidechain_col_dict = {"cb": 4, "sc": 5}
        return atoms_coord[:, sidechain_col_dict[self.sidechain_type], :]

    def cal_imputed_cb(self, coords):
        X_ca = coords[:, 1, :]
        b = coords[:, 1, :] - coords[:, 0, :]
        c = coords[:, 2, :] - coords[:, 1, :]
        a = torch.cross(b, c, dim=-1)
        return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X_ca

    def get_node_attr(self, coords_):
        backbone_coords = coords_[:, :4, :]
        dihedrals = self._dihedrals(backbone_coords)
        return dihedrals.float()

    def get_hetero_edge_attr(self, homo_graph, edge_type, mol1_coords_, mol2_coords_=None):
        mol1_backbone_coords = mol1_coords_[:, :4, :]
        mol1_sidechain = self.cal_sidechain_coord(mol1_coords_)
        num_positional_embeddings = 16

        if homo_graph:
            assert mol2_coords_ is None
            mol_X = self.cal_rep_atom_coord(mol1_coords_)
            if edge_type == "radius":
                edge_index = radius_graph(mol_X, r=self.edge_th, loop=True)
            else:
                raise KeyError(f"Unsupported edge_type: {edge_type}")
            pos_embeddings = self._positional_embeddings(edge_index, num_positional_embeddings)
            mol2_backbone_coords = mol1_backbone_coords
            mol2_sidechain = mol1_sidechain
        else:
            assert mol2_coords_ is not None
            mol2_backbone_coords = mol2_coords_[:, :4, :]
            mol2_sidechain = self.cal_sidechain_coord(mol2_coords_)
            mol1_X = self.cal_rep_atom_coord(mol1_coords_)
            mol2_X = self.cal_rep_atom_coord(mol2_coords_)
            if edge_type == "radius":
                edge_index = radius(mol2_X, mol1_X, r=self.edge_th)
            else:
                raise KeyError(f"Unsupported edge_type: {edge_type}")
            pos_embeddings = torch.zeros((edge_index.shape[1], num_positional_embeddings))

        RBF_all, _ = cal_edge_attr(mol1_backbone_coords, mol1_sidechain, mol2_backbone_coords, mol2_sidechain, edge_index)
        edge_s = torch.cat([pos_embeddings, RBF_all], dim=-1).float()
        edge_s = torch.nan_to_num(edge_s)

        if homo_graph:
            return edge_s, edge_index

        rev_edge_index = edge_index.flip(0)
        rev_RBF_all, _ = cal_edge_attr(mol2_backbone_coords, mol2_sidechain, mol1_backbone_coords, mol1_sidechain, rev_edge_index)
        rev_edge_s = torch.cat([pos_embeddings, rev_RBF_all], dim=-1).float()
        rev_edge_s = torch.nan_to_num(rev_edge_s)
        return edge_s, edge_index, rev_edge_s, rev_edge_index

    def _dihedrals(self, X, eps=1e-7):
        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        return torch.cat([torch.cos(D), torch.sin(D)], 1)

    def _positional_embeddings(self, edge_index, num_embeddings=16):
        d = edge_index[0] - edge_index[1]
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        return torch.cat((torch.cos(angles), torch.sin(angles)), -1)

    def concat_coord(self, chain_coord_dict, chain_list):
        coord = [chain_coord_dict[cid] for cid in chain_list]
        return torch.concat(coord, dim=0)

    def cal_multiprocess(self, func, max_workers: int):
        results = []
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for pdb_id in self.pdb_list:
                tasks.append(executor.submit(func, pdb_id))
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
                results.append(future.result())
        return results


def get_atom_coordinates(coord_file: str) -> dict:
    with open(coord_file, "rb") as f:
        chain_coord_dict = pickle.load(f)

    chain_residue_coord_dict = {}
    for chain_id, coord_dict in chain_coord_dict.items():
        atoms_coord = [
            coord_dict["backbone_atom_coords"],
            coord_dict["cb_coords"][:, None, :],
            coord_dict["sidechain_centers"][:, None, :],
            coord_dict["centers"][:, None, :],
        ]
        atoms_coord = np.concatenate(atoms_coord, axis=1)
        chain_residue_coord_dict[chain_id] = torch.tensor(atoms_coord)
    return chain_residue_coord_dict


def get_esmif_feat(esmif_file_or_dict, peptide_chains, receptor_chains):
    if isinstance(esmif_file_or_dict, str):
        with open(esmif_file_or_dict, "rb") as f:
            esmif_emb_dict = pickle.load(f)
    else:
        esmif_emb_dict = esmif_file_or_dict

    pep_esmif_emb = [
        torch.cat([esmif_emb_dict["multimer"][cid], esmif_emb_dict["monomer"][cid]], dim=1)
        for cid in peptide_chains
    ]
    pep_esmif_emb = torch.cat(pep_esmif_emb, dim=0)

    rec_esmif_emb = [
        torch.cat([esmif_emb_dict["multimer"][cid], esmif_emb_dict["monomer"][cid]], dim=1)
        for cid in receptor_chains
    ]
    rec_esmif_emb = torch.cat(rec_esmif_emb, dim=0)

    return pep_esmif_emb, rec_esmif_emb


def cal_edge_attr(mol1_backbone_coords, mol1_sidechain, mol2_backbone_coords, mol2_sidechain, edge_index):
    N1 = mol1_backbone_coords[:, 0, :][edge_index[0]]
    Ca1 = mol1_backbone_coords[:, 1, :][edge_index[0]]
    C1 = mol1_backbone_coords[:, 2, :][edge_index[0]]
    O1 = mol1_backbone_coords[:, 3, :][edge_index[0]]
    Cb1 = mol1_sidechain[edge_index[0]]

    N2 = mol2_backbone_coords[:, 0, :][edge_index[1]]
    Ca2 = mol2_backbone_coords[:, 1, :][edge_index[1]]
    C2 = mol2_backbone_coords[:, 2, :][edge_index[1]]
    O2 = mol2_backbone_coords[:, 3, :][edge_index[1]]
    Cb2 = mol2_sidechain[edge_index[1]]

    RBF_all = []
    vec_all = []
    for atom1 in [N1, Ca1, C1, O1, Cb1]:
        for atom2 in [N2, Ca2, C2, O2, Cb2]:
            vec = atom1 - atom2
            RBF_all.append(_rbf(vec.norm(dim=-1)))
            vec_all.append(_normalize(vec).unsqueeze(-2))
    return torch.cat(RBF_all, dim=1), torch.cat(vec_all, dim=1)


def parse_res_idx_biopython(res_idx: str):
    m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", res_idx.strip())
    if m is None:
        raise ValueError(f"Invalid res_idx: {res_idx}")
    resseq = int(m.group(1))
    icode = m.group(2) if m.group(2) else " "
    return (" ", resseq, icode)


def get_protein_coordinates(pdb_path: str, res_info_path_or_dict):
    if isinstance(res_info_path_or_dict, str):
        with open(res_info_path_or_dict, "r", encoding="utf-8") as f:
            res_info = json.load(f)
    else:
        res_info = res_info_path_or_dict

    ext = os.path.splitext(pdb_path)[1].lower()
    if ext == ".pdb":
        parser = PDBParser(QUIET=True)
    elif ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    structure = parser.get_structure("protein", pdb_path)
    chain_coord_dict = {}

    for chain in structure[0]:
        chain_id = chain.id
        residue_dict_ = {res.id: res for res in chain.get_residues()}
        residue_dict = {(" ", k[1], k[2]): v for k, v in residue_dict_.items()}

        selected_residues = []
        res_idx_list = res_info["chains"][chain_id]["res_idx"]
        for res_idx in res_idx_list:
            res_id = parse_res_idx_biopython(res_idx)
            assert res_id in residue_dict, f"Residue {res_id} not in PDB structure {pdb_path}"
            selected_residues.append(residue_dict[res_id])

        sequence = ""
        chain_all_atom_coords = []
        chain_backbone_atom_coords = []
        chain_cb_coords = []
        chain_centers = []
        chain_sidechain_centers = []

        for res in selected_residues:
            resname = res.get_resname().strip()
            if res.id[0] == "W" or resname in ("HOH", "WAT", "H2O"):
                continue

            aa = three_to_one.get(resname, "X")
            sequence += aa

            all_atom_coords = []
            sidechain_atom_coords = []
            for atom in res:
                atom_name = atom.get_name()
                if atom_name.startswith("H"):
                    continue
                all_atom_coords.append(atom.get_coord())
                if atom_name not in ["N", "CA", "C", "O"]:
                    sidechain_atom_coords.append(atom.get_coord())

            n_coord = res["N"].get_coord() if "N" in res else all_atom_coords[0]
            ca_coord = res["CA"].get_coord() if "CA" in res else n_coord
            c_coord = res["C"].get_coord() if "C" in res else ca_coord
            o_coord = res["O"].get_coord() if "O" in res else c_coord
            backbone_atom_coords = np.vstack([n_coord, ca_coord, c_coord, o_coord]).reshape(-1, 4, 3)

            all_atom_coords = np.vstack(all_atom_coords)
            if len(sidechain_atom_coords) == 0:
                sidechain_atom_coords.append(ca_coord)
            sidechain_atom_coords = np.vstack(sidechain_atom_coords)

            cb_coord = res["CB"].get_coord() if "CB" in res else ca_coord
            center = np.mean(all_atom_coords, axis=0)
            sidechain_center = np.mean(sidechain_atom_coords, axis=0)

            chain_all_atom_coords.append(all_atom_coords)
            chain_backbone_atom_coords.append(backbone_atom_coords)
            chain_cb_coords.append(cb_coord)
            chain_centers.append(center)
            chain_sidechain_centers.append(sidechain_center)

        assert len(sequence) == len(res_info["chains"][chain_id]["sequence"])
        chain_coord_dict[chain_id] = {
            "sequence": sequence,
            "all_atom_coords": chain_all_atom_coords,
            "backbone_atom_coords": np.concatenate(chain_backbone_atom_coords, axis=0),
            "cb_coords": np.vstack(chain_cb_coords),
            "centers": np.vstack(chain_centers),
            "sidechain_centers": np.vstack(chain_sidechain_centers),
        }

    return chain_coord_dict


def cal_minimize_atom_distance_map(mol1_coords, mol2_coords):
    n_res1 = len(mol1_coords)
    n_res2 = len(mol2_coords)
    if n_res1 == 0 or n_res2 == 0:
        return np.zeros((n_res1, n_res2), dtype=np.float64)

    atom_counts1 = np.asarray([len(x) for x in mol1_coords], dtype=np.int64)
    atom_counts2 = np.asarray([len(x) for x in mol2_coords], dtype=np.int64)
    if np.any(atom_counts1 == 0) or np.any(atom_counts2 == 0):
        raise ValueError("Each residue must contain at least one atom coordinate")

    all_atoms1 = np.concatenate(mol1_coords, axis=0)
    all_atoms2 = np.concatenate(mol2_coords, axis=0)
    atom_dist_map = cdist(all_atoms1, all_atoms2)

    col_starts = np.cumsum(np.concatenate(([0], atom_counts2[:-1])))
    min_over_res2 = np.minimum.reduceat(atom_dist_map, col_starts, axis=1)

    row_starts = np.cumsum(np.concatenate(([0], atom_counts1[:-1])))
    res_dist_map = np.minimum.reduceat(min_over_res2, row_starts, axis=0)
    return res_dist_map


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    D_mu = torch.linspace(D_min, D_max, D_count, dtype=torch.float32).to(device).view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1).to(dtype=torch.float32)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


def _normalize(tensor, dim=-1):
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training-set graph features with DockQ and edge labels.")
    parser.add_argument("--decoy_dir", required=True, help="Directory containing complex PDBs and decoy PDBs.")
    parser.add_argument("--feature_dir", required=True, help="Output directory for generated features and graphs.")
    parser.add_argument("--dockq_json", required=True, help="Path to JSON file containing ground-truth DockQ scores.")
    parser.add_argument("--zscale_path", required=True, help="Path to amino acid z-scale CSV file.")
    parser.add_argument("--info_dir", default=None, help="Directory containing per-structure info JSON files (default: feature_dir/info).")
    parser.add_argument("--esm_dir", default=None, help="Directory containing ESM embeddings (default: feature_dir/esm_embedding).")
    parser.add_argument("--esmif_dir", default=None, help="Directory containing ESM-IF embeddings (default: feature_dir/esmif_mutation_score).")
    parser.add_argument("--ncpus", type=int, default=8, help="Number of worker processes (default: 8).")
    args = parser.parse_args()

    PepDockQTrainFeature(
        decoy_dir=args.decoy_dir,
        feature_dir=args.feature_dir,
        dockq_json=args.dockq_json,
        zscale_path=args.zscale_path,
        ncpus=args.ncpus,
        info_dir=args.info_dir,
        esm_dir=args.esm_dir,
        esmif_dir=args.esmif_dir,
    )
