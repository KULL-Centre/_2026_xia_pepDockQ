import pickle
import csv
import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
import torch
from torch_geometric.data import Data, HeteroData
from Bio.PDB import PDBParser, MMCIFParser
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist
import warnings
from Bio import BiopythonWarning
from torch_cluster import knn_graph, knn, radius_graph, radius
import math
import torch.nn.functional as F
import re
import argparse

warnings.simplefilter("ignore", BiopythonWarning)
three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEP": "S"
}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWYX")   # 20 standard amino acids + X
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
MAX_PEPTIDE_LENGTH = 53

class PepDockQFeature():
    def __init__(self, pdb_dir, esm_dir, esmif_dir, info_dir, feature_dir, ncpus, zscale_path, input_csv):
        self.ncpus = ncpus
        self.pocket_th = None
        self.edge_th = 10
        self.edge_type = 'radius'
        self.sidechain_type = 'sc'  # 'imputed_cb', 'sc', 'cb', 'sc'
        self.rep_atom = 'sc'  # 'ca', 'cb', 'center','sc'
        self.AA_feat_dict = self.load_AA_df(zscale_path)

        self.pdb_dir = pdb_dir
        self.esm_dir = esm_dir
        self.esmif_dir = esmif_dir
        self.info_dir = info_dir
        self.input_csv = input_csv

        assert os.path.exists(self.pdb_dir), f"PDB directory not found: {self.pdb_dir}"
        assert os.path.exists(self.esm_dir), f"ESM embedding directory not found: {self.esm_dir}"
        assert os.path.exists(self.esmif_dir), f"ESM-IF embedding directory not found: {self.esmif_dir}"
        assert os.path.exists(self.info_dir), f"Info directory not found: {self.info_dir}"

        self.distance_map_path = os.path.join(feature_dir, f'distance_map')
        self.coord_path = os.path.join(feature_dir, 'coordinate')
        self.data_path = os.path.join(feature_dir, 'graph')
        
        os.makedirs(self.distance_map_path, exist_ok=True)
        os.makedirs(self.coord_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

        self.pdb_list = self.get_pdb_list()
        self.chain_id_dict, self.chain_fasta_dict, self.pdb_list = self.get_chain_list()
        if len(self.pdb_list) == 0:
            raise ValueError(f"No valid structures with both peptide and receptor chains found in {self.pdb_dir}")
        print(f'[INFO] The number of structures: {len(self.pdb_list)}')

        print('[RUN] get_coord...')
        self.cal_multiprocess(self.pdb2coordinate, max_workers=self.ncpus)

        print('[RUN] coordinate2distance_map...')
        self.cal_multiprocess(self.coordinate2distance_map, max_workers=self.ncpus)

        print('[RUN] get_features...')
        self.cal_multiprocess(self.get_graph_data, max_workers=self.ncpus)
        print(f'[INFO] Finished processing all structures')


    def pdb2coordinate(self, pdb_id):
        pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        res_info_path = os.path.join(self.info_dir, f'{pdb_id}.json')
        coord_file = os.path.join(self.coord_path, f"{pdb_id}.pkl")

        if not os.path.exists(coord_file):
            chain_coord_dict = get_protein_coordinates(pdb_file, res_info_path)
            with open(coord_file, 'wb') as f:
                pickle.dump(chain_coord_dict, f)
        return
    
    def coordinate2distance_map(self, pdb_id): 
        coord_file = os.path.join(self.coord_path, f"{pdb_id}.pkl")
        dist_file = os.path.join(self.distance_map_path, f"{pdb_id}.pkl")

        with open(coord_file, 'rb') as f:
            chain_coords = pickle.load(f)

        peptide_chains = self.chain_id_dict[pdb_id]['peptide']
        receptor_chains = self.chain_id_dict[pdb_id]['receptor']

        peptide_coords = self.concat_all_atom_coordinates(pdb_id, peptide_chains, chain_coords)
        receptor_coords = self.concat_all_atom_coordinates(pdb_id, receptor_chains, chain_coords)
        if peptide_coords is None or receptor_coords is None:
            raise ValueError(f"Invalid coordinates for {pdb_id}")
        
        inter_mini_atom_dist_map = cal_minimize_atom_distance_map(peptide_coords, receptor_coords)
        pep_mini_atom_dist_map = cal_minimize_atom_distance_map(peptide_coords, peptide_coords)
        rec_mini_atom_dist_map = cal_minimize_atom_distance_map(receptor_coords, receptor_coords)
                
        dist_data = {
            'peptide-protein': inter_mini_atom_dist_map, 
            'peptide': pep_mini_atom_dist_map, 
            'protein': rec_mini_atom_dist_map
        }
        
        with open(dist_file, 'wb') as f:
            pickle.dump(dist_data, f)
                
        return 
    
            
    def concat_all_atom_coordinates(self, pdb_id, chain_list, chain_coords):
        mol_all_atom_coords = []
        for chain_id in chain_list:
            # print(pdb_id, chain_id, '\n', self.chain_fasta_dict[pdb_id][chain_id], '\n', chain_coords[chain_id]["sequence"])
            assert len(self.chain_fasta_dict[pdb_id][chain_id]) == len(chain_coords[chain_id]["sequence"])
            mol_all_atom_coords += chain_coords[chain_id]["all_atom_coords"]
        return mol_all_atom_coords

    def load_AA_df(self, zscale_path):
        df = pd.read_csv(zscale_path)
        AA_feat_dict = df.set_index("aa")[["gz1","gz2","gz3","gz4","gz5","onehot"]]
        return AA_feat_dict

    def get_AA_feature(self, seq):
        aa_list = list(seq)
        feat = self.AA_feat_dict.reindex(aa_list).fillna(0).to_numpy()
        return torch.tensor(feat)

    def get_sequence_feature(self, pdb_id, chain_list, esm_file):
        with open(esm_file, 'rb') as f:
            esm_emb_dict = pickle.load(f)

        mol_feat = []
        for chain_id in chain_list:
            sequence = self.chain_fasta_dict[pdb_id][chain_id]
            esm_emb = esm_emb_dict[chain_id]
            
            assert len(esm_emb) == len(sequence), print(len(esm_emb), len(sequence))

            aa_feat = self.get_AA_feature(sequence)
            chain_feat = torch.hstack([esm_emb, aa_feat])
            assert chain_feat.shape[0] == len(sequence)
            assert chain_feat.shape[1] == 1286
            mol_feat.append(chain_feat)

        mol_feat = torch.vstack(mol_feat).float()
        return mol_feat
    

    def get_graph_data(self, pdb_id):
        saved_path = os.path.join(self.data_path, f"{pdb_id}.pt")
        if os.path.exists(saved_path):
            return
    
        peptide_chains = self.chain_id_dict[pdb_id]['peptide']
        receptor_chains = self.chain_id_dict[pdb_id]['receptor']

        esm_file = os.path.join(self.esm_dir, f"{pdb_id}.pkl")
        esmif_file = os.path.join(self.esmif_dir, f"{pdb_id}.pkl")
        dist_maps_file = os.path.join(self.distance_map_path, f"{pdb_id}.pkl")
        coord_file = os.path.join(self.coord_path, f"{pdb_id}.pkl")

        pep_seq_feat = self.get_sequence_feature(pdb_id, peptide_chains, esm_file)
        rec_seq_feat = self.get_sequence_feature(pdb_id, receptor_chains, esm_file)

        pep_esmif_emb, rec_esmif_emb = get_esmif_feat(esmif_file, peptide_chains, receptor_chains)
        chain_backbone_coord_dict = get_atom_coordinates(coord_file)  

        with open(dist_maps_file, 'rb') as f:
            dist_maps_dict = pickle.load(f)  # peptide-protein, peptide, protein

        pep_backbone_coord = self.concat_coord(chain_backbone_coord_dict, peptide_chains)
        rec_backbone_coord = self.concat_coord(chain_backbone_coord_dict, receptor_chains)

        assert len(pep_backbone_coord) == len(pep_seq_feat), print(len(pep_backbone_coord), len(pep_seq_feat))
        assert len(rec_backbone_coord) == len(rec_seq_feat), print(len(rec_backbone_coord), len(rec_seq_feat))

        pep_feat = torch.cat([pep_seq_feat, pep_esmif_emb], dim=1)
        rec_feat = torch.cat([rec_seq_feat, rec_esmif_emb], dim=1)

        assert self.pocket_th is None, "Currently we do not perform pocket clipping, so the distance map is not clipped."
        selected_dist_maps_dict = dist_maps_dict

        pep_mask = torch.isfinite(pep_backbone_coord.sum(dim=(1,2)))
        pep_backbone_coord[~pep_mask] = np.inf

        rec_mask = torch.isfinite(rec_backbone_coord.sum(dim=(1,2)))
        rec_backbone_coord[~rec_mask] = np.inf    

        if len(rec_backbone_coord) == 0:
            print(f'No receptor residue for {pdb_id}')
            return

        pep_node_s = self.get_node_attr(pep_backbone_coord)
        rec_node_s = self.get_node_attr(rec_backbone_coord)

        pep_edge_s, pep_edge_index = self.get_hetero_edge_attr(True, self.edge_type, pep_backbone_coord)
        rec_edge_s, rec_edge_index = self.get_hetero_edge_attr(True, self.edge_type, rec_backbone_coord)
        pep_rec_edge_s, pep_rec_edge_index, rec_pep_edge_s, rec_pep_edge_index = self.get_hetero_edge_attr(False, self.edge_type, pep_backbone_coord, rec_backbone_coord)
        
        data = HeteroData()
        data['peptide'].seq_feat = pep_feat.float()
        data['peptide'].node_s = pep_node_s

        data['protein'].seq_feat = rec_feat.float()
        data['protein'].node_s = rec_node_s

        data['peptide', 'near', 'peptide'].edge_index = pep_edge_index
        data['peptide', 'near', 'peptide'].edge_s = pep_edge_s

        data['protein', 'near', 'protein'].edge_index = rec_edge_index
        data['protein', 'near', 'protein'].edge_s = rec_edge_s

        data['peptide', 'near', 'protein'].edge_index = pep_rec_edge_index
        data['peptide', 'near', 'protein'].edge_s = pep_rec_edge_s

        data['protein', 'near', 'peptide'].edge_index = rec_pep_edge_index
        data['protein', 'near', 'peptide'].edge_s = rec_pep_edge_s

        decoy_inter_dist = torch.tensor(selected_dist_maps_dict['peptide-protein'][pep_rec_edge_index[0], pep_rec_edge_index[1]])
        if decoy_inter_dist.ndim == 0: 
            decoy_inter_dist = decoy_inter_dist.unsqueeze(0)

        torch.save(data, saved_path)    

        return

    def cal_rep_atom_coord(self, atoms_coord):
        rep_atom_col_dict = {'ca':1, 'cb':4, 'sc':5, 'center':6}
        rep_atom_coord = atoms_coord[:, rep_atom_col_dict[self.rep_atom], :]
        return rep_atom_coord

    def cal_sidechain_coord(self, atoms_coord):
        if self.sidechain_type == 'imputed_cb':
            sidechain_coord = self.cal_imputed_cb(atoms_coord)
        else:
            sidechain_col_dict = {'cb':4, 'sc':5}
            sidechain_coord = atoms_coord[:, sidechain_col_dict[self.sidechain_type], :]
        return sidechain_coord


    def cat_dist_map(self, dist_map_dict):
        pep_pep_dist = dist_map_dict['peptide']
        pep_rec_dist = dist_map_dict['peptide-protein']
        rec_rec_dist = dist_map_dict['protein']
        upper = np.concat([pep_pep_dist, pep_rec_dist], axis=1)
        down = np.concat([pep_rec_dist.T, rec_rec_dist], axis=1)
        dist_map = np.concat([upper, down], axis=0)
        return dist_map

    def get_node_attr(self, coords_):   # N, CA, C, O, CB, sidechain_center, center
        backbone_coords = coords_[:, :4, :]  # N, CA, C, O
        dihedrals = self._dihedrals(backbone_coords)                         
        node_s = dihedrals.float()
        return node_s
    
    def cal_imputed_cb(self, coords):
        X_ca = coords[:,1,:]
        b = coords[:,1,:] - coords[:,0,:]
        c = coords[:,2,:] - coords[:,1,:]
        a = torch.cross(b, c, dim=-1)
        X_cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X_ca  # imputed Cb position
        return X_cb

    def get_hetero_edge_attr(self, homo_graph, edge_type, mol1_coords_, mol2_coords_=None):

        mol1_backbone_coords = mol1_coords_[:, :4, :]  # N, CA, C, O
        mol1_sidechain = self.cal_sidechain_coord(mol1_coords_)

        num_positional_embeddings = 16
        if homo_graph:
            assert mol2_coords_ is None
            mol_X = self.cal_rep_atom_coord(mol1_coords_)

            if edge_type == 'radius':
                edge_index = radius_graph(mol_X, r=self.edge_th, loop=True)
            elif edge_type == 'knn':
                raise NotImplementedError
                edge_index = knn_graph(mol_X, k=30) 
            else:
                raise KeyError

            pos_embeddings = self._positional_embeddings(edge_index, num_positional_embeddings)  
            mol2_backbone_coords = mol1_backbone_coords
            mol2_sidechain = mol1_sidechain
        else:
            assert mol2_coords_ is not None
            mol2_backbone_coords = mol2_coords_[:, :4, :] # N, CA, C, O
            mol2_sidechain = self.cal_sidechain_coord(mol2_coords_)
            
            mol1_X = self.cal_rep_atom_coord(mol1_coords_)
            mol2_X = self.cal_rep_atom_coord(mol2_coords_)

            if edge_type == 'radius':
                edge_index = radius(mol2_X, mol1_X, r=self.edge_th)
            elif edge_type == 'knn':
                raise NotImplementedError
                edge_index = knn(mol2_X, mol1_X, k=30)
            else:
                raise KeyError
            pos_embeddings = torch.zeros((edge_index.shape[1], num_positional_embeddings))  # set to 0 for cross-chain edges
            

        RBF_all, vec_all = cal_edge_attr(mol1_backbone_coords, mol1_sidechain, mol2_backbone_coords, mol2_sidechain, edge_index)
        edge_s = torch.cat([pos_embeddings, RBF_all], dim=-1).float()
        edge_s = torch.nan_to_num(edge_s)

        assert edge_index.shape[1] == pos_embeddings.shape[0] == RBF_all.shape[0] == vec_all.shape[0]

        if homo_graph:
            return edge_s, edge_index
        else:
            rev_edge_index = edge_index.flip(0)
            rev_RBF_all, rev_vec_all = cal_edge_attr(mol2_backbone_coords, mol2_sidechain, mol1_backbone_coords, mol1_sidechain, rev_edge_index)
            rev_edge_s = torch.cat([pos_embeddings, rev_RBF_all], dim=-1).float()
            rev_edge_s = torch.nan_to_num(rev_edge_s)

            return edge_s, edge_index, rev_edge_s, rev_edge_index

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None, chain_ids=None, 
                               period_range=[2, 1000], num_positional_embeddings=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or num_positional_embeddings
        d = edge_index[0] - edge_index[1]
        

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)

        if chain_ids is not None:
            isin_diff_chain = chain_ids[edge_index[0]] != chain_ids[edge_index[1]]
            E[isin_diff_chain] = 0
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _imputed_sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 

    def _sidechains(self, X_ca, X_sidechain):  # X_sidechain can be CB or the center of sidechain
        vectors = X_ca - X_sidechain
        rbf = _rbf(vectors.norm(dim=-1), D_count=16)
        vec = _normalize(vectors).unsqueeze(-2)
        return rbf, vec 


    def concat_coord(self, chain_coord_dict, chain_list):
        coord = []
        for chainid in chain_list:
            coord.append(chain_coord_dict[chainid])
        coord = torch.concat(coord, dim=0)
        return coord
    
    def get_pdb_list(self):
        with open(self.input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "decoy_id" not in reader.fieldnames:
                raise ValueError(f"{self.input_csv} must contain a decoy_id column")
            pdb_list = [row["decoy_id"].strip() for row in reader]
        return sorted(pdb_list)
    
    def get_chain_list(self):

        chain_id_dict = {}
        chain_fasta_dict = {}
        selected_pdb_list = []

        for pdbid in tqdm(self.pdb_list):
            res_info_path = os.path.join(self.info_dir, f'{pdbid}.json')
            if not os.path.exists(res_info_path):
                print(f'[WARN] Residue info not found for {pdbid}, skipped.')
                continue
            with open(res_info_path) as f:
                res_info = json.load(f)

            available_chains = list(res_info["chains"].keys())
            chain_roles = res_info.get("chain_roles", {})
            peptide_chains = chain_roles.get("peptide", [])
            receptor_chains = chain_roles.get("receptor", [])

            if not peptide_chains or not receptor_chains:
                raise ValueError(f"{pdbid} is missing peptide or receptor chain roles in {res_info_path}")

            missing_chains = [cid for cid in peptide_chains + receptor_chains if cid not in available_chains]
            if missing_chains:
                raise ValueError(f"{pdbid} assigns unavailable chain(s) {missing_chains}; available chains: {available_chains}")

            # for cid in peptide_chains:
            #     seq_len = len(res_info["chains"][cid]["sequence"])
                # if seq_len > MAX_PEPTIDE_LENGTH:
                #     raise ValueError(f"{pdbid}: peptide chain '{cid}' exceeds {MAX_PEPTIDE_LENGTH} residues (length={seq_len})")
            # for cid in receptor_chains:
            #     seq_len = len(res_info["chains"][cid]["sequence"])
                # if seq_len <= MAX_PEPTIDE_LENGTH:
                #     raise ValueError(f"{pdbid}: receptor chain '{cid}' must exceed {MAX_PEPTIDE_LENGTH} residues (length={seq_len})")

            chain_fasta_dict[pdbid] = {}
            for chain_id in available_chains:
                sequence = res_info["chains"][chain_id]["sequence"]
                chain_fasta_dict[pdbid][chain_id] = sequence

            chain_id_dict[pdbid] = {'peptide': peptide_chains, 'receptor': receptor_chains}
            selected_pdb_list.append(pdbid)

        return chain_id_dict, chain_fasta_dict, selected_pdb_list



    def cal_multiprocess(self, func, max_workers):
        results = []
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for pdb_id in self.pdb_list:
                tasks.append(executor.submit(func, pdb_id))

            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
                results.append(future.result())
        return results


def get_atom_coordinates(coord_file):
    with open(coord_file, 'rb') as f:
        chain_coord_dict = pickle.load(f)

    chain_residue_coord_dict = {}
    for chain_id, coord_dict in chain_coord_dict.items():    
        # if atom_type == 'CA':  
        #     rep_atom_coord = coord_dict["backbone_atom_coords"][:, 1, :] # L * 4 * 3
        atoms_coord = [coord_dict["backbone_atom_coords"]]
        atoms_coord.append(coord_dict["cb_coords"][:, None, :])
        atoms_coord.append(coord_dict["sidechain_centers"][:, None, :])
        atoms_coord.append(coord_dict["centers"][:, None, :])
        atoms_coord = np.concatenate(atoms_coord, axis=1)  # N, CA, C, O, CB, sidechain_center, center

        chain_residue_coord_dict[chain_id] = torch.tensor(atoms_coord)
    return chain_residue_coord_dict


def get_esmif_feat(esmif_file, peptide_chains, receptor_chains):
    with open(esmif_file, 'rb') as f:
        esmif_emb_dict = pickle.load(f)

    pep_esmif_emb = []
    for chain_id in peptide_chains:
        pep_esmif_emb.append(torch.cat([esmif_emb_dict['multimer'][chain_id], esmif_emb_dict['monomer'][chain_id]], dim=1))
    pep_esmif_emb = torch.cat(pep_esmif_emb, dim=0)

    rec_esmif_emb = []
    for chain_id in receptor_chains:
        rec_esmif_emb.append(torch.cat([esmif_emb_dict['multimer'][chain_id], esmif_emb_dict['monomer'][chain_id]], dim=1))
    rec_esmif_emb = torch.cat(rec_esmif_emb, dim=0)

    return pep_esmif_emb, rec_esmif_emb


def cal_edge_attr(mol1_backbone_coords, mol1_sidechain, mol2_backbone_coords, mol2_sidechain, edge_index):

    N1 = mol1_backbone_coords[:,0,:][edge_index[0]]
    Ca1 = mol1_backbone_coords[:,1,:][edge_index[0]]
    C1 = mol1_backbone_coords[:,2,:][edge_index[0]]
    O1 = mol1_backbone_coords[:,3,:][edge_index[0]]
    Cb1 = mol1_sidechain[edge_index[0]]

    N2 = mol2_backbone_coords[:,0,:][edge_index[1]]
    Ca2 = mol2_backbone_coords[:,1,:][edge_index[1]]
    C2 = mol2_backbone_coords[:,2,:][edge_index[1]]
    O2 = mol2_backbone_coords[:,3,:][edge_index[1]]
    Cb2 = mol2_sidechain[edge_index[1]]

    RBF_all = []
    vec_all = []
    for atom1 in [N1, Ca1, C1, O1, Cb1]:
        for atom2 in [N2, Ca2, C2, O2, Cb2]:
            vec = atom1-atom2
            RBF_all.append(_rbf(vec.norm(dim=-1))) 
            vec_all.append(_normalize(vec).unsqueeze(-2))
    RBF_all = torch.cat(RBF_all, dim=1)
    vec_all = torch.cat(vec_all, dim=1)

    return RBF_all, vec_all


def parse_res_idx_biopython(res_idx: str):
    """
    '10'  -> (' ', 10, ' ')
    '10A' -> (' ', 10, 'A')
    """
    m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", res_idx.strip())

    if m is None:
        raise ValueError(f"Invalid res_idx: {res_idx}")

    resseq = int(m.group(1))
    icode = m.group(2) if m.group(2) else " "
    return (" ", resseq, icode)

def get_protein_coordinates(pdb_path, res_info_path):

    with open(res_info_path) as f:
        res_info = json.load(f)

    ext = os.path.splitext(pdb_path)[1].lower()
    if ext == ".pdb":
        parser = PDBParser(QUIET=True)
    elif ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}, only .pdb and .cif are supported.")

    structure = parser.get_structure("protein", pdb_path)

    chain_coord_dict = {}
    for chain in structure[0]:
        chain_id = chain.id

        residue_dict_ = {res.id: res for res in chain.get_residues()}
        residue_dict = {}
        for k, v in residue_dict_.items():
            residue_dict[(" ", k[1], k[2])] = v

        selected_residues = []
        res_idx_list = res_info["chains"][chain_id]["res_idx"]
        for res_idx in res_idx_list:
            
            res_id = parse_res_idx_biopython(res_idx)
            # print(res_idx, res_id)
            assert res_id in residue_dict
            selected_residues.append(residue_dict[res_id])

        # seen = set()
        sequence = ''
        chain_all_atom_coords = []
        chain_backbone_atom_coords = []
        chain_cb_coords = []
        chain_centers = []
        chain_sidechain_centers = []
        
        for res in selected_residues:
            # remove water
            resname = res.get_resname().strip()
            if res.id[0] == 'W':
                continue
            if resname in ('HOH', 'WAT', 'H2O'):
                continue

            if resname in three_to_one:
                aa = three_to_one[resname]
            else:
                aa = 'X'

            sequence += aa

            all_atom_coords = []
            sidechain_atom_coords = []
            for atom in res:
                atom_name = atom.get_name()
                if atom_name.startswith("H"): # skip hydrogen atoms
                    continue
                all_atom_coords.append(atom.get_coord())  # all atoms
                if atom_name not in ['N', 'CA', 'C', 'O']: # side chain atoms
                    sidechain_atom_coords.append(atom.get_coord())

            # backbone atoms (for non-standard residues or missing backbone atoms, use the following rough fallback logic; may be improved later)
            n_coord = res["N"].get_coord() if 'N' in res else all_atom_coords[0]
            ca_coord = res["CA"].get_coord() if 'CA' in res else n_coord
            c_coord = res["C"].get_coord() if 'C' in res else ca_coord
            o_coord = res["O"].get_coord() if 'O' in res else c_coord
            bachbone_atom_coords = np.vstack([n_coord, ca_coord, c_coord, o_coord]).reshape(-1, 4, 3)  # 1 * 4 * 3

            all_atom_coords = np.vstack(all_atom_coords)
            if len(sidechain_atom_coords) == 0:
                sidechain_atom_coords.append(ca_coord)
            sidechain_atom_coords = np.vstack(sidechain_atom_coords)

            # coordinates carrying side-chain info: CB, residue center, side-chain center
            cb_coord = res["CB"].get_coord() if 'CB' in res else ca_coord
            center = np.mean(all_atom_coords, axis=0)
            sidechain_center = np.mean(sidechain_atom_coords, axis=0)

            # append to the chain-level lists
            chain_all_atom_coords.append(all_atom_coords)
            chain_backbone_atom_coords.append(bachbone_atom_coords)
            chain_cb_coords.append(cb_coord)
            chain_centers.append(center)
            chain_sidechain_centers.append(sidechain_center)

        # print(structure_file)
        # print(sequence)
        # print(res_info["chains"][chain_id]["sequence"])
        assert len(sequence) == len(res_info["chains"][chain_id]["sequence"])
        coord_dict = {
            "sequence": sequence,
            "all_atom_coords": chain_all_atom_coords,
            "backbone_atom_coords": np.concatenate(chain_backbone_atom_coords, axis=0),  # L * 4 * 3
            "cb_coords": np.vstack(chain_cb_coords),
            "centers": np.vstack(chain_centers),
            "sidechain_centers": np.vstack(chain_sidechain_centers)
        }
    
        chain_coord_dict[chain_id] = coord_dict
        
    return chain_coord_dict


def cal_minimize_atom_distance_map(mol1_coords, mol2_coords):
    """Compute residue-wise minimum heavy-atom distance map using vectorized ops.

    mol1_coords / mol2_coords: list-like, each element is an array of shape (n_atoms, 3)
    return: (n_res1, n_res2) ndarray where each entry is the min atom-atom distance
    between two residues.
    """
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

    # One global atom-atom distance matrix (SciPy C implementation), then
    # grouped minima along residue boundaries for both dimensions.
    atom_dist_map = cdist(all_atoms1, all_atoms2)

    col_starts = np.cumsum(np.concatenate(([0], atom_counts2[:-1])))
    min_over_res2 = np.minimum.reduceat(atom_dist_map, col_starts, axis=1)

    row_starts = np.cumsum(np.concatenate(([0], atom_counts1[:-1])))
    res_dist_map = np.minimum.reduceat(min_over_res2, row_starts, axis=0)

    return res_dist_map

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, dtype=torch.float32).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1).to(dtype=torch.float32)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


        
def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDB files to extract features for pepDockQ.")
    parser.add_argument('--pdb_dir', required=True, help='Directory containing input PDB structures.')
    parser.add_argument('--esm_dir', required=True, help='Directory containing ESM embeddings (step2 output).')
    parser.add_argument('--esmif_dir', required=True, help='Directory containing ESM-IF embeddings (step1 output).')
    parser.add_argument('--info_dir', required=True, help='Directory containing per-structure info JSON files.')
    parser.add_argument('--feature_dir', required=True, help='Output directory for the generated features/graphs.')
    parser.add_argument('--zscale_path', required=True, help='Path to the amino acid z-scale CSV file.')
    parser.add_argument('--ncpus', type=int, default=4, help='Number of worker processes (default: 4).')
    parser.add_argument('--input_csv', required=True, help='CSV with decoy_id, receptor_chains, and peptide_chains.')
    args = parser.parse_args()

    PepDockQFeature(args.pdb_dir, args.esm_dir, args.esmif_dir, args.info_dir, args.feature_dir, args.ncpus, args.zscale_path, args.input_csv)
