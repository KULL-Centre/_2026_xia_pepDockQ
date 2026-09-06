import os
import json
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


class PepDataset(Dataset):
    def __init__(self, pdb_list, mode, data_path, selected_decoy_path=None):

        self.pdb_list = pdb_list
        self.model_pdb_list = []
        self.pdb_file_index = {}
        self.file_load_workers = min(8, max(1, os.cpu_count() or 1))
        
        self._indices = None
        self.transform = None
        
        self.data_path = data_path
        self.mode = mode

        # Build allowed filename set from selected_decoy_path JSON if provided
        self._allowed_filenames = None
        if selected_decoy_path is not None:
            with open(selected_decoy_path) as _f:
                _selected = json.load(_f)
            self._allowed_filenames = set()
            for _pdb_id, _models in _selected.items():
                for _model_key in _models:
                    self._allowed_filenames.add(f"{_pdb_id}_model_{_model_key}.pt")
            print(f"[INFO] selected_decoy_path loaded: {len(self._allowed_filenames)} allowed model files")

        os.makedirs(self.data_path, exist_ok=True)

        if mode in {'train', 'eval'}:
            self.pdb_file_index = self.build_pdb_file_index(self.pdb_list)

        # load dataset
        if mode == 'train':
            self.pdb_list, self.data_list = self.load_data(self.pdb_list)
            print(f'After load data, the number of pdbs is {len(self.pdb_list)}, the number of samples is {len(self.data_list)}')
            print('Done!')

        elif mode == 'eval':
            self.pdb_list, self.data_list, self.model_pdb_list = self.load_eval_data(self.pdb_list)
            print(f'After load eval data, the number of pdbs is {len(self.pdb_list)}, the number of samples is {len(self.data_list)}')
            print('Done!')

        elif mode == 'valid':
            self.pdb_list, self.data_list = self.load_data([self.pdb_list[0]])
            if len(self.data_list) == 0:
                self.pdb_list = 0

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        pdb_id, data_i_path = self.data_list[idx]
        data = self.load_data_file(data_i_path, pdb_id)
        return data

    def build_pdb_file_index(self, pdb_list):
        pdb_set = set(pdb_list)
        file_index = defaultdict(list)

        for entry in os.scandir(self.data_path):
            if not entry.is_file() or not entry.name.endswith('.pt'):
                continue
            if '_model_' not in entry.name:
                continue
            if self._allowed_filenames is not None and entry.name not in self._allowed_filenames:
                continue

            pdb_id = entry.name.rsplit('_model_', 1)[0]
            if pdb_id in pdb_set:
                file_index[pdb_id].append(entry.path)

        for pdb_id in file_index:
            file_index[pdb_id].sort()

        return dict(file_index)

    def get_pdb_model_paths(self, pdb_id):
        if self.pdb_file_index:
            return self.pdb_file_index.get(pdb_id, [])

        model_paths = []
        prefix = f"{pdb_id}_model_"
        for entry in os.scandir(self.data_path):
            if not entry.is_file() or not entry.name.endswith('.pt'):
                continue
            if entry.name.startswith(prefix):
                model_paths.append(entry.path)

        model_paths.sort()
        return model_paths

    def load_data_file(self, data_i_path, pdb_id):
        data = torch.load(data_i_path, weights_only=False)
        if data.y is None:
            print(pdb_id, data_i_path)
            raise ValueError
        return data

    def load_y_from_file(self, data_i_path, pdb_id):
        data = self.load_data_file(data_i_path, pdb_id)
        return float(data.y)
    
    def load_single_data(self, pdb_id):
        model_paths = self.get_pdb_model_paths(pdb_id)
        if not model_paths:
            return []

        max_workers = min(self.file_load_workers, len(model_paths))
        if max_workers <= 1:
            return [self.load_data_file(data_i_path, pdb_id) for data_i_path in model_paths]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda path: self.load_data_file(path, pdb_id), model_paths))

    def load_data(self, pdb_list):
        data_index_list = []
        out_pdb_list = set()

        for pdb_id in tqdm(pdb_list):
            model_paths = self.get_pdb_model_paths(pdb_id)
            data_index_list.extend([(pdb_id, path) for path in model_paths])
            out_pdb_list.add(pdb_id)

        out_pdb_list = list(out_pdb_list)
        return out_pdb_list, data_index_list

    def load_eval_data(self, pdb_list):
        data_index_list = []
        out_pdb_list = set()
        model_pdb_list = []

        for pdb_id in tqdm(pdb_list):
            model_paths = self.get_pdb_model_paths(pdb_id)
            data_index_list.extend([(pdb_id, path) for path in model_paths])
            out_pdb_list.add(pdb_id)
            model_pdb_list.extend([pdb_id] * len(model_paths))

        out_pdb_list = list(out_pdb_list)
        return out_pdb_list, data_index_list, model_pdb_list


class PepDatasetPair(PepDataset):
    def __init__(self, pdb_list, mode, data_path, selected_decoy_path=None):
        super().__init__(pdb_list, mode, data_path, selected_decoy_path=selected_decoy_path)
        if mode != 'train':
            raise KeyError

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        pdb_id, path1, path2 = self.data_list[idx]
        data1 = self.load_data_file(path1, pdb_id)
        if path2 is None:
            data2 = data1
            pair_mask = 0
        else:
            data2 = self.load_data_file(path2, pdb_id)
            pair_mask = 1
        return data1, data2, pair_mask
    
    def load_data(self, pdb_list):
        data_list = []
        out_pdb_list = set()

        for pdb_id in tqdm(pdb_list):
            model_paths = self.get_pdb_model_paths(pdb_id)
            if len(model_paths) == 0:
                continue
            if len(model_paths) == 1:
                data_list.append((pdb_id, model_paths[0], None))
                out_pdb_list.add(pdb_id)
                continue
            random.shuffle(model_paths)
            pair_i_list = [(pdb_id, p1, p2) for p1, p2 in zip(model_paths, model_paths[1:] + model_paths[:1])]
            data_list.extend(pair_i_list)
            out_pdb_list.add(pdb_id)

        out_pdb_list = list(out_pdb_list)
        return out_pdb_list, data_list


