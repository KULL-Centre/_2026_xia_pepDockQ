import os
import json
import csv
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


class PepDatasetPairWeighted(PepDataset):
    def __init__(self, pdb_list, mode, data_path, N_sample, selected_decoy_path=None):
        self.N_sample = N_sample
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

            item_list = [(path, self.load_y_from_file(path, pdb_id)) for path in model_paths]
            pair_i_list = build_pairs_for_complex(item_list, K=self.N_sample)
            pair_path_list = [(pdb_id, item1[0], item2[0]) for item1, item2 in pair_i_list]
            data_list.extend(pair_path_list)
            out_pdb_list.add(pdb_id)

        out_pdb_list = list(out_pdb_list)
        return out_pdb_list, data_list


def load_cluster_member_map(cluster_tsv_path):
    """
    Build mapping: pdb_id -> list of pdb_ids in the same cluster (including itself).
    """
    member_map = {}

    with open(cluster_tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            members_raw = row.get("members", "")
            if not members_raw:
                continue

            members = [item.strip() for item in members_raw.split(',') if item.strip()]
            if not members:
                continue

            for member in members:
                member_map[member] = members

    return member_map


class PepDatasetRandomPair(PepDataset):
    def __init__(self, pdb_list, mode, data_path, cluster_tsv_path, N_sample=None, selected_decoy_path=None):
        self.cluster_tsv_path = cluster_tsv_path
        self.N_sample = N_sample
        self.cluster_member_map = load_cluster_member_map(cluster_tsv_path)
        self._cluster_model_path_cache = {}

        super().__init__(pdb_list, mode, data_path, selected_decoy_path=selected_decoy_path)
        if mode != 'train':
            raise KeyError

    def len(self):
        return len(self.data_list)

    def _get_cached_model_paths(self, pdb_id):
        if pdb_id not in self._cluster_model_path_cache:
            self._cluster_model_path_cache[pdb_id] = self.get_pdb_model_paths(pdb_id)
        return self._cluster_model_path_cache[pdb_id]

    def _sample_cluster_decoy_path(self, pdb_id):
        cluster_members = self.cluster_member_map.get(pdb_id, [])
        if not cluster_members:
            return None

        random_members = list(cluster_members)
        random.shuffle(random_members)

        for other_pdb_id in random_members:
            if other_pdb_id == pdb_id:
                continue

            other_model_paths = self._get_cached_model_paths(other_pdb_id)
            if other_model_paths:
                return random.choice(other_model_paths)

        return None

    def get(self, idx):
        pdb_id, path1, path2, path3 = self.data_list[idx]

        data1 = self.load_data_file(path1, pdb_id)

        if path2 is None:
            data2 = data1
            pair_12_mask = 0
        else:
            data2 = self.load_data_file(path2, pdb_id)
            pair_12_mask = 1

        if path3 is None:
            data3 = data1
            pair_13_mask = 0
        else:
            data3 = self.load_data_file(path3, pdb_id)
            pair_13_mask = 1

        return data1, data2, data3, pair_12_mask, pair_13_mask

    def load_data(self, pdb_list):
        data_list = []
        out_pdb_list = set()

        for pdb_id in tqdm(pdb_list):
            model_paths = self.get_pdb_model_paths(pdb_id)
            if len(model_paths) == 0:
                continue

            if len(model_paths) == 1:
                path1 = model_paths[0]
                path2 = None
                path3 = self._sample_cluster_decoy_path(pdb_id)
                data_list.append((pdb_id, path1, path2, path3))
                out_pdb_list.add(pdb_id)
                continue

            if self.N_sample is not None and self.N_sample > 0:
                item_list = [(path, self.load_y_from_file(path, pdb_id)) for path in model_paths]
                pair_i_list = build_pairs_for_complex(item_list, K=self.N_sample)
                pair_12_list = [(item1[0], item2[0]) for item1, item2 in pair_i_list]
            else:
                random.shuffle(model_paths)
                pair_12_list = [(p1, p2) for p1, p2 in zip(model_paths, model_paths[1:] + model_paths[:1])]

            for path1, path2 in pair_12_list:
                path3 = self._sample_cluster_decoy_path(pdb_id)
                data_list.append((pdb_id, path1, path2, path3))

            out_pdb_list.add(pdb_id)

        out_pdb_list = list(out_pdb_list)
        return out_pdb_list, data_list


def build_pairs_for_complex(data_i_list, K, thr=0.49, min_gap=0.02):
    """
    data_i_list: list of [data, y]
    return: list of ((data1,y1),(data2,y2))
    """
    ys = [float(item[1]) for item in data_i_list]
    idx = list(range(len(data_i_list)))

    low = [i for i in idx if ys[i] < thr]
    high = [i for i in idx if ys[i] >= thr]

    # mid_idx = [i for i in idx if (mid[0] <= ys[i] <= mid[1])]

    pairs = []
    pairs_y = []
    # 1) 跨阈值 pairs
    n_cross = int(K * 0.4)
    if low and high:
        for _ in range(n_cross):
            i = random.choice(low)
            j = random.choice(high)
            # 保证差值足够大，减少噪声pair
            if abs(ys[j] - ys[i]) >= min_gap:
                pairs.append((data_i_list[i], data_i_list[j]))
                pairs_y.append((data_i_list[i][1], data_i_list[j][1]))

    # 2) high内部 pairs
    n_high = int(K * 0.4)
    if len(high) >= 2:
        for _ in range(n_high):
            i, j = random.sample(high, 2)
            if abs(ys[j] - ys[i]) >= 0:
                pairs.append((data_i_list[i], data_i_list[j]))
                pairs_y.append((data_i_list[i][1], data_i_list[j][1]))


    # 2) low内部 pairs
    n_lower = int(K * 0.2)
    if len(low) >= 2:
        for _ in range(n_lower):
            i, j = random.sample(low, 2)
            if abs(ys[j] - ys[i]) >= min_gap:
                pairs.append((data_i_list[i], data_i_list[j]))
                pairs_y.append((data_i_list[i][1], data_i_list[j][1]))


    # 3) 随机 pairs（补充全局排序）
    n_ext = K - len(pairs)
    for _ in range(n_ext):
        i, j = random.sample(idx, 2)
        if abs(ys[j] - ys[i]) >= min_gap:
            pairs.append((data_i_list[i], data_i_list[j]))
            pairs_y.append((data_i_list[i][1], data_i_list[j][1]))


    # 去重（可选）：防止重复pair太多
    pairs = list({(id(a[0]), id(b[0])): (a,b) for (a,b) in pairs}.values())

    return pairs
