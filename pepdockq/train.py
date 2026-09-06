import os
import torch
import pickle
import argparse
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import sys
import json
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter

from pepdockq.models.hgnn import HGNN
from pepdockq.models.losses import (
    DockQPairwiseThresholdLoss,
    EdgeBinnedSoftTargetLoss,
)
from pepdockq.data import PepDataset, PepDatasetPair, PepDatasetPairWeighted
from pepdockq.metrics import (
    calculate_accuracy_per_category_numpy,
    compute_success_rate_sorted,
)
import csv

DOCKQ_THRESHOLDS = [0.23, 0.49, 0.8]

def find_optimal_pdockq_thresholds(all_dockq, all_pdockq, dockq_thresholds=None, n_steps=100):
    if dockq_thresholds is None:
        dockq_thresholds = DOCKQ_THRESHOLDS

    if all_dockq.size == 0 or all_pdockq.size == 0:
        return {str(th): th for th in dockq_thresholds}

    optimal_thresholds = {}
    candidate_thresholds = [step / n_steps for step in range(1, n_steps)]

    for dockq_th in dockq_thresholds:
        y_true = (all_dockq >= dockq_th).astype(np.int32)
        best_mcc = -2.0
        best_pdockq_th = dockq_th

        for pdockq_th in candidate_thresholds:
            y_pred = (all_pdockq >= pdockq_th).astype(np.int32)
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                continue
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_pdockq_th = pdockq_th

        optimal_thresholds[str(dockq_th)] = best_pdockq_th

    return optimal_thresholds


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab',buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass
    
def setup_experiment(args, base_dir="outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        if hasattr(args, "__dict__"):  # argparse.Namespace
            json.dump(vars(args), f, indent=4)
        else:
            json.dump(args, f, indent=4)
    print(f"[INFO] Config saved to {config_path}")

    log_path = os.path.join(exp_dir, "log.txt")
    sys.stdout = Logger(log_path)

    print(f"[INFO] Logging to {log_path}")
    model_saved_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(model_saved_dir, exist_ok=True)

    return model_saved_dir

def train(model, args, model_save_dir):

    train_list = args.train_list

    print('[RUN] Load dataset... ')
    train_data = PepDatasetPair(train_list, 'train', args.data_path)

    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model.to(args.device)

    global_loss_fn = DockQPairwiseThresholdLoss(
        lambda_pair=args.lambda_pair,
        huber_delta=args.huber_delta,
    ).to(args.device)

    local_loss_fn = EdgeBinnedSoftTargetLoss(
        num_error_bins=args.num_error_bins,
        min_error=args.min_error,
        max_error=args.max_error,
        sigma_bins=args.sigma_bins,
        lambda_error=args.lambda_error,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"[INFO] Optimizer initialized with learning rate {args.lr}")

    best_val_per_bin = {'0.0-0.4': -1e9, '0.4-0.8': -1e9, '0.8-1.0': -1e9}

    global_step = 0
    for epoch in range(args.epochs):
        model.train()

        total_loss_list = []
        local_loss_list = []
        global_reg_loss_list = []
        global_pair_loss_list = []

        for data1, data2, _ in tqdm(train_loader):
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)

            pair_error_pred1, pair_error_label1, global_pred1 = model(data1)
            pair_error_pred2, pair_error_label2, global_pred2 = model(data2)

            local_loss_1 = local_loss_fn(pair_error_pred1, pair_error_label1)
            local_loss_2 = local_loss_fn(pair_error_pred2, pair_error_label2)

            local_loss = local_loss_1 + local_loss_2
            
            global_loss, global_reg_loss, global_pair_loss = global_loss_fn(global_pred1, global_pred2, data1.y, data2.y)
            loss = global_loss + local_loss

            optimizer.zero_grad()
            loss.backward()

            total_loss_list.append(loss.detach())
            local_loss_list.append(local_loss.detach())
            global_reg_loss_list.append(global_reg_loss.detach())
            global_pair_loss_list.append(global_pair_loss.detach())

            writer.add_scalar("Loss/train", loss.item(), global_step)
            optimizer.step()
            global_step += 1

        avg_total_loss = torch.stack(total_loss_list).mean()
        avg_local_loss = torch.stack(local_loss_list).mean()
        avg_global_reg_loss = torch.stack(global_reg_loss_list).mean()
        avg_global_pair_loss = torch.stack(global_pair_loss_list).mean()

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train | reg: {avg_global_reg_loss:.4f} pair: {avg_global_pair_loss:.4f} total: {avg_total_loss:.4f} local: {avg_local_loss:.4f} ")

        val_result_all = {}
        test_result_all = {}

        for bin in ["0.0-0.4", "0.4-0.8", "0.8-1.0"]:
            print('-----------evaluate on bin:', bin, '-----------')
            valid_list = args.valid_sim_dict[bin]
            val_pearson, val_result, val_pdockq_thresholds = evaluate(
                model,
                valid_list,
                local_loss_fn,
                global_loss_fn,
                args,
                f'{bin}-valid',
                epoch,
                fixed_pdockq_thresholds={'0.23': 0.23, '0.49': 0.49, '0.8': 0.8},
            )

            test_list = args.test_sim_dict[bin]
            test_pearson, test_result, _ = evaluate(
                model,
                test_list,
                local_loss_fn,
                global_loss_fn,
                args,
                f'{bin}-test',
                epoch,
                fixed_pdockq_thresholds={'0.23': 0.23, '0.49': 0.49, '0.8': 0.8},
            )

            if val_pearson >= best_val_per_bin[bin]:
                best_val_per_bin[bin] = val_pearson
                print(f'for {bin}, best overall pearson corr: {best_val_per_bin[bin]:.4f}')

            for pdb_id in val_result:
                val_result_all[pdb_id] = val_result[pdb_id]
            for pdb_id in test_result:
                test_result_all[pdb_id] = test_result[pdb_id]
            
        torch.save(model.state_dict(), f"{model_save_dir}/model_epoch{epoch}.pth")
        with open(f"{model_save_dir}/prediction_epoch{epoch}.pkl", 'wb') as f:
            pickle.dump(
                {
                    'valid': val_result_all,
                    'test': test_result_all,
                    'valid_selected_pdockq_thresholds': val_pdockq_thresholds,
                },
                f,
            )

    return 



def evaluate(model, val_list, local_loss_fn, global_loss_fn, args, mode, epoch, fixed_pdockq_thresholds=None):

    model.eval()

    all_topk_dockqs = []
    all_best_dockq = []
    avg_spearman_corr = []
    avg_pearson_corr = []
    pdb_num = 0

    local_loss_list = []
    global_loss_list = []

    all_pdockq = []
    all_dockq = []
    results = {}


    val_data = PepDataset(val_list, 'eval', args.data_path)
    all_val_sample_pdb_ids = val_data.model_pdb_list
    val_loader = DataLoader(
        val_data,
        args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    # Collect per-PDB global predictions for downstream per-complex metrics.
    per_pdb_pred = {}
    per_pdb_target = {}

    with torch.no_grad():
        sample_cursor = 0
        for data in tqdm(val_loader, desc=f"Evaluating {mode}"):
            data = data.to(args.device)

            global_target = data.y
            pair_error_pred, pair_error_label, global_pred = model(data)
            global_loss = global_loss_fn.cal_test_loss(global_pred, global_target)
            local_loss = local_loss_fn(pair_error_pred, pair_error_label)

            local_loss_list.append(local_loss.detach())
            global_loss_list.append(global_loss.detach())

            all_pdockq.append(global_pred.reshape(-1))
            all_dockq.append(global_target.reshape(-1))

            batch_size = int(global_target.numel())
            batch_pdb_ids = all_val_sample_pdb_ids[sample_cursor: sample_cursor + batch_size]
            sample_cursor += batch_size

            global_pred_cpu = global_pred.detach().reshape(-1).cpu().numpy()
            global_target_cpu = global_target.detach().reshape(-1).cpu().numpy()
            for idx, pdb_id in enumerate(batch_pdb_ids):
                per_pdb_pred.setdefault(pdb_id, []).append(float(global_pred_cpu[idx]))
                per_pdb_target.setdefault(pdb_id, []).append(float(global_target_cpu[idx]))

    for pdb_id in per_pdb_pred:
        pdb_pred = torch.tensor(per_pdb_pred[pdb_id])
        pdb_target = torch.tensor(per_pdb_target[pdb_id])
        results[pdb_id] = list(zip(list(pdb_pred.numpy()), list(pdb_target.numpy())))

        if len(pdb_pred) >= 5:
            spearman_corr = spearmanr(pdb_pred, pdb_target).statistic
            pearson_corr = pearsonr(pdb_pred, pdb_target).statistic

            if np.isnan(spearman_corr):
                spearman_corr = 0.0
            if np.isnan(pearson_corr):
                pearson_corr = 0.0

            avg_spearman_corr.append(spearman_corr)
            avg_pearson_corr.append(pearson_corr)

            topk_dockqs = cal_topk_dockq(pdb_pred, pdb_target, k_list=[1, 3, 5])
            all_best_dockq.append(torch.max(pdb_target))
            all_topk_dockqs.append(topk_dockqs)
            pdb_num += 1

    all_pdockq = torch.cat(all_pdockq, dim=0).cpu().detach().numpy()
    all_dockq = torch.cat(all_dockq, dim=0).cpu().detach().numpy()

    spearman_corr = spearmanr(all_pdockq, all_dockq).statistic
    pearson_corr = pearsonr(all_pdockq, all_dockq).statistic

    if pdb_num > 0:
        avg_topk_dockqs = torch.mean(torch.tensor(all_topk_dockqs), dim=0).tolist()
        all_best_dockq = torch.tensor(all_best_dockq).mean()
        avg_spearman_corr = sum(avg_spearman_corr) / pdb_num
        avg_pearson_corr = sum(avg_pearson_corr) / pdb_num
    else:
        avg_topk_dockqs = [0.0, 0.0, 0.0]
        all_best_dockq = torch.tensor(0.0)
        avg_spearman_corr = 0.0
        avg_pearson_corr = 0.0

    avg_local_loss = torch.stack(local_loss_list).mean()
    avg_global_loss = torch.stack(global_loss_list).mean()

    writer.add_scalar(f"Loss/{mode}", avg_global_loss, epoch)

    print(f"{mode} | local: {avg_local_loss:.4f} global: {avg_global_loss:.4f} pdb_num: {pdb_num} | dockq: best={all_best_dockq:.3f} top1={avg_topk_dockqs[0]:.3f} top3={avg_topk_dockqs[1]:.3f} top5={avg_topk_dockqs[2]:.3f} | avg corr: spear={avg_spearman_corr:.3f} pear={avg_pearson_corr:.3f}")

    accuracy_per_category = calculate_accuracy_per_category_numpy(all_pdockq, all_dockq)
    print(f"overall corr: spear={spearman_corr:.3f} pear={pearson_corr:.3f} | overall accuracy: incorrect={accuracy_per_category['incorrect']:.3f} acceptable={accuracy_per_category['acceptable']:.3f} medium={accuracy_per_category['medium']:.3f} high={accuracy_per_category['high']:.3f}")

    for threshold in DOCKQ_THRESHOLDS:
        topk_success_rate = compute_success_rate_sorted(results, k_components=[1, 3, 5], threshold=threshold, descending=True)
        print(f"Success Rate on th={threshold:.2f}: top1={topk_success_rate[1]:.2f}% top3={topk_success_rate[3]:.2f}% top5={topk_success_rate[5]:.2f}%")

    if fixed_pdockq_thresholds is None:
        pdockq_thresholds = find_optimal_pdockq_thresholds(all_dockq, all_pdockq, dockq_thresholds=DOCKQ_THRESHOLDS)
        print("Selected pDockQ thresholds on this split (MCC-max):")
    else:
        pdockq_thresholds = fixed_pdockq_thresholds
        print("Using provided pDockQ thresholds (from validation):")

    for threshold in DOCKQ_THRESHOLDS:
        pdockq_th = pdockq_thresholds.get(str(threshold), threshold)
        print(f"  DockQ>={threshold:.2f} -> pDockQ threshold={pdockq_th:.2f}")

    # Threshold-based binary evaluation using DockQ threshold for label, and pDockQ threshold selected from valid set.
    for threshold in DOCKQ_THRESHOLDS:
        pdockq_th = pdockq_thresholds.get(str(threshold), threshold)
        y_true = (all_dockq >= threshold).astype(np.int32)
        y_pred = (all_pdockq >= pdockq_th).astype(np.int32)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mcc = matthews_corrcoef(y_true, y_pred) if (y_pred.sum() > 0 and y_pred.sum() < len(y_pred)) else 0.0

        if np.unique(y_true).size < 2:
            auroc = 0.0
            auprc = 0.0
        else:
            auroc = float(roc_auc_score(y_true, all_pdockq))
            auprc = float(average_precision_score(y_true, all_pdockq))

        print(
            f"Binary metrics @ dockq_th={threshold:.2f} pdockq_th={pdockq_th:.2f}: "
            f"recall={recall:.3f} precision={precision:.3f} specificity={specificity:.3f} "
            f"f={f_score:.3f} mcc={mcc:.3f} auroc={auroc:.3f} auprc={auprc:.3f}"
        )

    return pearson_corr, results, pdockq_thresholds

def cal_topk_dockq(pred, dockq, k_list):
    topk_dockq_list = []
    for k in k_list:
        values, indices = torch.topk(pred, k=k)
        topk_dockq = dockq[indices]
        max_dockq = torch.max(topk_dockq).item()
        topk_dockq_list.append(max_dockq)
    return  topk_dockq_list


def load_all_pdbs_with_similarity(tsv_path):
    """Load all PDB IDs with their similarity values from TSV (no filtering)."""
    pdb_sim = {}
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                pdb_sim[row['pep_id']] = {
                    'max_pdb_receptor_seq_identity': float(row['max_pdb_receptor_seq_identity']),
                }
            except (ValueError, KeyError):
                continue

    _bins = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.01)]
    _bin_labels = ['0.0-0.4', '0.4-0.8', '0.8-1.0']

    similar_dict = {}
    for (lo, hi), label in zip(_bins, _bin_labels):
        similar_dict[label] = [
            pdb_id for pdb_id in pdb_sim.keys()
            if lo <= pdb_sim.get(pdb_id, {}).get('max_pdb_receptor_seq_identity', -1) < hi
        ]
    return similar_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=list, default=['esm', 'zscale', 'dihedrals', 'esmif_mutation_score']) # 'esm','zscale','dihedrals','esmif_mutation_score'

    # loss weights
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--lambda_pair", type=float, default=1.0)
    parser.add_argument("--lambda_error", type=float, default=0.00)
    
    parser.add_argument("--num_error_bins", type=int, default=32) 
    parser.add_argument("--min_error", type=float, default=-5.0)
    parser.add_argument("--max_error", type=float, default=30.0)
    parser.add_argument("--sigma_bins", type=float, default=1.5)


    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)

    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory of pre-computed graph .pt features (output of collect_inputs.py, "
                             "typically <feature_dir>/graph)")
    parser.add_argument("--train_list", dest="train_list_path", type=str, required=True,
                        help="Text file with one training PDB id per line")
    parser.add_argument("--valid_list", dest="valid_list_path", type=str, required=True,
                        help="TSV file for validation split with columns pep_id, max_receptor_seq_identity, "
                             "max_pdb_receptor_seq_identity (see docs/data_format.md)")
    parser.add_argument("--test_list", dest="test_list_path", type=str, required=True,
                        help="TSV file for test split, same schema as --valid_list")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Root directory for experiment outputs (checkpoints, logs, tensorboard)")

    return parser.parse_args()


def _read_list(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def load_splits(args):
    """Read the split files listed in ``args`` and attach them to ``args``."""
    args.train_list = _read_list(args.train_list_path)
    args.valid_sim_dict = load_all_pdbs_with_similarity(args.valid_list_path)
    args.test_sim_dict = load_all_pdbs_with_similarity(args.test_list_path)

    print('train number:', len(args.train_list))
    print("\nTest PDBs by max_pdb_receptor_seq_identity bin:")
    for label, pdb_list in args.test_sim_dict.items():
        print(f"  {label}: {len(pdb_list)} PDBs")
    print("\nValid PDBs by max_pdb_receptor_seq_identity bin:")
    for label, pdb_list in args.valid_sim_dict.items():
        print(f"  {label}: {len(pdb_list)} PDBs")


def main():
    global writer
    args = parse_arguments()
    load_splits(args)
    model_saved_dir = setup_experiment(args, base_dir=args.output_dir)
    writer = SummaryWriter(log_dir=model_saved_dir)

    model = HGNN(num_layers=args.num_layers, hidden_channels=args.hidden_channels,
                 features=args.features, heads=args.heads, num_error_bins=args.num_error_bins)

    train(model, args, model_save_dir=model_saved_dir)


if __name__ == "__main__":
    main()

