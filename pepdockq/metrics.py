import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, average_precision_score
from typing import Dict, List, Tuple, Iterable, Optional

Pair = Tuple[float, float]  # (pred_dockq, real_dockq)
Results = Dict[str, List[Pair]]

def compute_success_rate_sorted(
    results: Results,
    k_components: list,
    threshold: float = 0.49,
    sort_by: str = "pred",      # "pred" or "real"
    descending: bool = True,
    min_preds: int = 1,         # case 至少要有多少预测才计入分母
) -> dict:
    """
    Success criterion (Top-K):
      For each case, sort predictions (default by pred_dockq desc),
      take top K, and if any real_dockq >= threshold -> case success.

    Args:
        results: {case_id: [(pred_dockq, real_dockq), ...]}
        k: consider top-k predictions after sorting
        threshold: DockQ success threshold on real_dockq
        sort_by: "pred" (default) sorts by pred_dockq; "real" sorts by real_dockq
        descending: sort order (usually True)
        min_preds: cases with < min_preds predictions are excluded from denominator

    Returns:
        success_rate (%)
    """

    if sort_by not in {"pred", "real"}:
        raise ValueError("sort_by must be 'pred' or 'real'")

    topk_success_rates = {}
    for k in k_components:

        sort_idx = 0 if sort_by == "pred" else 1

        total_cases = 0
        success_cases = 0

        for case_id, pairs in results.items():
            if not pairs or len(pairs) < min_preds:
                continue

            total_cases += 1

            # 1) sort
            pairs_sorted = sorted(pairs, key=lambda x: x[sort_idx], reverse=descending)

            # 2) take top-k (if fewer than k, just take all)
            topk = pairs_sorted[:k]

            # 3) success if any real_dockq >= threshold
            if any(real >= threshold for _, real in topk):
                success_cases += 1

        if total_cases == 0:
            topk_success_rates[k] = 0.0
            continue
        
        topk_success_rates[k] = success_cases / total_cases * 100.0

    return topk_success_rates



def classify_dockq_numpy(scores):
    """将分数分类为整数类别：0=incorrect, 1=acceptable, 2=medium, 3=high"""
    categories = np.zeros_like(scores, dtype=int)
    categories[(scores >= 0.23) & (scores < 0.49)] = 1  # acceptable
    categories[(scores >= 0.49) & (scores < 0.8)] = 2  # medium
    categories[(scores >= 0.8)] = 3  # high
    return categories
 



def calculate_accuracy_per_category_numpy(predict_array, label_array):
    # 分类
    pred_categories = classify_dockq_numpy(predict_array)
    true_categories = classify_dockq_numpy(label_array)

    # 定义类别
    categories = [0, 1, 2, 3]  # 0=incorrect, 1=acceptable, 2=medium, 3=high
    correct_counts = np.zeros(len(categories), dtype=int)
    total_counts = np.zeros(len(categories), dtype=int)

    # 统计每个类别的正确预测数量和总数量
    for cat in categories:
        total_counts[cat] = np.sum(true_categories == cat)
        correct_counts[cat] = np.sum((pred_categories == cat) & (true_categories == cat))

    # 计算每个类别的准确性
    accuracy_per_category = {
        "incorrect": correct_counts[0] / total_counts[0] if total_counts[0] > 0 else 0,
        "acceptable": correct_counts[1] / total_counts[1] if total_counts[1] > 0 else 0,
        "medium": correct_counts[2] / total_counts[2] if total_counts[2] > 0 else 0,
        "high": correct_counts[3] / total_counts[3] if total_counts[3] > 0 else 0,
    }

    return accuracy_per_category


def eval_metrics(probs, targets):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert isinstance(probs, np.ndarray)
    assert isinstance(targets, np.ndarray)

    threshold_list = []
    for i in range(1, 50):
        threshold_list.append(i / 50.0)

    if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
        fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(),
                                         y_score=probs.detach().cpu().numpy())
    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
        fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError
    auc_ = auc(x=fpr, y=tpr)

    ap_ = average_precision_score(targets, probs)

    threshold_best, rec_best, pre_best,F1_best, spe_best, mcc_best, pred_bi_best = 0, 0, 0,0, 0, -1, None
    for threshold in threshold_list:
        threshold, rec, pre,F1, spe, mcc, pred_bi = th_eval_metrics_withoutAUC(threshold, probs, targets)
        if mcc > mcc_best:
            threshold_best, rec_best, pre_best,F1_best, spe_best, mcc_best, pred_bi_best = threshold, rec, pre,F1, spe, mcc, pred_bi

    return threshold_best, rec_best, pre_best,F1_best, spe_best, mcc_best, auc_, ap_



def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    if p*fp1 == 0:
        roce = 0
    else:
        roce = (tp1*n)/(p*fp1)
    return roce

def th_eval_metrics_withoutAUC(threshold, probs, targets):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert isinstance(probs, np.ndarray)
    assert isinstance(targets, np.ndarray)

    pred_bi = np.abs(np.ceil(probs - threshold))

    try:
        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
    except:
        a = confusion_matrix(targets, pred_bi).ravel()
        a=0
    if tp >0 :
        rec = tp / (tp + fn)
    else:
        rec = 1e-8
    if tp >0:
        pre = tp / (tp + fp)
    else:
        pre = 1e-8
    if tn>0:
        spe = tn / (tn + fp)
    else:
        spe = 1e-8
    mcc = matthews_corrcoef(targets, pred_bi)
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0

    return threshold, rec, pre,F1, spe, mcc, pred_bi


def th_eval_metrics(threshold, probs, targets):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert isinstance(probs, np.ndarray)
    assert isinstance(targets, np.ndarray)

    fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs)
    auc_ = auc(x=fpr, y=tpr)
    ap_ = average_precision_score(targets, probs)

    re0_5 = getROCE(probs.tolist(), targets.tolist(), 0.5)
    re1 = getROCE(probs.tolist(), targets.tolist(), 1)
    re2 = getROCE(probs.tolist(), targets.tolist(), 2)
    re5 = getROCE(probs.tolist(), targets.tolist(), 5)

    pred_bi = np.abs(np.ceil(probs - threshold))

    try:
        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
    except:
        a = confusion_matrix(targets, pred_bi).ravel()
        a = 0
    if tp > 0:
        rec = tp / (tp + fn)
    else:
        rec = 1e-8
    if tp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 1e-8
    if tn > 0:
        spe = tn / (tn + fp)
    else:
        spe = 1e-8
    mcc = matthews_corrcoef(targets, pred_bi)
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0

    return threshold, rec, pre, F1, spe, mcc, auc_, ap_,re0_5,re1,re2,re5,  pred_bi


