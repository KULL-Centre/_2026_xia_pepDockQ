import numpy as np
from typing import Dict, List, Tuple

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


