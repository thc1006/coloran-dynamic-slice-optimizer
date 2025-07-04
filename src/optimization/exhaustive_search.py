# src/optimization/exhaustive_search.py
import numpy as np

def exhaustive_optimizer(evaluator_func, total_rbgs=17):
    """
    窮舉搜尋最佳分配方案。

    Args:
        evaluator_func (function): 接受分配列表並返回效率分數的函數。
        total_rbgs (int): 可用的總資源區塊組數量。

    Returns:
        tuple: (最佳分配方案, 最高效率分數)
    """
    allocations = [
        [a, b, total_rbgs - a - b]
        for a in range(1, total_rbgs - 1)
        for b in range(1, total_rbgs - a)
        if total_rbgs - a - b >= 1
    ]

    if not allocations:
        return [int(total_rbgs/3)] * 3, 0

    efficiencies = evaluator_func(allocations)
    best_idx = np.argmax(efficiencies)
    return allocations[best_idx], efficiencies[best_idx]

