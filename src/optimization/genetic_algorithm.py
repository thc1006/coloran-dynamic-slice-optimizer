# src/optimization/genetic_algorithm.py
import numpy as np

def genetic_optimizer(evaluator_func, total_rbgs=17, pop_size=80, generations=15):
    """
    使用遺傳演算法搜尋最佳分配方案。

    Args:
        evaluator_func (function): 接受分配列表並返回效率分數的函數。
        total_rbgs (int): 可用的總資源區塊組數量。
        pop_size (int): 族群大小。
        generations (int): 迭代世代數。

    Returns:
        tuple: (最佳分配方案, 最高效率分數)
    """
    # 初始化族群
    population = [np.random.multinomial(total_rbgs - 3, [1/3.]*3) + 1 for _ in range(pop_size)]

    for _ in range(generations):
        fitness = evaluator_func(population)
        if np.sum(fitness) == 0: break

        selection_probs = fitness / np.sum(fitness)

        # 選擇、交叉、變異
        new_population = []
        for _ in range(pop_size):
            # 選擇
            parents_idx = np.random.choice(len(population), 2, p=selection_probs, replace=False)
            p1, p2 = population[parents_idx[0]], population[parents_idx[1]]

            # 交叉
            child = np.concatenate((p1[:1], p2[1:])).tolist()

            # 修復
            child_sum = sum(child)
            if child_sum > 0:
                child = [int(c * total_rbgs / child_sum) for c in child]
                child[0] = total_rbgs - sum(child[1:])
            child = [max(1,c) for c in child] # 確保最小值為1
            child[-1] = total_rbgs - sum(child[:-1])
            child = [max(1,c) for c in child]

            new_population.append(child)
        population = new_population

    final_fitness = evaluator_func(population)
    best_idx = np.argmax(final_fitness)
    return population[best_idx], final_fitness[best_idx]
