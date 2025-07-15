# src/optimization/allocator.py

import numpy as np
import pandas as pd
import time
import joblib

class SliceResourceAllocator:
    """
    網路切片資源分配器。
    - 使用預訓練的機器學習模型來預測不同資源分配方案的效率。
    - 實現窮舉搜尋和遺傳演算法來尋找最佳分配。
    """
    def __init__(self, model_path, scaler_path, features, total_rbgs=17):
        print("🚀 初始化資源分配器...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = features
        self.total_rbgs = total_rbgs
        self.timeout_s = 600
        print("✅ 模型與標準化器載入成功。")

    def _predict_efficiency(self, feature_matrix):
        """使用載入的模型預測效率。"""
        X_scaled = self.scaler.transform(feature_matrix.astype(np.float32))
        return np.clip(self.model.predict(X_scaled), 0.0, 1.0)

    def _get_feature_matrix(self, state, allocations):
        """為多個分配方案產生特徵矩陣。"""
        feature_matrix = []
        for alloc in allocations:
            for sid in range(3): # eMBB, URLLC, mMTC
                s = state.copy()
                s['slice_id'] = sid
                s['allocated_rbgs'] = alloc[sid]
                # 簡化的特徵生成，實際應用中可能更複雜
                s['prb_utilization'] = min(1.0, s.get('sum_requested_prbs', 1) / max(1, s['allocated_rbgs']))
                feature_row = [s.get(f, 0) for f in self.features]
                feature_matrix.append(feature_row)
        return np.array(feature_matrix)

    def _evaluate_allocations(self, state, allocations):
        """評估一組分配方案的平均效率。"""
        feature_matrix = self._get_feature_matrix(state, allocations)
        predictions = self._predict_efficiency(feature_matrix)
        return predictions.reshape(len(allocations), 3).mean(axis=1)

    def optimize_exhaustive(self, state):
        """窮舉搜尋最佳分配。"""
        allocations = [[a, b, self.total_rbgs - a - b]
                       for a in range(1, self.total_rbgs - 1)
                       for b in range(1, self.total_rbgs - a)
                       if self.total_rbgs - a - b >= 1]
        if not allocations:
            return state.get('current_rbg_allocation', [5, 7, 5]), 0
        
        efficiencies = self._evaluate_allocations(state, allocations)
        best_idx = np.argmax(efficiencies)
        return allocations[best_idx], efficiencies[best_idx]

    def optimize_genetic(self, state, pop_size=80, generations=15):
        """遺傳演算法搜尋最佳分配。"""
        population = [np.random.multinomial(self.total_rbgs - 3, [1/3.]*3) + 1 for _ in range(pop_size)]
        
        for _ in range(generations):
            fitness = self._evaluate_allocations(state, population)
            if fitness.sum() == 0: break
            selection_probs = fitness / fitness.sum()
            
            new_population = []
            for _ in range(pop_size):
                parents_idx = np.random.choice(len(population), 2, p=selection_probs, replace=False)
                p1, p2 = population[parents_idx[0]], population[parents_idx[1]]
                child = np.concatenate((p1[:1], p2[1:])).tolist()
                # 修復
                child_sum = sum(child)
                child = [int(c * self.total_rbgs / child_sum) for c in child]
                child[0] = self.total_rbgs - sum(child[1:])
                child = [max(1,c) for c in child]
                
                new_population.append(child)
            population = new_population

        final_fitness = self._evaluate_allocations(state, population)
        best_idx = np.argmax(final_fitness)
        return population[best_idx], final_fitness[best_idx]

    def simulate(self, steps=50, method="genetic"):
        """執行模擬以評估最佳化演算法。"""
        print(f"\n🎯 開始 {method} 模擬...")
        optimizer = self.optimize_genetic if method == "genetic" else self.optimize_exhaustive
        results = []
        for t in range(steps):
            # 隨機產生網路狀態
            state = {
                'num_ues': np.random.randint(4, 25), 'sched_policy_num': np.random.randint(3),
                'bs_id': 1, 'exp_id': 1, 'sum_requested_prbs': np.random.randint(5, 20),
                'hour': np.random.randint(24), 'minute': np.random.randint(60), 'day_of_week': np.random.randint(7),
                'current_rbg_allocation': [5, 7, 5]
            }
            # 簡化特徵
            state['qos_score'] = np.random.rand()
            state['network_load'] = state['num_ues'] / 42
            state['throughput_efficiency'] = np.random.rand()
            
            base_efficiency = self._evaluate_allocations(state, [state['current_rbg_allocation']])[0]
            best_alloc, opt_efficiency = optimizer(state)
            
            results.append({
                'step': t, 'improvement': opt_efficiency - base_efficiency,
                'ues': state['num_ues'], 'slice': t % 3,
                'best_alloc': best_alloc
            })
        return pd.DataFrame(results)

