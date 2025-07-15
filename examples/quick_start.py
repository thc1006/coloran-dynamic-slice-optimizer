# examples/quick_start.py

import os
import sys
import pandas as pd

# 將 src 目錄添加到 Python 路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.allocator import SliceResourceAllocator
from src.visualization.plotting import plot_efficiency_improvement

def quick_start_demo(model_dir):
    """
    執行快速展示，載入預訓練模型並進行簡短模擬。
    """
    print("🚀 5G Slice Optimizer - 快速開始展示 🚀")

    if not os.path.exists(model_dir):
        print(f"❌ 錯誤: 找不到模型目錄 '{model_dir}'")
        print("👉 請先執行 'python experiments/run_experiment.py' 來產生模型檔。")
        return

    # 1. 使用預訓練模型初始化分配器
    print(f"\n[1/2] 從 '{model_dir}' 載入模型...")
    try:
        # 假設特徵列表為已知或可從某處載入
        features = [
            'num_ues', 'slice_id', 'sched_policy_num', 'allocated_rbgs', 'bs_id', 
            'exp_id', 'sum_requested_prbs', 'sum_granted_prbs', 'prb_utilization', 
            'throughput_efficiency', 'qos_score', 'network_load', 'hour', 
            'minute', 'day_of_week'
        ]
        allocator = SliceResourceAllocator(
            model_path=os.path.join(model_dir, 'rf_model.pkl'),
            scaler_path=os.path.join(model_dir, 'scaler.pkl'),
            features=features
        )
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return

    # 2. 執行一個簡短的模擬
    print("\n[2/2] 執行資源分配模擬...")
    sim_results = allocator.simulate(steps=20, method="genetic")
    print("\n模擬結果:")
    print(sim_results.head())
    print(f"\n平均效率改善: {sim_results['improvement'].mean():.4f}")

    print("\n✅ 展示成功完成！")

if __name__ == '__main__':
    # 注意：您需要將此路徑替換為您自己執行 run_experiment.py 後產生的實際結果目錄
    # 例如：'experiment_results/20250704_102400'
    latest_experiment_dir = 'experiment_results/<YOUR_TIMESTAMPED_FOLDER>'
    quick_start_demo(latest_experiment_dir)
