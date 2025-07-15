# experiments/run_experiment.py

import sys
import os
import json
import pandas as pd
from datetime import datetime

# 將 src 目錄添加到 Python 路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import ColoRANDataLoader
from src.data.data_processor import MemoryOptimizedProcessor
from src.models.ml_trainer import A100OptimizedTrainer
from src.optimization.allocator import SliceResourceAllocator
from src.visualization.plotting import create_comprehensive_visualization

class ExperimentManager:
    """
    實驗管理器，負責協調整個端到端的流程。
    1. 載入資料
    2. 處理特徵
    3. 訓練模型
    4. 執行最佳化模擬
    5. 產生視覺化報告
    6. 儲存所有產出
    """
    def __init__(self, output_dir='./experiment_results'):
        self.output_dir = os.path.join(output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 實驗結果將儲存於: {self.output_dir}")
        self.slice_configs = self._get_slice_configs()
        self.results = {}

    def _get_slice_configs(self):
        """提供 ColO-RAN 資料集的切片配置。"""
        return {f'tr{i}': v for i, v in enumerate([
            [2, 13, 2], [4, 11, 2], [6, 9, 2], [8, 7, 2], [10, 5, 2], [12, 3, 2], [14, 1, 2],
            [2, 11, 4], [4, 9, 4], [6, 7, 4], [8, 5, 4], [10, 3, 4], [12, 1, 4], [2, 9, 6],
            [4, 7, 6], [6, 5, 6], [8, 3, 6], [10, 1, 6], [2, 7, 8], [4, 5, 8], [6, 3, 8],
            [8, 1, 8], [2, 5, 10], [4, 3, 10], [6, 1, 10], [2, 3, 12], [4, 1, 12], [2, 1, 14]
        ])}

    def run_full_pipeline(self):
        """執行完整的實驗流程。"""
        # 1. 資料載入
        print("--- 步驟 1: 載入資料 ---")
        loader = ColoRANDataLoader()
        loader.download_dataset()
        raw_data = loader.load_raw_data()
        if raw_data is None:
            print("❌ 資料載入失敗，終止流程。")
            return

        # 2. 特徵工程
        print("\n--- 步驟 2: 特徵工程與處理 ---")
        processor = MemoryOptimizedProcessor(self.slice_configs)
        processed_data = processor.process_data(raw_data)
        
        # 3. 模型訓練
        print("\n--- 步驟 3: 模型訓練 ---")
        trainer = A100OptimizedTrainer(use_gpu=True)
        features = processor.feature_columns
        target = 'allocation_efficiency'
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, features, target)
        
        trainer.train_random_forest(X_train, y_train)
        history = trainer.train_neural_network(X_train, y_train, X_test, y_test)
        
        self.results['model_performance'] = trainer.evaluate_models(X_test, y_test)
        trainer.save_models(path=self.output_dir)

        # 4. 最佳化模擬
        print("\n--- 步驟 4: 最佳化模擬 ---")
        model_path = os.path.join(self.output_dir, 'rf_model.pkl')
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        allocator = SliceResourceAllocator(model_path, scaler_path, features)
        sim_results = allocator.simulate(steps=100, method="genetic")
        
        # 5. 視覺化
        print("\n--- 步驟 5: 產生視覺化報告 ---")
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': trainer.rf_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        create_comprehensive_visualization(
            feature_importance_df=feature_importance_df,
            history=history,
            sim_results=sim_results,
            output_path=os.path.join(self.output_dir, 'final_visualization.png')
        )
        
        # 6. 儲存最終報告
        self.save_summary()
        print("\n🎉🎉🎉 實驗流程執行完畢！ 🎉🎉🎉")

    def save_summary(self):
        """儲存實驗摘要 JSON 檔案。"""
        summary_path = os.path.join(self.output_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"✅ 實驗摘要已儲存至: {summary_path}")

def main():
    """主執行函數，用於 console script"""
    manager = ExperimentManager()
    manager.run_full_pipeline()

if __name__ == '__main__':
    main()
