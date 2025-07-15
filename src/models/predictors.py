# src/models/predictors.py
import joblib
import numpy as np
import tensorflow as tf
import os

class SliceEfficiencyPredictor:
    """
    使用已訓練的模型預測網路切片效率。
    """
    def __init__(self, model_dir='.'):
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.models = {}
        # 嘗試載入隨機森林模型
        try:
            self.models['rf'] = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
        except FileNotFoundError:
            print("警告: 未找到隨機森林模型 (rf_model.pkl)")
        # 嘗試載入神經網路模型
        try:
            self.models['nn'] = tf.keras.models.load_model(os.path.join(model_dir, 'nn_model.h5'))
        except (IOError, ImportError):
            print("警告: 未找到或無法載入神經網路模型 (nn_model.h5)")
        print(f"✅ 預測器初始化成功，已載入模型: {list(self.models.keys())}")

    def predict(self, feature_matrix, model_type='rf'):
        """
        對輸入的特徵矩陣進行預測。
        """
        if model_type not in self.models:
            raise ValueError(f"模型 '{model_type}' 未被載入。可用的模型: {list(self.models.keys())}")
        
        X_scaled = self.scaler.transform(feature_matrix.astype(np.float32))
        
        model = self.models[model_type]
        if model_type == 'nn':
            predictions = model.predict(X_scaled).flatten()
        else: # 預設使用隨機森林或任何其他 sklearn-like 模型
            predictions = model.predict(X_scaled)
            
        return np.clip(predictions, 0.0, 1.0)
