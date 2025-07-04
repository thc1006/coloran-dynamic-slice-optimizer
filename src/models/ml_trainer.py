# src/models/ml_trainer.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from cuml.ensemble import RandomForestRegressor as cuRF
import joblib
import warnings

warnings.filterwarnings('ignore')

class A100OptimizedTrainer:
    """
    在 NVIDIA A100 GPU 上進行最佳化訓練的訓練器。
    - 支援 cuML 加速的隨機森林。
    - 支援 TensorFlow 混合精度訓練的神經網路。
    - 自動處理資料分割、標準化和模型評估。
    """
    def __init__(self, use_gpu=True, sample_size=5_000_000):
        self.use_gpu = use_gpu
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        self.rf_model = None
        self.nn_model = None
        self.feature_names = None
        
        # 偵測 cuML 可用性
        try:
            import cuml
            self.cuml_available = True
            print("✅ cuML 已找到，將使用 GPU 進行隨機森林訓練。")
        except ImportError:
            self.cuml_available = False
            print("⚠️ cuML 未找到，隨機森林將使用 CPU。")

    def prepare_data(self, df, features, target):
        """準備訓練資料，包括採樣、分割和標準化。"""
        self.feature_names = features
        print("📊 準備訓練資料...")
        
        if len(df) > self.sample_size:
            print(f"大型資料集，採樣 {self.sample_size:,} 筆記錄進行訓練...")
            df = df.sample(n=self.sample_size, random_state=42)
            
        X = df[features].astype(np.float32)
        y = df[target].astype(np.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ 資料準備完成。訓練集: {X_train.shape}, 測試集: {X_test.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_random_forest(self, X_train, y_train):
        """訓練隨機森林模型，優先使用 GPU。"""
        print("\n🌲 訓練隨機森林...")
        if self.use_gpu and self.cuml_available:
            self.rf_model = cuRF(n_estimators=300, max_depth=16, random_state=42, max_features=0.5)
        else:
            self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, verbose=1)
        
        self.rf_model.fit(X_train, y_train)
        print("✅ 隨機森林訓練完成。")

    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """訓練神經網路模型，使用 TensorFlow。"""
        print("\n🧠 訓練神經網路...")
        if self.use_gpu:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("已啟用混合精度訓練。")

        self.nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, dtype='float32') # 輸出層使用 float32
        ])
        
        self.nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.nn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=4096,
            callbacks=callbacks,
            verbose=1
        )
        print("✅ 神經網路訓練完成。")
        return history

    def evaluate_models(self, X_test, y_test):
        """評估已訓練模型的效能。"""
        results = {}
        if self.rf_model:
            preds = self.rf_model.predict(X_test)
            results['Random Forest'] = {'R2': r2_score(y_test, preds), 'MAE': mean_absolute_error(y_test, preds)}
        
        if self.nn_model:
            preds = self.nn_model.predict(X_test).flatten()
            results['Neural Network'] = {'R2': r2_score(y_test, preds), 'MAE': mean_absolute_error(y_test, preds)}
        
        print("\n📈 模型評估結果:")
        for model, metrics in results.items():
            print(f"  - {model}: R²={metrics['R2']:.6f}, MAE={metrics['MAE']:.6f}")
        return results

    def save_models(self, path='.'):
        """儲存模型和標準化器。"""
        if self.rf_model:
            joblib.dump(self.rf_model, f'{path}/rf_model.pkl')
        if self.nn_model:
            self.nn_model.save(f'{path}/nn_model.h5')
        if self.scaler:
            joblib.dump(self.scaler, f'{path}/scaler.pkl')
        print(f"✅ 模型已儲存至 '{path}' 目錄。")

