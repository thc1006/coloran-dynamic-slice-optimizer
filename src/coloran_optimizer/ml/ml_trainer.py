import tensorflow as tf
import cuml
from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

class MLTrainer:
    def __init__(self):
        self.rf_model = None
        self.nn_model = None

    def train_random_forest(self, X, y):
        print("Training cuML Random Forest...")
        self.rf_model = cuMLRandomForestRegressor(n_estimators=300, max_depth=16, random_state=42)
        self.rf_model.fit(X, y)
        print("cuML Random Forest training complete.")

    def train_neural_network(self, X, y):
        print("Training TensorFlow Neural Network...")
        self.nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.nn_model.compile(optimizer='adam', loss='mse')
        
        # Enable mixed precision training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Convert to NumPy for TensorFlow if X is a cuDF DataFrame
        if isinstance(X, cuml.DataFrame):
            X_np = X.to_numpy()
            y_np = y.to_numpy()
        else:
            X_np = X
            y_np = y

        X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
        
        self.nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
        print("TensorFlow Neural Network training complete.")

    def evaluate_model(self, model, X, y):
        if isinstance(X, cuml.DataFrame):
            X_np = X.to_numpy()
            y_np = y.to_numpy()
        else:
            X_np = X
            y_np = y

        predictions = model.predict(X_np)
        if isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()
        return r2_score(y_np, predictions)
