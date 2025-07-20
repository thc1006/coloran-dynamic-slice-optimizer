import pytest
import numpy as np
import tensorflow as tf
from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
from src.coloran_optimizer.ml.ml_trainer import MLTrainer
from src.coloran_optimizer.ml.predictors import Predictors

@pytest.fixture
def sample_data():
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.rand(100, 1).astype(np.float32)
    return X, y

def test_ml_trainer_rf_training(sample_data):
    X, y = sample_data
    trainer = MLTrainer()
    trainer.train_random_forest(X, y.flatten()) # cuML RF expects 1D array for y
    assert trainer.rf_model is not None
    # Basic check that it can predict
    predictions = trainer.rf_model.predict(X)
    assert predictions.shape == (100,)

def test_ml_trainer_nn_training(sample_data):
    X, y = sample_data
    trainer = MLTrainer()
    trainer.train_neural_network(X, y)
    assert trainer.nn_model is not None
    # Basic check that it can predict
    predictions = trainer.nn_model.predict(X)
    assert predictions.shape == (100, 1)

def test_predictors_rf_prediction(sample_data):
    X, y = sample_data
    rf_model = cuMLRandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf_model.fit(X, y.flatten())
    predictors = Predictors(rf_model=rf_model, nn_model=None)
    predictions = predictors.predict_with_rf(X)
    assert predictions.shape == (100,)

def test_predictors_nn_prediction(sample_data):
    X, y = sample_data
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X.shape[1],))
    ])
    nn_model.compile(optimizer='adam', loss='mse')
    nn_model.fit(X, y, epochs=1, verbose=0)
    predictors = Predictors(rf_model=None, nn_model=nn_model)
    predictions = predictors.predict_with_nn(X)
    assert predictions.shape == (100, 1)

def test_predictors_ensemble_prediction(sample_data):
    X, y = sample_data
    
    rf_model = cuMLRandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf_model.fit(X, y.flatten())

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X.shape[1],))
    ])
    nn_model.compile(optimizer='adam', loss='mse')
    nn_model.fit(X, y, epochs=1, verbose=0)

    predictors = Predictors(rf_model=rf_model, nn_model=nn_model)
    predictions = predictors.ensemble_predict(X)
    assert predictions.shape == (100,)