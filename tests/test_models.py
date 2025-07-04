# tests/test_models.py
import pytest
import numpy as np
import pandas as pd
from src.models.ml_trainer import A100OptimizedTrainer

@pytest.fixture
def dummy_data():
    features = [f'f{i}' for i in range(10)]
    data = pd.DataFrame(np.random.rand(100, 11), columns=features + ['allocation_efficiency'])
    return data, features, 'allocation_efficiency'

def test_trainer_prepare_data(dummy_data):
    df, features, target = dummy_data
    trainer = A100OptimizedTrainer(use_gpu=False, sample_size=100)
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, features, target)
    
    assert X_train.shape == (80, 10)
    assert X_test.shape == (20, 10)
    assert len(y_train) == 80
    assert trainer.scaler is not None

# Note: Full training tests can be slow. These are basic sanity checks.
def test_trainer_runs_without_gpu(dummy_data):
    df, features, target = dummy_data
    trainer = A100OptimizedTrainer(use_gpu=False, sample_size=100)
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, features, target)

    # Test RF (CPU)
    trainer.train_random_forest(X_train, y_train)
    assert trainer.rf_model is not None
    
    # Test NN (CPU)
    trainer.train_neural_network(X_train, y_train, X_test, y_test)
    assert trainer.nn_model is not None

