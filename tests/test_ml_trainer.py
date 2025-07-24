# tests/test_ml_trainer.py

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.ml_trainer import A100OptimizedTrainer
from coloran_optimizer.config import ConfigurationManager


class TestA100OptimizedTrainer:
    """Test suite for A100OptimizedTrainer."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock()
        config.get_training_config.return_value = {
            'use_gpu': False,  # Use CPU for testing
            'sample_size': 1000,
            'batch_size': 32,
            'epochs': 2,
            'early_stopping_patience': 2,
            'learning_rate_patience': 1,
            'mixed_precision': False
        }
        config.get_model_config.return_value = {
            'save_path': './test_models',
            'rf_n_estimators': 10,  # Small for testing
            'rf_max_depth': 5,
            'nn_hidden_layers': [32, 16],
            'dropout_rate': 0.2
        }
        config.get.return_value = 42  # random_seed
        return config
    
    @pytest.fixture
    def trainer(self, mock_config):
        """Create trainer instance for testing."""
        return A100OptimizedTrainer(config_manager=mock_config)
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 500
        
        # Create realistic features
        data = pd.DataFrame({
            'slice_id': np.random.randint(0, 3, n_samples),
            'sched_policy_num': np.random.randint(0, 3, n_samples),
            'allocated_rbgs': np.random.randint(1, 10, n_samples),
            'sum_requested_prbs': np.random.randint(1, 20, n_samples),
            'sum_granted_prbs': np.random.randint(1, 20, n_samples),
            'prb_utilization': np.random.uniform(0, 1, n_samples),
            'throughput_efficiency': np.random.uniform(0, 1, n_samples),
            'qos_score': np.random.uniform(0, 1, n_samples),
            'network_load': np.random.uniform(0, 1, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'minute': np.random.randint(0, 60, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'allocation_efficiency': np.random.uniform(0, 1, n_samples)
        })
        
        return data
    
    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.use_gpu == False  # Set to False in mock config
        assert trainer.sample_size == 1000
        assert trainer.batch_size == 32
        assert trainer.epochs == 2
        assert trainer.random_seed == 42
        assert trainer.scaler is not None
        assert trainer.training_history == {}
        assert trainer.model_metadata == {}
    
    def test_random_seed_setting(self, trainer):
        """Test that random seeds are set properly."""
        # This is called in __init__, just verify it doesn't raise exceptions
        trainer._set_random_seeds()
        
        # Check that numpy random state is set
        first_random = np.random.random()
        trainer._set_random_seeds()
        second_random = np.random.random()
        
        assert first_random == second_random  # Should be reproducible
    
    def test_gpu_environment_setup(self, trainer):
        """Test GPU environment setup."""
        # Since we're using CPU mode, this should not raise exceptions
        trainer._setup_gpu_environment()
        
        # Verify GPU setup doesn't break CPU mode
        assert trainer.use_gpu == False
    
    def test_cuml_detection(self, trainer):
        """Test cuML availability detection."""
        trainer._detect_cuml_availability()
        
        # Should not raise exceptions regardless of cuML availability
        assert hasattr(trainer, 'cuml_available')
    
    def test_prepare_data(self, trainer, sample_training_data):
        """Test data preparation."""
        features = [
            'slice_id', 'sched_policy_num', 'allocated_rbgs',
            'sum_requested_prbs', 'sum_granted_prbs', 'prb_utilization',
            'throughput_efficiency', 'qos_score', 'network_load',
            'hour', 'minute', 'day_of_week'
        ]
        target = 'allocation_efficiency'
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            sample_training_data, features, target
        )
        
        # Check output shapes
        assert X_train.shape[1] == len(features)
        assert X_test.shape[1] == len(features)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check that data is scaled
        assert X_train.mean() < 0.1  # Should be approximately zero-centered
        
        # Check metadata is stored
        assert trainer.feature_names == features
        assert 'feature_names' in trainer.model_metadata
        assert 'target_name' in trainer.model_metadata
        assert 'train_size' in trainer.model_metadata
        assert 'test_size' in trainer.model_metadata
    
    def test_prepare_data_missing_features(self, trainer, sample_training_data):
        """Test data preparation with missing features."""
        features = ['nonexistent_feature']
        target = 'allocation_efficiency'
        
        with pytest.raises(ValueError, match="Missing features in dataset"):
            trainer.prepare_data(sample_training_data, features, target)
    
    def test_prepare_data_missing_target(self, trainer, sample_training_data):
        """Test data preparation with missing target."""
        features = ['slice_id']
        target = 'nonexistent_target'
        
        with pytest.raises(ValueError, match="Target column .* not found"):
            trainer.prepare_data(sample_training_data, features, target)
    
    def test_data_quality_validation(self, trainer, sample_training_data):
        """Test data quality validation."""
        features = ['slice_id', 'allocated_rbgs']
        X = sample_training_data[features]
        y = sample_training_data['allocation_efficiency']
        
        # Should not raise exceptions for good data
        trainer._validate_data_quality(X, y)
        
        # Test with constant target (should raise exception)
        y_constant = pd.Series([0.5] * len(y))
        with pytest.raises(ValueError, match="Target variable must have non-zero variance"):
            trainer._validate_data_quality(X, y_constant)
    
    def test_train_random_forest(self, trainer, sample_training_data):
        """Test Random Forest training."""
        features = [
            'slice_id', 'sched_policy_num', 'allocated_rbgs',
            'sum_requested_prbs', 'sum_granted_prbs', 'prb_utilization'
        ]
        target = 'allocation_efficiency'
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            sample_training_data, features, target
        )
        
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Check that model was trained
        assert trainer.rf_model is not None
        assert isinstance(trainer.rf_model, RandomForestRegressor)
        
        # Check that metrics were recorded
        assert 'random_forest' in trainer.training_history
        rf_metrics = trainer.training_history['random_forest']
        assert 'r2' in rf_metrics
        assert 'mae' in rf_metrics
        assert 'mse' in rf_metrics
        assert 'rmse' in rf_metrics
    
    def test_train_neural_network(self, trainer, sample_training_data):
        """Test Neural Network training."""
        features = [
            'slice_id', 'sched_policy_num', 'allocated_rbgs',
            'prb_utilization', 'qos_score', 'network_load'
        ]
        target = 'allocation_efficiency'
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            sample_training_data, features, target
        )
        
        history = trainer.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Check that model was trained
        assert trainer.nn_model is not None
        assert isinstance(trainer.nn_model, tf.keras.Sequential)
        
        # Check that metrics were recorded
        assert 'neural_network' in trainer.training_history
        nn_metrics = trainer.training_history['neural_network']
        assert 'r2' in nn_metrics
        assert 'mae' in nn_metrics
        
        # Check training history
        assert 'training_history' in trainer.training_history
        training_hist = trainer.training_history['training_history']
        assert 'loss' in training_hist
        assert 'val_loss' in training_hist
        
        # Check that history object is returned
        assert history is not None
        assert hasattr(history, 'history')
    
    def test_calculate_metrics(self, trainer):
        """Test metrics calculation."""
        y_true = np.array([0.8, 0.6, 0.7, 0.9, 0.5])
        y_pred = np.array([0.75, 0.65, 0.72, 0.85, 0.55])
        
        metrics = trainer._calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'r2' in metrics
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['r2'] <= 1  # R² for good predictions
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
    
    def test_evaluate_models(self, trainer, sample_training_data):
        """Test model evaluation."""
        features = ['slice_id', 'allocated_rbgs', 'prb_utilization']
        target = 'allocation_efficiency'
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            sample_training_data, features, target
        )
        
        # Train both models
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
        trainer.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Evaluate
        results = trainer.evaluate_models(X_test, y_test)
        
        assert isinstance(results, dict)
        assert 'Random Forest' in results
        assert 'Neural Network' in results
        
        for model_results in results.values():
            assert 'r2' in model_results
            assert 'mae' in model_results
            assert 'mse' in model_results
            assert 'rmse' in model_results
    
    def test_cross_validation(self, trainer, sample_training_data):
        """Test cross-validation."""
        features = ['slice_id', 'allocated_rbgs', 'prb_utilization']
        target = 'allocation_efficiency'
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            sample_training_data, features, target
        )
        
        # Train RF model first
        trainer.train_random_forest(X_train, y_train)
        
        # Perform cross-validation
        cv_results = trainer.perform_cross_validation(X_train, y_train, cv_folds=3)
        
        assert isinstance(cv_results, dict)
        assert 'Random Forest' in cv_results
        
        rf_cv = cv_results['Random Forest']
        assert 'mean_r2' in rf_cv
        assert 'std_r2' in rf_cv
        assert 'scores' in rf_cv
        assert len(rf_cv['scores']) == 3  # 3-fold CV
    
    def test_save_and_load_models(self, trainer, sample_training_data, tmp_path):
        """Test model saving and loading."""
        features = ['slice_id', 'allocated_rbgs']
        target = 'allocation_efficiency'
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            sample_training_data, features, target
        )
        
        # Train models
        trainer.train_random_forest(X_train, y_train)
        trainer.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Save models
        save_info = trainer.save_models(str(tmp_path))
        
        assert isinstance(save_info, dict)
        assert 'save_path' in save_info
        assert 'timestamp' in save_info
        assert 'files_saved' in save_info
        
        # Check files were created
        timestamp = save_info['timestamp']
        assert (tmp_path / f'rf_model_{timestamp}.pkl').exists()
        assert (tmp_path / f'nn_model_{timestamp}.h5').exists()
        assert (tmp_path / f'scaler_{timestamp}.pkl').exists()
        assert (tmp_path / f'metadata_{timestamp}.json').exists()
        
        # Create new trainer and load models
        new_trainer = A100OptimizedTrainer(config_manager=trainer.config)
        loaded_timestamp = new_trainer.load_models(str(tmp_path), timestamp)
        
        assert loaded_timestamp == timestamp
        assert new_trainer.rf_model is not None
        assert new_trainer.nn_model is not None
        assert new_trainer.scaler is not None
        assert new_trainer.model_metadata is not None
    
    def test_model_reproducibility(self, mock_config):
        """Test that training is reproducible with same random seed."""
        sample_data = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'target': np.random.random(100)
        })
        
        # Train first model
        trainer1 = A100OptimizedTrainer(config_manager=mock_config)
        X_train1, X_test1, y_train1, y_test1 = trainer1.prepare_data(
            sample_data, ['feature1', 'feature2'], 'target'
        )
        trainer1.train_random_forest(X_train1, y_train1)
        pred1 = trainer1.rf_model.predict(X_test1)
        
        # Train second model with same seed
        trainer2 = A100OptimizedTrainer(config_manager=mock_config)
        X_train2, X_test2, y_train2, y_test2 = trainer2.prepare_data(
            sample_data, ['feature1', 'feature2'], 'target'
        )
        trainer2.train_random_forest(X_train2, y_train2)
        pred2 = trainer2.rf_model.predict(X_test2)
        
        # Results should be identical due to same random seed
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
    
    def test_error_handling(self, trainer):
        """Test error handling in various scenarios."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            trainer.prepare_data(empty_df, ['feature1'], 'target')
        
        # Test with insufficient data
        tiny_df = pd.DataFrame({
            'feature1': [1, 2],
            'target': [0.1, 0.2]
        })
        
        # Should handle gracefully
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            tiny_df, ['feature1'], 'target'
        )
        
        # Should have some data even if very small
        assert len(X_train) > 0 or len(X_test) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])