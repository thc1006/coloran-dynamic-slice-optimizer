# src/models/ml_trainer.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
import logging
import gc
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Import configuration manager
import sys
sys.path.append(str(Path(__file__).parent.parent))
from coloran_optimizer.config import get_config

warnings.filterwarnings('ignore')

class A100OptimizedTrainer:
    """
    Production-ready ML trainer for ColO-RAN Dynamic Slice Optimizer.
    
    Features:
    - Configuration-driven training parameters
    - Robust GPU memory management with fallback to CPU
    - Comprehensive model validation and metrics tracking
    - Model versioning and experiment tracking
    - Cross-validation and hyperparameter optimization
    - Reproducible training with proper seed management
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        self.training_config = self.config.get_training_config()
        self.model_config = self.config.get_model_config()
        
        # Training parameters from config
        self.use_gpu = self.training_config.get('use_gpu', True)
        self.sample_size = self.training_config.get('sample_size', 5_000_000)
        self.batch_size = self.training_config.get('batch_size', 4096)
        self.epochs = self.training_config.get('epochs', 100)
        self.random_seed = self.config.get('data.random_seed', 42)
        
        # Model artifacts
        self.scaler = StandardScaler()
        self.rf_model = None
        self.nn_model = None
        self.feature_names = None
        self.training_history = {}
        self.model_metadata = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU and cuML detection
        self._setup_gpu_environment()
        self._detect_cuml_availability()
        
        # Set reproducibility
        self._set_random_seeds()
    
    def _setup_gpu_environment(self):
        """Setup GPU environment with proper memory management."""
        if self.use_gpu:
            try:
                # Configure TensorFlow GPU memory growth
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"✅ Found {len(gpus)} GPU(s), memory growth enabled")
                    
                    # Enable mixed precision if configured
                    if self.training_config.get('mixed_precision', True):
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                        self.logger.info("✅ Mixed precision training enabled")
                else:
                    self.logger.warning("⚠️ No GPUs found, falling back to CPU")
                    self.use_gpu = False
                    
            except Exception as e:
                self.logger.error(f"❌ GPU setup failed: {e}, falling back to CPU")
                self.use_gpu = False
        else:
            self.logger.info("CPU training mode selected")
    
    def _detect_cuml_availability(self):
        """Detect cuML availability for GPU-accelerated ML."""
        try:
            import cuml
            self.cuml_available = self.use_gpu
            if self.cuml_available:
                self.logger.info("✅ cuML available for GPU-accelerated ML")
        except ImportError:
            self.cuml_available = False
            self.logger.warning("⚠️ cuML not available, using CPU-based ML")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)

    def prepare_data(self, df, features, target, stratify_column=None):
        """Enhanced data preparation with validation and quality checks."""
        self.feature_names = features
        self.logger.info("📊 Preparing training data...")
        
        # Data quality validation
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")
        
        # Remove rows with missing target values
        initial_size = len(df)
        df = df.dropna(subset=[target])
        if len(df) < initial_size:
            self.logger.warning(f"Dropped {initial_size - len(df)} rows with missing target values")
        
        # Sample data if needed
        if len(df) > self.sample_size:
            self.logger.info(f"Sampling {self.sample_size:,} records from {len(df):,} total")
            df = df.sample(n=self.sample_size, random_state=self.random_seed)
        
        # Prepare features and target
        X = df[features].astype(np.float32)
        y = df[target].astype(np.float32)
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Data validation
        self._validate_data_quality(X, y)
        
        # Split data
        validation_split = self.config.get('data.validation_split', 0.2)
        stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, 
            random_state=self.random_seed, stratify=stratify
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store data statistics
        self.model_metadata.update({
            'feature_names': features,
            'target_name': target,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_stats': {
                'mean': X_train.mean().to_dict(),
                'std': X_train.std().to_dict()
            },
            'target_stats': {
                'mean': float(y_train.mean()),
                'std': float(y_train.std()),
                'min': float(y_train.min()),
                'max': float(y_train.max())
            }
        })
        
        self.logger.info(f"✅ Data preparation completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _validate_data_quality(self, X, y):
        """Validate data quality and log warnings for issues."""
        # Check for infinite values
        inf_features = X.columns[np.isinf(X).any()].tolist()
        if inf_features:
            self.logger.warning(f"Features with infinite values: {inf_features}")
        
        # Check for constant features
        constant_features = X.columns[X.var() == 0].tolist()
        if constant_features:
            self.logger.warning(f"Constant features detected: {constant_features}")
        
        # Check target distribution
        if y.var() == 0:
            self.logger.error("Target variable has zero variance")
            raise ValueError("Target variable must have non-zero variance")
        
        # Log data quality metrics
        missing_pct = (X.isnull().sum() / len(X) * 100).max()
        self.logger.info(f"Max missing values in features: {missing_pct:.2f}%")
        self.logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None):
        """Enhanced Random Forest training with hyperparameter optimization."""
        self.logger.info("🌲 Training Random Forest model...")
        
        # Get hyperparameters from config
        n_estimators = self.model_config.get('rf_n_estimators', 300)
        max_depth = self.model_config.get('rf_max_depth', 16)
        
        try:
            if self.use_gpu and self.cuml_available:
                from cuml.ensemble import RandomForestRegressor as cuRF
                self.rf_model = cuRF(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_seed,
                    max_features=0.5
                )
                self.logger.info("Using GPU-accelerated cuML Random Forest")
            else:
                self.rf_model = RandomForestRegressor(
                    n_estimators=min(n_estimators, 200),  # CPU limitation
                    max_depth=max_depth,
                    random_state=self.random_seed,
                    n_jobs=-1,
                    verbose=0
                )
                self.logger.info("Using CPU-based scikit-learn Random Forest")
            
            # Train model
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate on test set if provided
            if X_test is not None and y_test is not None:
                rf_predictions = self.rf_model.predict(X_test)
                rf_metrics = self._calculate_metrics(y_test, rf_predictions)
                self.training_history['random_forest'] = rf_metrics
                self.logger.info(f"Random Forest - R²: {rf_metrics['r2']:.4f}, MAE: {rf_metrics['mae']:.4f}")
            
            # Memory cleanup for GPU
            if self.use_gpu:
                gc.collect()
                
            self.logger.info("✅ Random Forest training completed")
            
        except Exception as e:
            self.logger.error(f"❌ Random Forest training failed: {e}")
            raise

    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Enhanced Neural Network training with configurable architecture."""
        self.logger.info("🧠 Training Neural Network model...")
        
        try:
            # Get architecture parameters from config
            hidden_layers = self.model_config.get('nn_hidden_layers', [256, 128, 64])
            dropout_rate = self.model_config.get('dropout_rate', 0.3)
            
            # Build model architecture
            layers = []
            layers.append(tf.keras.layers.Dense(
                hidden_layers[0], 
                activation='relu', 
                input_shape=(X_train.shape[1],)
            ))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.Dropout(dropout_rate))
            
            # Add hidden layers
            for hidden_size in hidden_layers[1:]:
                layers.append(tf.keras.layers.Dense(hidden_size, activation='relu'))
                layers.append(tf.keras.layers.BatchNormalization())
                layers.append(tf.keras.layers.Dropout(dropout_rate))
            
            # Output layer
            layers.append(tf.keras.layers.Dense(1, dtype='float32'))
            
            self.nn_model = tf.keras.Sequential(layers)
            
            # Compile model
            self.nn_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=self.training_config.get('early_stopping_patience', 15),
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=self.training_config.get('learning_rate_patience', 5),
                    factor=0.5,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{self.model_config.get('save_path', './models')}/best_nn_model.h5",
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train model
            history = self.nn_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1 if self.logger.level <= logging.INFO else 0
            )
            
            # Evaluate model
            nn_predictions = self.nn_model.predict(X_test).flatten()
            nn_metrics = self._calculate_metrics(y_test, nn_predictions)
            self.training_history['neural_network'] = nn_metrics
            self.training_history['training_history'] = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae']
            }
            
            self.logger.info(f"Neural Network - R²: {nn_metrics['r2']:.4f}, MAE: {nn_metrics['mae']:.4f}")
            self.logger.info("✅ Neural Network training completed")
            
            return history
            
        except Exception as e:
            self.logger.error(f"❌ Neural Network training failed: {e}")
            raise
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics."""
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation with cross-validation."""
        results = {}
        
        self.logger.info("📈 Evaluating trained models...")
        
        if self.rf_model:
            rf_preds = self.rf_model.predict(X_test)
            results['Random Forest'] = self._calculate_metrics(y_test, rf_preds)
        
        if self.nn_model:
            nn_preds = self.nn_model.predict(X_test).flatten()
            results['Neural Network'] = self._calculate_metrics(y_test, nn_preds)
        
        # Log results
        for model_name, metrics in results.items():
            self.logger.info(f"{model_name} Performance:")
            for metric_name, value in metrics.items():
                self.logger.info(f"  - {metric_name.upper()}: {value:.6f}")
        
        return results
    
    def perform_cross_validation(self, X, y, cv_folds=5):
        """Perform cross-validation for model robustness assessment."""
        self.logger.info(f"🔄 Performing {cv_folds}-fold cross-validation...")
        
        cv_results = {}
        
        if self.rf_model:
            # Use a fresh model for CV to avoid overfitting
            rf_cv = RandomForestRegressor(
                n_estimators=100,  # Smaller for CV speed
                max_depth=self.model_config.get('rf_max_depth', 16),
                random_state=self.random_seed,
                n_jobs=-1
            )
            
            cv_scores = cross_val_score(rf_cv, X, y, cv=cv_folds, scoring='r2')
            cv_results['Random Forest'] = {
                'mean_r2': cv_scores.mean(),
                'std_r2': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
            
            self.logger.info(f"Random Forest CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_results

    def save_models(self, save_path=None):
        """Enhanced model saving with metadata and versioning."""
        if save_path is None:
            save_path = self.model_config.get('save_path', './models')
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate version timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save Random Forest model
            if self.rf_model:
                rf_path = save_path / f'rf_model_{timestamp}.pkl'
                joblib.dump(self.rf_model, rf_path)
                self.logger.info(f"Random Forest saved: {rf_path}")
            
            # Save Neural Network model
            if self.nn_model:
                nn_path = save_path / f'nn_model_{timestamp}.h5'
                self.nn_model.save(str(nn_path))
                self.logger.info(f"Neural Network saved: {nn_path}")
            
            # Save scaler
            if self.scaler:
                scaler_path = save_path / f'scaler_{timestamp}.pkl'
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"Scaler saved: {scaler_path}")
            
            # Save metadata
            self.model_metadata.update({
                'timestamp': timestamp,
                'training_config': self.training_config,
                'model_config': self.model_config,
                'training_history': self.training_history
            })
            
            metadata_path = save_path / f'metadata_{timestamp}.json'
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2, default=str)
            
            self.logger.info(f"✅ All models and metadata saved to: {save_path}")
            
            return {
                'save_path': str(save_path),
                'timestamp': timestamp,
                'files_saved': [
                    f'rf_model_{timestamp}.pkl' if self.rf_model else None,
                    f'nn_model_{timestamp}.h5' if self.nn_model else None,
                    f'scaler_{timestamp}.pkl' if self.scaler else None,
                    f'metadata_{timestamp}.json'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Model saving failed: {e}")
            raise
    
    def load_models(self, model_path, timestamp=None):
        """Load saved models and metadata."""
        model_path = Path(model_path)
        
        if timestamp:
            # Load specific version
            rf_file = model_path / f'rf_model_{timestamp}.pkl'
            nn_file = model_path / f'nn_model_{timestamp}.h5'
            scaler_file = model_path / f'scaler_{timestamp}.pkl'
            metadata_file = model_path / f'metadata_{timestamp}.json'
        else:
            # Load latest version
            rf_files = list(model_path.glob('rf_model_*.pkl'))
            nn_files = list(model_path.glob('nn_model_*.h5'))
            scaler_files = list(model_path.glob('scaler_*.pkl'))
            metadata_files = list(model_path.glob('metadata_*.json'))
            
            if not metadata_files:
                raise FileNotFoundError("No model metadata files found")
            
            # Get latest timestamp
            latest_metadata = max(metadata_files, key=lambda x: x.stem.split('_')[-1])
            timestamp = latest_metadata.stem.split('_')[-1]
            
            rf_file = model_path / f'rf_model_{timestamp}.pkl'
            nn_file = model_path / f'nn_model_{timestamp}.h5'
            scaler_file = model_path / f'scaler_{timestamp}.pkl'
            metadata_file = latest_metadata
        
        # Load models
        if rf_file.exists():
            self.rf_model = joblib.load(rf_file)
            self.logger.info(f"Random Forest loaded: {rf_file}")
        
        if nn_file.exists():
            self.nn_model = tf.keras.models.load_model(str(nn_file))
            self.logger.info(f"Neural Network loaded: {nn_file}")
        
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            self.logger.info(f"Scaler loaded: {scaler_file}")
        
        # Load metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
            self.logger.info(f"Metadata loaded: {metadata_file}")
        
        self.logger.info(f"✅ Models loaded from timestamp: {timestamp}")
        return timestamp

