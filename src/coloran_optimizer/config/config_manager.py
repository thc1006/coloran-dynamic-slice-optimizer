# src/coloran_optimizer/config/config_manager.py

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class ConfigurationManager:
    """
    Comprehensive configuration management system for ColO-RAN Dynamic Slice Optimizer.
    Supports environment-specific configurations, security best practices, and validation.
    """
    
    def __init__(self, config_dir: Optional[str] = None, environment: str = "development"):
        self.environment = environment
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self._config = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load configuration files in priority order."""
        # 1. Load base configuration
        base_config_path = self.config_dir / "base.yaml"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        
        # 2. Load environment-specific configuration
        env_config_path = self.config_dir / f"{self.environment}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
                self._deep_merge(self._config, env_config)
        
        # 3. Override with environment variables
        self._load_env_overrides()
        
        # 4. Validate configuration
        self._validate_config()
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mapping = {
            'COLORAN_DATA_PATH': 'data.base_path',
            'COLORAN_GPU_ENABLED': 'training.use_gpu',
            'COLORAN_BATCH_SIZE': 'training.batch_size',
            'COLORAN_LOG_LEVEL': 'logging.level',
            'COLORAN_MODEL_PATH': 'model.save_path',
            'COLORAN_JWT_SECRET': 'security.jwt_secret',
            'COLORAN_DB_URL': 'database.url',
        }
        
        for env_var, config_path in env_mapping.items():
            if env_var in os.environ:
                self._set_nested_value(config_path, os.environ[env_var])
    
    def _set_nested_value(self, path: str, value: str):
        """Set nested dictionary value using dot notation."""
        keys = path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Type conversion for common types
        final_key = keys[-1]
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                current[final_key] = value.lower() == 'true'
            elif value.isdigit():
                current[final_key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                current[final_key] = float(value)
            else:
                current[final_key] = value
        else:
            current[final_key] = value
    
    def _validate_config(self):
        """Validate critical configuration values."""
        required_paths = [
            'data.base_path',
            'training.batch_size',
            'model.save_path'
        ]
        
        for path in required_paths:
            value = self.get(path)
            if value is None or (isinstance(value, str) and not value.strip()):
                self.logger.warning(f"Required configuration '{path}' is missing, using default")
                # Set reasonable defaults
                if path == 'data.base_path':
                    self.set(path, './data')
                elif path == 'training.batch_size':
                    self.set(path, 1000)
                elif path == 'model.save_path':
                    self.set(path, './models')
        
        # Validate specific values
        if self.get('training.batch_size', 0) <= 0:
            raise ValueError("training.batch_size must be positive")
        
        # Security validations
        jwt_secret = self.get('security.jwt_secret')
        if jwt_secret and len(jwt_secret) < 32:
            self.logger.warning("JWT secret should be at least 32 characters long")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = path.split('.')
        current = self._config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """Set configuration value using dot notation."""
        self._set_nested_value(path, value)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return self.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.get('security', {})
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        if not path:
            path = self.config_dir / f"{self.environment}_generated.yaml"
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def create_default_configs(self):
        """Create default configuration files."""
        # Base configuration
        base_config = {
            'data': {
                'base_path': '/content',
                'batch_size': 75000,
                'validation_split': 0.2,
                'random_seed': 42,
                'quality_threshold': 70.0
            },
            'training': {
                'use_gpu': True,
                'sample_size': 5000000,
                'batch_size': 4096,
                'epochs': 100,
                'early_stopping_patience': 15,
                'learning_rate_patience': 5,
                'mixed_precision': True
            },
            'model': {
                'save_path': './models',
                'rf_n_estimators': 300,
                'rf_max_depth': 16,
                'nn_hidden_layers': [256, 128, 64],
                'dropout_rate': 0.3
            },
            'optimization': {
                'total_rbgs': 17,
                'timeout_seconds': 600,
                'genetic_population_size': 80,
                'genetic_generations': 15
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_path': './logs/coloran.log'
            },
            'security': {
                'jwt_secret': '${COLORAN_JWT_SECRET}',
                'api_rate_limit': 100,
                'input_validation': True
            }
        }
        
        # Development configuration
        dev_config = {
            'data': {
                'base_path': './data',
                'batch_size': 10000
            },
            'training': {
                'sample_size': 100000,
                'epochs': 10
            },
            'logging': {
                'level': 'DEBUG'
            }
        }
        
        # Production configuration
        prod_config = {
            'data': {
                'base_path': '/opt/coloran/data',
                'batch_size': 100000
            },
            'training': {
                'sample_size': 10000000,
                'epochs': 200
            },
            'logging': {
                'level': 'WARNING',
                'file_path': '/var/log/coloran/coloran.log'
            },
            'security': {
                'api_rate_limit': 1000,
                'input_validation': True
            }
        }
        
        # Save configuration files
        configs = {
            'base.yaml': base_config,
            'development.yaml': dev_config,
            'production.yaml': prod_config
        }
        
        for filename, config in configs.items():
            config_path = self.config_dir / filename
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        print(f"Default configuration files created in {self.config_dir}")

# Global configuration instance
_config_manager = None

def get_config(environment: str = None) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None or (environment and _config_manager.environment != environment):
        env = environment or os.getenv('COLORAN_ENV', 'development')
        _config_manager = ConfigurationManager(environment=env)
    
    return _config_manager

def init_config(config_dir: str = None, environment: str = None, create_defaults: bool = False):
    """Initialize configuration system."""
    global _config_manager
    
    env = environment or os.getenv('COLORAN_ENV', 'development')
    _config_manager = ConfigurationManager(config_dir=config_dir, environment=env)
    
    if create_defaults:
        _config_manager.create_default_configs()
    
    return _config_manager