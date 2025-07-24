# tests/test_config_manager.py

import pytest
import os
import yaml
import json
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coloran_optimizer.config.config_manager import ConfigurationManager, get_config, init_config


class TestConfigurationManager:
    """Test suite for ConfigurationManager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_configs(self, temp_config_dir):
        """Create sample configuration files."""
        # Base configuration
        base_config = {
            'data': {
                'base_path': '/default/path',
                'batch_size': 10000,
                'validation_split': 0.2
            },
            'training': {
                'use_gpu': True,
                'epochs': 100
            },
            'security': {
                'jwt_secret': 'default_secret',
                'api_rate_limit': 100
            }
        }
        
        # Development configuration
        dev_config = {
            'data': {
                'base_path': '/dev/path',
                'batch_size': 5000
            },
            'training': {
                'epochs': 10
            }
        }
        
        # Production configuration
        prod_config = {
            'data': {
                'base_path': '/prod/path'
            },
            'training': {
                'epochs': 200
            },
            'security': {
                'api_rate_limit': 1000
            }
        }
        
        # Save configs
        with open(temp_config_dir / 'base.yaml', 'w') as f:
            yaml.dump(base_config, f)
        
        with open(temp_config_dir / 'development.yaml', 'w') as f:
            yaml.dump(dev_config, f)
        
        with open(temp_config_dir / 'production.yaml', 'w') as f:
            yaml.dump(prod_config, f)
        
        return base_config, dev_config, prod_config
    
    def test_initialization_default(self, temp_config_dir, sample_configs):
        """Test configuration manager initialization with defaults."""
        config_manager = ConfigurationManager(
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        assert config_manager.environment == 'development'
        assert config_manager.config_dir == temp_config_dir
        
        # Check that configs were loaded and merged
        assert config_manager.get('data.base_path') == '/dev/path'  # Overridden in dev
        assert config_manager.get('data.batch_size') == 5000  # Overridden in dev
        assert config_manager.get('data.validation_split') == 0.2  # From base
        assert config_manager.get('training.epochs') == 10  # Overridden in dev
        assert config_manager.get('training.use_gpu') == True  # From base
    
    def test_environment_specific_config(self, temp_config_dir, sample_configs):
        """Test loading environment-specific configurations."""
        # Test production environment
        config_manager = ConfigurationManager(
            config_dir=str(temp_config_dir),
            environment='production'
        )
        
        assert config_manager.get('data.base_path') == '/prod/path'
        assert config_manager.get('training.epochs') == 200
        assert config_manager.get('security.api_rate_limit') == 1000
        
        # Values from base config should still be available
        assert config_manager.get('data.validation_split') == 0.2
        assert config_manager.get('training.use_gpu') == True
    
    @patch.dict(os.environ, {
        'COLORAN_DATA_PATH': '/env/data/path',
        'COLORAN_GPU_ENABLED': 'false',
        'COLORAN_BATCH_SIZE': '2048',
        'COLORAN_JWT_SECRET': 'env_jwt_secret'
    })
    def test_environment_variable_overrides(self, temp_config_dir, sample_configs):
        """Test that environment variables override config files."""
        config_manager = ConfigurationManager(
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        # Check environment variable overrides
        assert config_manager.get('data.base_path') == '/env/data/path'
        assert config_manager.get('training.use_gpu') == False
        assert config_manager.get('training.batch_size') == 2048
        assert config_manager.get('security.jwt_secret') == 'env_jwt_secret'
    
    def test_deep_merge(self, temp_config_dir):
        """Test deep merging of configuration dictionaries."""
        config_manager = ConfigurationManager(config_dir=str(temp_config_dir))
        
        base_dict = {
            'level1': {
                'level2a': {'value': 'base'},
                'level2b': {'value': 'base_only'}
            }
        }
        
        update_dict = {
            'level1': {
                'level2a': {'value': 'updated'},
                'level2c': {'value': 'new'}
            }
        }
        
        config_manager._deep_merge(base_dict, update_dict)
        
        assert base_dict['level1']['level2a']['value'] == 'updated'
        assert base_dict['level1']['level2b']['value'] == 'base_only'
        assert base_dict['level1']['level2c']['value'] == 'new'
    
    def test_set_nested_value(self, temp_config_dir):
        """Test setting nested configuration values."""
        config_manager = ConfigurationManager(config_dir=str(temp_config_dir))
        
        # Test setting new nested value
        config_manager._set_nested_value('new.nested.value', '123')
        assert config_manager.get('new.nested.value') == 123  # Should be converted to int
        
        # Test setting boolean
        config_manager._set_nested_value('bool.value', 'true')
        assert config_manager.get('bool.value') == True
        
        # Test setting float
        config_manager._set_nested_value('float.value', '3.14')
        assert config_manager.get('float.value') == 3.14
        
        # Test setting string
        config_manager._set_nested_value('string.value', 'hello')
        assert config_manager.get('string.value') == 'hello'
    
    def test_validation_required_paths(self, temp_config_dir):
        """Test validation of required configuration paths."""
        # Create config without required values
        minimal_config = {}
        
        with open(temp_config_dir / 'base.yaml', 'w') as f:
            yaml.dump(minimal_config, f)
        
        with pytest.raises(ValueError, match="Required configuration .* is missing"):
            ConfigurationManager(config_dir=str(temp_config_dir))
    
    def test_validation_batch_size(self, temp_config_dir):
        """Test validation of batch size."""
        invalid_config = {
            'data': {'base_path': '/path'},
            'training': {'batch_size': 0},  # Invalid
            'model': {'save_path': '/model/path'}
        }
        
        with open(temp_config_dir / 'base.yaml', 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ValueError, match="training.batch_size must be positive"):
            ConfigurationManager(config_dir=str(temp_config_dir))
    
    def test_jwt_secret_validation(self, temp_config_dir, caplog):
        """Test JWT secret validation and warnings."""
        config_with_short_secret = {
            'data': {'base_path': '/path'},
            'training': {'batch_size': 1000},
            'model': {'save_path': '/model/path'},
            'security': {'jwt_secret': 'short'}  # Too short
        }
        
        with open(temp_config_dir / 'base.yaml', 'w') as f:
            yaml.dump(config_with_short_secret, f)
        
        ConfigurationManager(config_dir=str(temp_config_dir))
        
        # Should log warning about short JWT secret
        assert "JWT secret should be at least 32 characters long" in caplog.text
    
    def test_get_method(self, temp_config_dir, sample_configs):
        """Test the get method with various scenarios."""
        config_manager = ConfigurationManager(
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        # Test existing value
        assert config_manager.get('data.base_path') == '/dev/path'
        
        # Test non-existing value with default
        assert config_manager.get('non.existing.path', 'default') == 'default'
        
        # Test non-existing value without default
        assert config_manager.get('non.existing.path') is None
        
        # Test partial path
        data_config = config_manager.get('data')
        assert isinstance(data_config, dict)
        assert 'base_path' in data_config
    
    def test_set_method(self, temp_config_dir, sample_configs):
        """Test the set method."""
        config_manager = ConfigurationManager(
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        # Set new value
        config_manager.set('new.setting', 'test_value')
        assert config_manager.get('new.setting') == 'test_value'
        
        # Override existing value
        config_manager.set('data.base_path', '/new/path')
        assert config_manager.get('data.base_path') == '/new/path'
    
    def test_get_config_methods(self, temp_config_dir, sample_configs):
        """Test specific config getter methods."""
        config_manager = ConfigurationManager(
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        # Test training config
        training_config = config_manager.get_training_config()
        assert isinstance(training_config, dict)
        assert 'epochs' in training_config
        assert training_config['epochs'] == 10
        
        # Test data config
        data_config = config_manager.get_data_config()
        assert isinstance(data_config, dict)
        assert 'base_path' in data_config
        
        # Test model config
        model_config = config_manager.get_model_config()
        assert isinstance(model_config, dict)
        
        # Test security config
        security_config = config_manager.get_security_config()
        assert isinstance(security_config, dict)
        assert 'jwt_secret' in security_config
    
    def test_save_config(self, temp_config_dir, sample_configs):
        """Test saving configuration to file."""
        config_manager = ConfigurationManager(
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        # Modify a value
        config_manager.set('test.value', 'modified')
        
        # Save config
        save_path = temp_config_dir / 'saved_config.yaml'
        config_manager.save_config(str(save_path))
        
        assert save_path.exists()
        
        # Load and verify
        with open(save_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['test']['value'] == 'modified'
    
    def test_create_default_configs(self, temp_config_dir):
        """Test creation of default configuration files."""
        config_manager = ConfigurationManager(config_dir=str(temp_config_dir))
        config_manager.create_default_configs()
        
        # Check that default files were created
        assert (temp_config_dir / 'base.yaml').exists()
        assert (temp_config_dir / 'development.yaml').exists()
        assert (temp_config_dir / 'production.yaml').exists()
        
        # Verify content of base config
        with open(temp_config_dir / 'base.yaml', 'r') as f:
            base_config = yaml.safe_load(f)
        
        assert 'data' in base_config
        assert 'training' in base_config
        assert 'model' in base_config
        assert 'optimization' in base_config
        assert 'logging' in base_config
        assert 'security' in base_config
    
    def test_global_config_functions(self, temp_config_dir, sample_configs):
        """Test global configuration functions."""
        # Test init_config
        config_manager = init_config(
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        assert isinstance(config_manager, ConfigurationManager)
        assert config_manager.environment == 'development'
        
        # Test get_config
        global_config = get_config()
        assert global_config is config_manager
        
        # Test get_config with different environment
        new_config = get_config('production')
        assert new_config.environment == 'production'
        assert new_config is not config_manager
    
    def test_missing_config_files(self, temp_config_dir):
        """Test behavior with missing configuration files."""
        # Create config manager with non-existent directory
        empty_dir = temp_config_dir / 'empty'
        empty_dir.mkdir()
        
        config_manager = ConfigurationManager(config_dir=str(empty_dir))
        
        # Should initialize with empty config
        assert config_manager._config == {}
        
        # get() should return defaults
        assert config_manager.get('non.existent', 'default') == 'default'
    
    def test_config_file_format_errors(self, temp_config_dir):
        """Test handling of malformed configuration files."""
        # Create invalid YAML file
        invalid_yaml = temp_config_dir / 'base.yaml'
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        # Should handle gracefully
        with pytest.raises(yaml.YAMLError):
            ConfigurationManager(config_dir=str(temp_config_dir))


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_configuration_in_application_context(self, tmp_path):
        """Test configuration usage in application context."""
        # Create realistic configuration
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        app_config = {
            'data': {
                'base_path': str(tmp_path / "data"),
                'batch_size': 50000,
                'validation_split': 0.2,
                'random_seed': 42,
                'quality_threshold': 80.0
            },
            'training': {
                'use_gpu': False,  # For testing
                'sample_size': 100000,
                'batch_size': 1024,
                'epochs': 5,
                'early_stopping_patience': 3
            },
            'model': {
                'save_path': str(tmp_path / "models"),
                'rf_n_estimators': 50,
                'rf_max_depth': 10
            },
            'optimization': {
                'total_rbgs': 17,
                'timeout_seconds': 300
            },
            'security': {
                'jwt_secret': 'test_secret_key_that_is_long_enough_for_security',
                'api_rate_limit': 100
            }
        }
        
        with open(config_dir / 'base.yaml', 'w') as f:
            yaml.dump(app_config, f)
        
        # Initialize configuration
        config_manager = ConfigurationManager(
            config_dir=str(config_dir),
            environment='development'
        )
        
        # Test that all required components can access config
        training_config = config_manager.get_training_config()
        assert training_config['batch_size'] == 1024
        assert training_config['epochs'] == 5
        
        data_config = config_manager.get_data_config()
        assert data_config['random_seed'] == 42
        assert data_config['quality_threshold'] == 80.0
        
        model_config = config_manager.get_model_config()
        assert model_config['rf_n_estimators'] == 50
        
        security_config = config_manager.get_security_config()
        assert len(security_config['jwt_secret']) >= 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])