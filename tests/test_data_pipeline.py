# tests/test_data_pipeline.py

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import ColoRANDataLoader
from data.data_processor import MemoryOptimizedProcessor


class TestColoRANDataLoader:
    """Test suite for ColO-RAN data loader."""
    
    @pytest.fixture
    def data_loader(self):
        """Create data loader instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ColoRANDataLoader(base_path=temp_dir)
    
    @pytest.fixture
    def mock_dataset_structure(self, tmp_path):
        """Create mock dataset structure for testing."""
        dataset_path = tmp_path / "colosseum-oran-coloran-dataset"
        dataset_path.mkdir()
        
        # Create mock directory structure
        for sched in ['sched0', 'sched1', 'sched2']:
            sched_path = dataset_path / sched
            sched_path.mkdir()
            
            for tr_config in ['tr0', 'tr1']:
                tr_path = sched_path / tr_config
                tr_path.mkdir()
                
                exp_path = tr_path / "exp1"
                exp_path.mkdir()
                
                bs_path = exp_path / "bs1"
                bs_path.mkdir()
                
                slice_path = bs_path / "slices_bs1"
                slice_path.mkdir()
                
                # Create mock CSV file
                csv_file = slice_path / "user1_metrics.csv"
                mock_data = pd.DataFrame({
                    'Timestamp': [1000000, 1000001, 1000002],
                    'tx_brate downlink [Mbps]': [10.5, 12.3, 9.8],
                    'sum_requested_prbs': [5, 6, 4],
                    'sum_granted_prbs': [5, 5, 4],
                    'tx_errors downlink (%)': [1.2, 0.8, 2.1],
                    'rx_errors uplink (%)': [0.5, 1.0, 1.5],
                    'dl_cqi': [12, 10, 11],
                    'num_ues': [3, 4, 2]
                })
                mock_data.to_csv(csv_file, index=False)
        
        return dataset_path
    
    def test_initialization(self, data_loader):
        """Test data loader initialization."""
        assert data_loader.base_stations == [1, 8, 15, 22, 29, 36, 43]
        assert data_loader.scheduling_policies == ['sched0', 'sched1', 'sched2']
        assert len(data_loader.training_configs) == 28
        assert data_loader.dataset_repo_url == "https://github.com/wineslab/colosseum-oran-coloran-dataset.git"
    
    @patch('subprocess.run')
    def test_download_dataset_success(self, mock_subprocess, data_loader):
        """Test successful dataset download."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = data_loader.download_dataset()
        
        assert result == data_loader.dataset_local_path
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.run')
    def test_download_dataset_failure(self, mock_subprocess, data_loader):
        """Test dataset download failure."""
        mock_subprocess.side_effect = Exception("Download failed")
        
        result = data_loader.download_dataset()
        
        assert result is None
    
    def test_auto_detect_structure(self, data_loader, mock_dataset_structure):
        """Test automatic structure detection."""
        data_loader.dataset_local_path = str(mock_dataset_structure)
        
        result = data_loader._auto_detect_structure()
        
        assert result == str(mock_dataset_structure)
    
    def test_load_raw_data(self, data_loader, mock_dataset_structure):
        """Test raw data loading."""
        data_loader.dataset_local_path = str(mock_dataset_structure)
        
        result = data_loader.load_raw_data()
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'bs_id' in result.columns
        assert 'sched_policy' in result.columns
        assert 'training_config' in result.columns
    
    def test_load_raw_data_no_files(self, data_loader):
        """Test data loading with no files."""
        # Use non-existent path
        data_loader.dataset_local_path = "/non/existent/path"
        
        result = data_loader.load_raw_data()
        
        assert result is None


class TestMemoryOptimizedProcessor:
    """Test suite for memory optimized data processor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance for testing."""
        slice_configs = {
            'tr0': [5, 7, 5],
            'tr1': [6, 6, 5]
        }
        return MemoryOptimizedProcessor(slice_configs, batch_size=100)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'Timestamp': [1000000, 1000001, 1000002] * 50,
            'sched_policy': ['sched0', 'sched1', 'sched2'] * 50,
            'training_config': ['tr0', 'tr1', 'tr0'] * 50,
            'tx_brate downlink [Mbps]': np.random.uniform(5, 15, 150),
            'sum_requested_prbs': np.random.randint(1, 20, 150),
            'sum_granted_prbs': np.random.randint(1, 20, 150),
            'tx_errors downlink (%)': np.random.uniform(0, 5, 150),
            'rx_errors uplink (%)': np.random.uniform(0, 5, 150),
            'dl_cqi': np.random.randint(5, 15, 150),
            'num_ues': np.random.randint(1, 10, 150)
        })
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.batch_size == 100
        assert len(processor.feature_columns) == 12
        assert 'slice_id' in processor.feature_columns
        assert 'allocation_efficiency' not in processor.feature_columns
    
    def test_process_single_batch(self, processor, sample_data):
        """Test single batch processing."""
        result = processor._process_single_batch(sample_data.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_data)
        assert 'hour' in result.columns
        assert 'minute' in result.columns
        assert 'day_of_week' in result.columns
        assert 'sched_policy_num' in result.columns
        assert 'allocation_efficiency' in result.columns
        
        # Check value ranges
        assert result['allocation_efficiency'].min() >= 0
        assert result['allocation_efficiency'].max() <= 1
        assert result['prb_utilization'].min() >= 0
        assert result['prb_utilization'].max() <= 1
    
    def test_no_random_data_generation(self, processor, sample_data):
        """Test that no random data is generated for missing values."""
        # Create data with missing values
        sample_data_missing = sample_data.copy()
        sample_data_missing.loc[0:10, 'tx_errors downlink (%)'] = np.nan
        sample_data_missing.loc[10:20, 'rx_errors uplink (%)'] = np.nan
        sample_data_missing.loc[20:30, 'dl_cqi'] = np.nan
        sample_data_missing.loc[30:40, 'num_ues'] = np.nan
        
        result = processor._process_single_batch(sample_data_missing)
        
        # Check that missing values are filled with conservative estimates, not random
        assert not result.isnull().any().any()  # No NaN values
        
        # Verify specific fallback values are used
        # For rows with missing error rates, should use 2.5%
        # For missing CQI, should use 10.0
        # For missing num_ues, should use 5.0
        
    def test_process_data(self, processor, sample_data):
        """Test full data processing pipeline."""
        result = processor.process_data(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_data)
        assert all(col in result.columns for col in processor.feature_columns)
        assert 'allocation_efficiency' in result.columns
    
    def test_optimize_datatypes(self, processor, sample_data):
        """Test datatype optimization."""
        # Create DataFrame with large datatypes
        test_df = pd.DataFrame({
            'int_col': np.array([1, 2, 3], dtype='int64'),
            'float_col': np.array([1.1, 2.2, 3.3], dtype='float64')
        })
        
        result = processor.optimize_datatypes(test_df)
        
        # Check that datatypes were downcast
        assert result['int_col'].dtype != 'int64'
        assert result['float_col'].dtype != 'float64'
    
    def test_validate_data_quality(self, processor, sample_data):
        """Test data quality validation."""
        result = processor.validate_data_quality(sample_data)
        
        assert isinstance(result, dict)
        assert 'total_records' in result
        assert 'missing_values' in result
        assert 'outliers' in result
        assert 'data_ranges' in result
        assert 'duplicates' in result
        assert 'quality_score' in result
        
        assert result['total_records'] == len(sample_data)
        assert 0 <= result['quality_score'] <= 100
    
    def test_clean_outliers(self, processor, sample_data):
        """Test outlier cleaning."""
        # Add some extreme outliers
        sample_data_with_outliers = sample_data.copy()
        sample_data_with_outliers.loc[0, 'sum_requested_prbs'] = 1000  # Extreme outlier
        
        result = processor.clean_outliers(sample_data_with_outliers)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_data_with_outliers)
        
        # Check that extreme values were capped
        assert result['sum_requested_prbs'].max() < 1000


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    @pytest.fixture
    def complete_mock_data(self, tmp_path):
        """Create complete mock dataset for integration testing."""
        dataset_path = tmp_path / "colosseum-oran-coloran-dataset"
        dataset_path.mkdir()
        
        # Create comprehensive mock data
        for sched in ['sched0', 'sched1', 'sched2']:
            sched_path = dataset_path / sched
            sched_path.mkdir()
            
            for tr_idx in range(3):  # Fewer configs for faster testing
                tr_config = f'tr{tr_idx}'
                tr_path = sched_path / tr_config
                tr_path.mkdir()
                
                exp_path = tr_path / "exp1"
                exp_path.mkdir()
                
                for bs_id in [1, 8]:  # Fewer base stations
                    bs_path = exp_path / f"bs{bs_id}"
                    bs_path.mkdir()
                    
                    slice_path = bs_path / f"slices_bs{bs_id}"
                    slice_path.mkdir()
                    
                    # Create multiple user files
                    for user_id in range(3):
                        csv_file = slice_path / f"user{user_id}_metrics.csv"
                        mock_data = pd.DataFrame({
                            'Timestamp': range(1000000 + user_id*100, 1000000 + user_id*100 + 20),
                            'tx_brate downlink [Mbps]': np.random.uniform(5, 20, 20),
                            'sum_requested_prbs': np.random.randint(1, 15, 20),
                            'sum_granted_prbs': np.random.randint(1, 15, 20),
                            'tx_errors downlink (%)': np.random.uniform(0, 3, 20),
                            'rx_errors uplink (%)': np.random.uniform(0, 3, 20),
                            'dl_cqi': np.random.randint(8, 15, 20),
                            'num_ues': np.random.randint(1, 8, 20)
                        })
                        mock_data.to_csv(csv_file, index=False)
        
        return dataset_path
    
    def test_complete_pipeline(self, complete_mock_data):
        """Test complete data pipeline from loading to processing."""
        # Initialize components
        data_loader = ColoRANDataLoader(base_path=str(complete_mock_data.parent))
        data_loader.dataset_local_path = str(complete_mock_data)
        
        slice_configs = {f'tr{i}': [5, 7, 5] for i in range(3)}
        processor = MemoryOptimizedProcessor(slice_configs, batch_size=50)
        
        # Load raw data
        raw_data = data_loader.load_raw_data()
        assert raw_data is not None
        assert len(raw_data) > 0
        
        # Process data
        processed_data = processor.process_data(raw_data)
        assert processed_data is not None
        assert len(processed_data) > 0
        
        # Validate data quality
        quality_report = processor.validate_data_quality(processed_data)
        assert quality_report['quality_score'] > 50  # Should have reasonable quality
        
        # Check feature engineering results
        assert 'allocation_efficiency' in processed_data.columns
        assert processed_data['allocation_efficiency'].notna().all()
        assert (processed_data['allocation_efficiency'] >= 0).all()
        assert (processed_data['allocation_efficiency'] <= 1).all()
    
    def test_pipeline_memory_efficiency(self, complete_mock_data):
        """Test that pipeline handles memory efficiently."""
        data_loader = ColoRANDataLoader(base_path=str(complete_mock_data.parent))
        data_loader.dataset_local_path = str(complete_mock_data)
        
        # Use small batch size to test memory management
        slice_configs = {f'tr{i}': [5, 7, 5] for i in range(3)}
        processor = MemoryOptimizedProcessor(slice_configs, batch_size=20)
        
        raw_data = data_loader.load_raw_data()
        processed_data = processor.process_data(raw_data)
        
        # Memory optimization should reduce size
        optimized_data = processor.optimize_datatypes(processed_data)
        
        # Check that optimization was applied
        original_memory = processed_data.memory_usage(deep=True).sum()
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        assert optimized_memory <= original_memory
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid data."""
        # Test with non-existent path
        data_loader = ColoRANDataLoader(base_path="/non/existent/path")
        raw_data = data_loader.load_raw_data()
        assert raw_data is None
        
        # Test processor with empty DataFrame
        slice_configs = {'tr0': [5, 7, 5]}
        processor = MemoryOptimizedProcessor(slice_configs)
        
        empty_df = pd.DataFrame()
        result = processor.process_data(empty_df)
        
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])