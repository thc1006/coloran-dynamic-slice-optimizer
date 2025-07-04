# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data.data_processor import MemoryOptimizedProcessor

@pytest.fixture
def sample_data():
    data = {
        'Timestamp': [1672531200000 + i * 1000 for i in range(10)],
        'sched_policy': ['sched0'] * 5 + ['sched1'] * 5,
        'training_config': ['tr0'] * 10,
        'slice_id': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        'sum_requested_prbs': [10, 5, 2, 12, 6, 3, 8, 4, 1, 9],
        'sum_granted_prbs': [10, 5, 2, 10, 6, 3, 8, 4, 1, 8],
        'tx_brate downlink [Mbps]': [50, 20, 5, 60, 25, 8, 40, 15, 3, 45],
        'tx_errors downlink (%)': [0, 1, 0, 2, 0, 1, 0, 0, 3, 1],
    }
    return pd.DataFrame(data)

@pytest.fixture
def slice_configs():
    return {'tr0': [8, 6, 3]}

def test_processor_creates_features(sample_data, slice_configs):
    processor = MemoryOptimizedProcessor(slice_configs, batch_size=10)
    processed = processor.process_data(sample_data)
    
    assert 'allocation_efficiency' in processed.columns
    assert not processed['allocation_efficiency'].isnull().any()
    assert 'prb_utilization' in processed.columns
    assert processed['prb_utilization'].max() <= 1.0

def test_memory_optimization(sample_data, slice_configs):
    processor = MemoryOptimizedProcessor(slice_configs, batch_size=10)
    processed = processor.process_data(sample_data)
    optimized = processor.optimize_datatypes(processed.copy())
    
    initial_mem = processed.memory_usage(deep=True).sum()
    optimized_mem = optimized.memory_usage(deep=True).sum()
    
    assert optimized_mem < initial_mem
