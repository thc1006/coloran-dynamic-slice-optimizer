# tests/test_optimization.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.optimization.allocator import SliceResourceAllocator

@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    # Mock to return efficiency equal to the sum of allocated RBGs
    # divided by a constant, to have a predictable outcome.
    def mock_predict(feature_matrix):
        return np.clip(feature_matrix[:, 3] / 20.0, 0, 1) # allocated_rbgs is 4th feature
    predictor.predict.side_effect = mock_predict
    return predictor

@pytest.fixture
def mock_allocator(mock_predictor):
    # The allocator requires the predictor object, not the class
    allocator_obj = SliceResourceAllocator.__new__(SliceResourceAllocator)
    allocator_obj.model = mock_predictor
    allocator_obj.scaler = MagicMock()
    allocator_obj.features = [f'f{i}' for i in range(15)]
    allocator_obj.total_rbgs = 17
    return allocator_obj

def test_exhaustive_optimizer(mock_allocator):
    from src.optimization.exhaustive_search import exhaustive_optimizer
    
    # This mock evaluation function is simplified.
    # It returns the sum of products of allocation and some weights.
    def mock_evaluator(allocations):
        weights = np.array([0.2, 0.5, 0.3])
        return np.dot(np.array(allocations), weights)

    # We expect the optimizer to find the allocation that maximizes the dot product.
    # For total_rbgs=5, allocations are like [1,1,3], [1,2,2], [1,3,1], [2,1,2], [2,2,1], [3,1,1]
    # Scores: 1.6, 1.7, 2.0, 1.5, 1.6, 1.4. Max is 2.0 at [1,3,1]
    best_alloc, best_score = exhaustive_optimizer(mock_evaluator, total_rbgs=5)
    
    assert best_alloc == [1, 3, 1]
    assert best_score == pytest.approx(2.0)

