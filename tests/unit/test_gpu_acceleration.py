import pytest
import cupy as cp
from src.coloran_optimizer.gpu.optimizer import GPUOptimizer

@pytest.mark.skipif(not pytest.gpu_available(), reason="GPU not available")
def test_gpu_optimizer_initialization(gpu_device):
    optimizer = GPUOptimizer(device_id=0)
    assert optimizer.device_id == 0

@pytest.mark.skipif(not pytest.gpu_available(), reason="GPU not available")
def test_gpu_optimizer_optimization(gpu_device):
    optimizer = GPUOptimizer(device_id=0)
    test_matrix = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    optimized_matrix = optimizer.optimize_allocation(test_matrix)
    assert cp.array_equal(optimized_matrix, test_matrix * 2)

@pytest.mark.skipif(not pytest.gpu_available(), reason="GPU not available")
def test_gpu_memory_allocation(gpu_device):
    optimizer = GPUOptimizer(device_id=0)
    initial_info = optimizer.get_gpu_memory_info()
    
    # Allocate 10MB
    size_to_allocate = 10 * 1024 * 1024 
    mem = optimizer.allocate_gpu_memory(size_to_allocate)
    assert mem is not None
    
    # Check if used memory increased
    after_alloc_info = optimizer.get_gpu_memory_info()
    assert after_alloc_info["used"] > initial_info["used"]
    
    # Free memory (rely on CuPy's garbage collection for explicit freeing)
    del mem
    # Trigger garbage collection to ensure memory is released for testing purposes
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    
    final_info = optimizer.get_gpu_memory_info()
    # This assertion might be tricky due to CuPy's memory management, 
    # but generally, used memory should decrease or return to near initial state.
    assert final_info["used"] <= initial_info["used"] + 1024 # Allow for minor overhead

@pytest.mark.skipif(pytest.gpu_available(), reason="Test only for CPU fallback")
def test_gpu_optimizer_cpu_fallback():
    optimizer = GPUOptimizer(device_id=999) # Non-existent device
    assert optimizer.device_id == -1
    test_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    optimized_matrix = optimizer.optimize_allocation(test_matrix)
    assert np.array_equal(optimized_matrix, test_matrix * 2)