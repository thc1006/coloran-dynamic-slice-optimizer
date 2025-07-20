import pytest
import cupy as cp
import time
from src.coloran_optimizer.gpu.optimizer import GPUOptimizer

@pytest.mark.skipif(not pytest.gpu_available(), reason="GPU not available")
def test_gpu_optimization_performance(gpu_device):
    optimizer = GPUOptimizer(device_id=0)
    
    # Test with a larger matrix to measure performance
    matrix_size = 10000 # 10000x10000 matrix
    test_matrix = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)

    start_time = time.time()
    optimized_matrix = optimizer.optimize_allocation(test_matrix)
    cp.cuda.Stream.null.synchronize() # Ensure all GPU operations are complete
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nGPU optimization of {matrix_size}x{matrix_size} matrix took {duration:.4f} seconds.")
    
    # Assert that the operation was performed (e.g., values are changed)
    assert not cp.array_equal(optimized_matrix, test_matrix)
    # Assert that the performance is within an acceptable range (e.g., < 1 second for this size)
    # This threshold might need adjustment based on the actual GPU and operation complexity
    assert duration < 1.0

@pytest.mark.skipif(not pytest.gpu_available(), reason="GPU not available")
def test_gpu_memory_bandwidth(gpu_device):
    optimizer = GPUOptimizer(device_id=0)
    
    # Test memory bandwidth by performing a simple copy operation on a large array
    array_size = 500 * 1024 * 1024 // 4 # 500 MB of float32 data
    test_array = cp.random.rand(array_size, dtype=cp.float32)
    
    start_time = time.time()
    copied_array = cp.copy(test_array)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    bandwidth = (array_size * 4 * 2) / (duration * 1024 * 1024 * 1024) # Read and write, in GB/s
    print(f"\nGPU memory bandwidth test: {bandwidth:.2f} GB/s.")
    
    # Assert a reasonable bandwidth (e.g., > 100 GB/s for modern GPUs)
    assert bandwidth > 100.0
