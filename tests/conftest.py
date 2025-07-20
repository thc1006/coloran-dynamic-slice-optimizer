# Pytest configuration and fixtures

import pytest

@pytest.fixture(scope="session")
def gpu_available():
    """Fixture to check if a GPU is available for testing."""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except ImportError:
        return False

@pytest.fixture(scope="session")
def gpu_device():
    """Fixture to return the default GPU device."""
    if gpu_available():
        import cupy as cp
        return cp.cuda.Device(0)
    return None
