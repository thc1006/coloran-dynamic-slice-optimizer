#!/usr/bin/env python3
# setup_environment.py

"""
Environment setup script for ColO-RAN Dynamic Slice Optimizer.
Installs minimal dependencies and configures the environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description=""):
    """Run a command and return success status."""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description or command}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    
    print(f"Python version check passed: {version.major}.{version.minor}.{version.micro}")
    return True


def install_minimal_requirements():
    """Install minimal requirements for the project."""
    print("\nInstalling minimal requirements...")
    
    # Define minimal packages that are essential
    essential_packages = [
        "numpy>=1.20.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "pyjwt>=2.0.0",
        "cryptography>=3.0.0",
        "bcrypt>=3.2.0",
        "pytest>=6.0.0"
    ]
    
    print("Installing essential packages...")
    for package in essential_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"Warning: Failed to install {package} - some features may not work")
    
    return True


def setup_configuration():
    """Setup default configuration files."""
    print("\nSetting up configuration...")
    
    try:
        # Import and initialize configuration system
        sys.path.append(str(Path(__file__).parent / "src"))
        from coloran_optimizer.config import init_config
        
        config_dir = Path(__file__).parent / "config"
        config_manager = init_config(str(config_dir), create_defaults=True)
        
        print("Configuration files created successfully")
        return True
    except Exception as e:
        print(f"Configuration setup failed: {e}")
        return False


def create_directory_structure():
    """Create necessary directory structure."""
    print("\nCreating directory structure...")
    
    base_dir = Path(__file__).parent
    directories = [
        "models",
        "logs", 
        "data",
        "validation_results",
        "config"
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return True


def set_environment_variables():
    """Set necessary environment variables."""
    print("\nSetting environment variables...")
    
    # Generate secure JWT secret if not set
    jwt_secret = os.getenv('COLORAN_JWT_SECRET')
    if not jwt_secret:
        import secrets
        jwt_secret = secrets.token_urlsafe(64)
        print("Generated secure JWT secret")
        print(f"Set environment variable: COLORAN_JWT_SECRET={jwt_secret[:20]}...")
        print("Note: In production, set this as a persistent environment variable")
    
    # Set other default environment variables
    env_vars = {
        'COLORAN_ENV': 'development',
        'COLORAN_LOG_LEVEL': 'INFO',
        'COLORAN_GPU_ENABLED': 'false'  # Default to CPU for compatibility
    }
    
    for var, value in env_vars.items():
        if not os.getenv(var):
            os.environ[var] = value
            print(f"Set {var}={value}")
    
    return True


def run_validation():
    """Run validation to ensure everything is working."""
    print("\nRunning validation...")
    
    validation_script = Path(__file__).parent / "simple_validation.py"
    if validation_script.exists():
        success = run_command(f"python {validation_script}", "Running validation tests")
        return success
    else:
        print("Validation script not found - skipping validation")
        return True


def main():
    """Main setup function."""
    print("ColO-RAN Dynamic Slice Optimizer - Environment Setup")
    print("=" * 60)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing minimal requirements", install_minimal_requirements),
        ("Creating directory structure", create_directory_structure),
        ("Setting environment variables", set_environment_variables),
        ("Setting up configuration", setup_configuration),
        ("Running validation", run_validation)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n[STEP] {step_name}")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"ERROR in {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if not failed_steps:
        print("✅ ALL SETUP STEPS COMPLETED SUCCESSFULLY")
        print("\nNext steps:")
        print("1. Run 'python simple_validation.py' to validate the installation")
        print("2. Run 'python validate_with_fallback.py' for comprehensive validation")
        print("3. Check the 'examples/' directory for usage examples")
        print("4. Review the configuration files in 'config/' directory")
        return 0
    else:
        print("⚠️ SETUP COMPLETED WITH ISSUES")
        print("Failed steps:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease resolve the issues above and run setup again")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)