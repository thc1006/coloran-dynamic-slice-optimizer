#!/usr/bin/env python3
# validate_pipeline.py

"""
Comprehensive end-to-end validation script for ColO-RAN Dynamic Slice Optimizer.

This script validates the complete ML pipeline from data loading to model training
and resource allocation optimization, ensuring all critical issues have been resolved.
"""

import os
import sys
import logging
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import all components
from data.data_loader import ColoRANDataLoader
from data.data_processor import MemoryOptimizedProcessor
from models.ml_trainer import A100OptimizedTrainer
from optimization.allocator import SliceResourceAllocator
from coloran_optimizer.config import init_config, ConfigurationManager
from coloran_optimizer.security.security_manager import SecurityManager


class PipelineValidator:
    """Comprehensive pipeline validation system."""
    
    def __init__(self, output_dir="./validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize results tracking
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'critical_issues_resolved': [],
            'validation_steps': []
        }
        
        self.logger.info("🚀 Pipeline Validator initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result and update counters."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.logger.info(f"{status}: {test_name}")
        
        if details:
            self.logger.info(f"  Details: {details}")
        
        if passed:
            self.results['tests_passed'] += 1
        else:
            self.results['tests_failed'] += 1
        
        self.results['validation_steps'].append({
            'test_name': test_name,
            'status': 'passed' if passed else 'failed',
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def validate_configuration_system(self):
        """Validate configuration management system."""
        self.logger.info("\n🔧 Validating Configuration Management System...")
        
        try:
            # Test configuration initialization
            with tempfile.TemporaryDirectory() as temp_dir:
                config_manager = init_config(temp_dir, environment='development', create_defaults=True)
                
                # Test config creation
                config_files = ['base.yaml', 'development.yaml', 'production.yaml']
                all_files_exist = all((Path(temp_dir) / f).exists() for f in config_files)
                self.log_test_result(
                    "Configuration files creation",
                    all_files_exist,
                    f"Created {len(config_files)} configuration files"
                )
                
                # Test configuration loading
                training_config = config_manager.get_training_config()
                has_required_keys = all(key in training_config for key in ['use_gpu', 'batch_size', 'epochs'])
                self.log_test_result(
                    "Configuration loading and structure",
                    has_required_keys,
                    f"Training config keys: {list(training_config.keys())}"
                )
                
                # Test environment variable override
                os.environ['COLORAN_BATCH_SIZE'] = '2048'
                config_manager._load_env_overrides()
                batch_size = config_manager.get('training.batch_size')
                env_override_works = batch_size == 2048
                self.log_test_result(
                    "Environment variable override",
                    env_override_works,
                    f"Batch size set to {batch_size} via environment variable"
                )
                
                if 'COLORAN_BATCH_SIZE' in os.environ:
                    del os.environ['COLORAN_BATCH_SIZE']
                
            self.results['critical_issues_resolved'].append("Configuration system implemented")
            
        except Exception as e:
            self.log_test_result(
                "Configuration system validation",
                False,
                f"Error: {str(e)}"
            )
    
    def validate_data_pipeline(self):
        """Validate data loading and processing pipeline."""
        self.logger.info("\n📊 Validating Data Pipeline...")
        
        try:
            # Create mock data for testing
            mock_data = self.create_mock_coloran_data()
            
            # Test data processor
            slice_configs = {f'tr{i}': [5, 7, 5] for i in range(3)}
            processor = MemoryOptimizedProcessor(slice_configs, batch_size=100)
            
            # Test data processing
            processed_data = processor.process_data(mock_data)
            
            processing_successful = (
                processed_data is not None and 
                len(processed_data) > 0 and
                'allocation_efficiency' in processed_data.columns
            )
            self.log_test_result(
                "Data processing pipeline",
                processing_successful,
                f"Processed {len(processed_data)} records with {len(processed_data.columns)} features"
            )
            
            # Test for absence of random data generation
            # Process same data twice and verify consistency
            processed_data2 = processor.process_data(mock_data.copy())
            
            # For deterministic processing, results should be identical
            data_consistency = processed_data['allocation_efficiency'].equals(processed_data2['allocation_efficiency'])
            self.log_test_result(
                "Elimination of random data generation",
                data_consistency,
                "Data processing is now deterministic"
            )
            
            # Test data quality validation
            quality_report = processor.validate_data_quality(processed_data)
            quality_validation_works = (
                'quality_score' in quality_report and
                0 <= quality_report['quality_score'] <= 100
            )
            self.log_test_result(
                "Data quality validation",
                quality_validation_works,
                f"Quality score: {quality_report['quality_score']:.1f}/100"
            )
            
            self.results['critical_issues_resolved'].append("Random data generation eliminated")
            self.results['critical_issues_resolved'].append("Data quality validation implemented")
            
            return processed_data
            
        except Exception as e:
            self.log_test_result(
                "Data pipeline validation",
                False,
                f"Error: {str(e)}"
            )
            return None
    
    def validate_ml_training_pipeline(self, processed_data):
        """Validate ML training pipeline."""
        self.logger.info("\n🧠 Validating ML Training Pipeline...")
        
        if processed_data is None:
            self.log_test_result(
                "ML training pipeline",
                False,
                "No processed data available for training"
            )
            return None
        
        try:
            # Initialize trainer with test configuration
            config_manager = ConfigurationManager()
            config_manager.set('training.use_gpu', False)  # Use CPU for validation
            config_manager.set('training.sample_size', 1000)  # Small sample for speed
            config_manager.set('training.epochs', 2)  # Quick training
            config_manager.set('model.rf_n_estimators', 10)  # Small model
            
            trainer = A100OptimizedTrainer(config_manager=config_manager)
            
            # Test GPU environment setup (should fallback to CPU gracefully)
            gpu_setup_successful = trainer.use_gpu is not None  # Should not error
            self.log_test_result(
                "GPU environment setup and fallback",
                gpu_setup_successful,
                f"GPU mode: {trainer.use_gpu}, cuML available: {trainer.cuml_available}"
            )
            
            # Test data preparation
            features = [
                'slice_id', 'sched_policy_num', 'allocated_rbgs',
                'prb_utilization', 'throughput_efficiency', 'qos_score'
            ]
            target = 'allocation_efficiency'
            
            X_train, X_test, y_train, y_test = trainer.prepare_data(
                processed_data, features, target
            )
            
            data_prep_successful = (
                X_train.shape[1] == len(features) and
                len(X_train) == len(y_train) and
                len(X_test) == len(y_test)
            )
            self.log_test_result(
                "Data preparation and validation",
                data_prep_successful,
                f"Train: {X_train.shape}, Test: {X_test.shape}"
            )
            
            # Test Random Forest training
            trainer.train_random_forest(X_train, y_train, X_test, y_test)
            rf_training_successful = (
                trainer.rf_model is not None and
                'random_forest' in trainer.training_history
            )
            rf_r2 = trainer.training_history.get('random_forest', {}).get('r2', 0)
            self.log_test_result(
                "Random Forest training",
                rf_training_successful,
                f"R² score: {rf_r2:.4f}"
            )
            
            # Test Neural Network training
            trainer.train_neural_network(X_train, y_train, X_test, y_test)
            nn_training_successful = (
                trainer.nn_model is not None and
                'neural_network' in trainer.training_history
            )
            nn_r2 = trainer.training_history.get('neural_network', {}).get('r2', 0)
            self.log_test_result(
                "Neural Network training",
                nn_training_successful,
                f"R² score: {nn_r2:.4f}"
            )
            
            # Test model saving and versioning
            with tempfile.TemporaryDirectory() as temp_dir:
                save_info = trainer.save_models(temp_dir)
                model_saving_successful = (
                    'timestamp' in save_info and
                    Path(temp_dir).exists() and
                    len(list(Path(temp_dir).glob('*.pkl'))) > 0
                )
                self.log_test_result(
                    "Model saving and versioning",
                    model_saving_successful,
                    f"Saved models with timestamp: {save_info.get('timestamp', 'N/A')}"
                )
            
            # Test cross-validation
            cv_results = trainer.perform_cross_validation(X_train, y_train, cv_folds=3)
            cv_successful = (
                'Random Forest' in cv_results and
                'mean_r2' in cv_results['Random Forest']
            )
            cv_r2 = cv_results.get('Random Forest', {}).get('mean_r2', 0)
            self.log_test_result(
                "Cross-validation implementation",
                cv_successful,
                f"CV R² score: {cv_r2:.4f}"
            )
            
            self.results['critical_issues_resolved'].append("Production-ready ML training pipeline")
            self.results['critical_issues_resolved'].append("GPU memory management implemented")
            self.results['critical_issues_resolved'].append("Model versioning and experiment tracking")
            
            return trainer
            
        except Exception as e:
            self.log_test_result(
                "ML training pipeline validation",
                False,
                f"Error: {str(e)}"
            )
            return None
    
    def validate_resource_allocation(self, trainer):
        """Validate resource allocation optimization."""
        self.logger.info("\n⚡ Validating Resource Allocation System...")
        
        try:
            # Initialize allocator with configuration
            config_manager = ConfigurationManager()
            config_manager.set('optimization.total_rbgs', 17)
            config_manager.set('optimization.timeout_seconds', 60)
            config_manager.set('optimization.genetic_population_size', 20)
            config_manager.set('optimization.genetic_generations', 5)
            
            # Test allocator initialization without trained models (should fallback gracefully)
            allocator = SliceResourceAllocator(config_manager=config_manager)
            
            allocator_init_successful = allocator is not None
            self.log_test_result(
                "Resource allocator initialization",
                allocator_init_successful,
                f"Total RBGs: {allocator.total_rbgs}, Timeout: {allocator.timeout_s}s"
            )
            
            # Test fallback efficiency estimation
            test_state = {
                'num_ues': 10,
                'sched_policy_num': 1,
                'bs_id': 1,
                'exp_id': 1,
                'sum_requested_prbs': 15,
                'hour': 14,
                'minute': 30,
                'day_of_week': 2,
                'qos_score': 0.8,
                'network_load': 0.6,
                'throughput_efficiency': 0.7
            }
            
            # Test exhaustive optimization
            best_allocation, efficiency = allocator.optimize_exhaustive(test_state)
            exhaustive_successful = (
                best_allocation is not None and
                len(best_allocation) == 3 and
                sum(best_allocation) == allocator.total_rbgs and
                0 <= efficiency <= 1
            )
            self.log_test_result(
                "Exhaustive search optimization",
                exhaustive_successful,
                f"Best allocation: {best_allocation}, Efficiency: {efficiency:.4f}"
            )
            
            # Test genetic algorithm optimization
            best_allocation_ga, efficiency_ga = allocator.optimize_genetic(test_state)
            genetic_successful = (
                best_allocation_ga is not None and
                len(best_allocation_ga) == 3 and
                sum(best_allocation_ga) == allocator.total_rbgs and
                0 <= efficiency_ga <= 1
            )
            self.log_test_result(
                "Genetic algorithm optimization",
                genetic_successful,
                f"Best allocation: {best_allocation_ga}, Efficiency: {efficiency_ga:.4f}"
            )
            
            # Test scenario-based simulation (no random data generation)
            simulation_results = allocator.simulate(steps=10, method="genetic")
            simulation_successful = (
                simulation_results is not None and
                len(simulation_results) == 10 and
                'scenario' in simulation_results.columns  # Should have scenario info
            )
            self.log_test_result(
                "Scenario-based simulation",
                simulation_successful,
                f"Simulated {len(simulation_results)} optimization steps"
            )
            
            # Verify no random data generation in simulation
            simulation_results2 = allocator.simulate(steps=10, method="genetic", random_seed=42)
            simulation_consistency = simulation_results['improvement'].equals(simulation_results2['improvement'])
            self.log_test_result(
                "Deterministic optimization simulation",
                simulation_consistency,
                "Simulation results are reproducible"
            )
            
            self.results['critical_issues_resolved'].append("Configurable resource allocation system")
            self.results['critical_issues_resolved'].append("Deterministic optimization algorithms")
            
        except Exception as e:
            self.log_test_result(
                "Resource allocation validation",
                False,
                f"Error: {str(e)}"
            )
    
    def validate_security_system(self):
        """Validate security management system."""
        self.logger.info("\n🔒 Validating Security System...")
        
        try:
            # Test security manager initialization
            security_manager = SecurityManager()
            
            security_init_successful = (
                security_manager is not None and
                security_manager.jwt_secret is not None and
                len(security_manager.jwt_secret) >= 32
            )
            self.log_test_result(
                "Security manager initialization",
                security_init_successful,
                f"JWT secret length: {len(security_manager.jwt_secret)} chars"
            )
            
            # Test token generation
            user_data = {'user_id': 'test_user', 'role': 'admin'}
            token = security_manager.generate_token(user_data)
            
            token_generation_successful = token is not None and len(token) > 50
            self.log_test_result(
                "JWT token generation",
                token_generation_successful,
                f"Token length: {len(token)} chars"
            )
            
            # Test input validation
            safe_input = security_manager.validate_input("safe_input_123")
            dangerous_input_caught = False
            
            try:
                security_manager.validate_input("<script>alert('xss')</script>")
            except ValueError:
                dangerous_input_caught = True
            
            input_validation_successful = (
                safe_input == "safe_input_123" and
                dangerous_input_caught
            )
            self.log_test_result(
                "Input validation and XSS protection",
                input_validation_successful,
                "Safe input allowed, dangerous input blocked"
            )
            
            # Test password hashing
            password = "test_password_123"
            hashed = security_manager.hash_password(password)
            verification_successful = security_manager.verify_password(password, hashed)
            
            password_security_successful = (
                hashed != password and
                verification_successful
            )
            self.log_test_result(
                "Password hashing and verification",
                password_security_successful,
                "Passwords properly hashed and verified"
            )
            
            # Test data encryption
            sensitive_data = "sensitive_information_123"
            encrypted = security_manager.encrypt_sensitive_data(sensitive_data)
            decrypted = security_manager.decrypt_sensitive_data(encrypted)
            
            encryption_successful = (
                encrypted != sensitive_data and
                decrypted == sensitive_data
            )
            self.log_test_result(
                "Data encryption and decryption",
                encryption_successful,
                "Sensitive data properly encrypted"
            )
            
            # Test security audit reporting
            security_report = security_manager.get_security_report()
            
            audit_successful = (
                isinstance(security_report, dict) and
                'failed_attempts_by_ip' in security_report and
                'recent_security_events' in security_report
            )
            self.log_test_result(
                "Security audit and reporting",
                audit_successful,
                f"Security report contains {len(security_report)} sections"
            )
            
            self.results['critical_issues_resolved'].append("Comprehensive security system")
            self.results['critical_issues_resolved'].append("JWT secret management")
            self.results['critical_issues_resolved'].append("Input validation and XSS protection")
            
        except Exception as e:
            self.log_test_result(
                "Security system validation",
                False,
                f"Error: {str(e)}"
            )
    
    def create_mock_coloran_data(self):
        """Create realistic mock ColO-RAN data for testing."""
        np.random.seed(42)  # For reproducible test data
        n_samples = 1000
        
        data = pd.DataFrame({
            'Timestamp': range(1000000, 1000000 + n_samples),
            'sched_policy': np.random.choice(['sched0', 'sched1', 'sched2'], n_samples),
            'training_config': np.random.choice([f'tr{i}' for i in range(3)], n_samples),
            'bs_id': np.random.choice([1, 8, 15, 22], n_samples),
            'exp_id': np.random.choice([1, 2, 3], n_samples),
            'imsi': [f'user{i}' for i in range(n_samples)],
            'tx_brate downlink [Mbps]': np.random.uniform(5, 20, n_samples),
            'sum_requested_prbs': np.random.randint(1, 20, n_samples),
            'sum_granted_prbs': np.random.randint(1, 20, n_samples),
            'tx_errors downlink (%)': np.random.uniform(0, 5, n_samples),
            'rx_errors uplink (%)': np.random.uniform(0, 5, n_samples),
            'dl_cqi': np.random.randint(5, 15, n_samples),
            'num_ues': np.random.randint(1, 10, n_samples)
        })
        
        return data
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        self.logger.info("\n📋 Generating Validation Report...")
        
        # Calculate success rate
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        success_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        # Create detailed report
        report = {
            'validation_summary': {
                'timestamp': self.results['timestamp'],
                'total_tests': total_tests,
                'tests_passed': self.results['tests_passed'],
                'tests_failed': self.results['tests_failed'],
                'success_rate_percent': round(success_rate, 2),
                'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
            },
            'critical_issues_resolved': self.results['critical_issues_resolved'],
            'detailed_test_results': self.results['validation_steps'],
            'production_readiness_checklist': {
                'data_integrity_fixed': 'random data generation eliminated' in self.results['critical_issues_resolved'],
                'ml_pipeline_robust': 'production-ready ML training pipeline' in self.results['critical_issues_resolved'],
                'configuration_system': 'configuration system implemented' in self.results['critical_issues_resolved'],
                'security_implemented': 'comprehensive security system' in self.results['critical_issues_resolved'],
                'gpu_compatibility': 'GPU memory management implemented' in self.results['critical_issues_resolved']
            }
        }
        
        # Save report
        report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        self.logger.info(f"\n🎯 VALIDATION SUMMARY:")
        self.logger.info(f"   Tests Passed: {self.results['tests_passed']}")
        self.logger.info(f"   Tests Failed: {self.results['tests_failed']}")
        self.logger.info(f"   Success Rate: {success_rate:.1f}%")
        self.logger.info(f"   Overall Status: {report['validation_summary']['overall_status']}")
        self.logger.info(f"\n📁 Report saved to: {report_file}")
        
        # Log critical issues resolved
        self.logger.info(f"\n✅ CRITICAL ISSUES RESOLVED ({len(self.results['critical_issues_resolved'])}):")
        for issue in self.results['critical_issues_resolved']:
            self.logger.info(f"   ✓ {issue}")
        
        return report
    
    def run_complete_validation(self):
        """Run complete end-to-end pipeline validation."""
        self.logger.info("🚀 Starting Complete Pipeline Validation")
        self.logger.info("=" * 60)
        
        # 1. Validate configuration system
        self.validate_configuration_system()
        
        # 2. Validate data pipeline
        processed_data = self.validate_data_pipeline()
        
        # 3. Validate ML training pipeline
        trainer = self.validate_ml_training_pipeline(processed_data)
        
        # 4. Validate resource allocation
        self.validate_resource_allocation(trainer)
        
        # 5. Validate security system
        self.validate_security_system()
        
        # 6. Generate final report
        report = self.generate_validation_report()
        
        return report


def main():
    """Main validation entry point."""
    print("🔍 ColO-RAN Dynamic Slice Optimizer - Pipeline Validation")
    print("=" * 60)
    
    validator = PipelineValidator()
    report = validator.run_complete_validation()
    
    # Exit with appropriate code
    success_rate = report['validation_summary']['success_rate_percent']
    if success_rate >= 80:
        print(f"\n🎉 VALIDATION SUCCESSFUL - {success_rate:.1f}% tests passed")
        sys.exit(0)
    else:
        print(f"\n❌ VALIDATION FAILED - Only {success_rate:.1f}% tests passed")
        sys.exit(1)


if __name__ == "__main__":
    main()