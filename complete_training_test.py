#!/usr/bin/env python3
# complete_training_test.py

"""
Complete Training Test for ColO-RAN Dynamic Slice Optimizer
This script demonstrates the entire ML pipeline from data loading to model deployment.
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import core dependencies with fallbacks
try:
    import numpy as np
    import pandas as pd
    print("✅ Core dependencies (numpy, pandas) loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import core dependencies: {e}")
    print("Please install: pip install numpy pandas")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    print("✅ Scikit-learn loaded successfully")
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Scikit-learn not available - will use basic ML implementation")
    SKLEARN_AVAILABLE = False

# Import project modules
try:
    from data.data_loader import ColoRANDataLoader
    from data.data_processor import MemoryOptimizedProcessor
    print("✅ Data modules loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import data modules: {e}")
    sys.exit(1)

try:
    from coloran_optimizer.config import ConfigurationManager, init_config
    print("✅ Configuration system loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import configuration system: {e}")
    sys.exit(1)


class CompleteTrainingTest:
    """Complete training test for ColO-RAN Dynamic Slice Optimizer."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent / "training_test_results"
        self.test_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize configuration
        self.setup_configuration()
        
        # Test results tracking
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_phases': [],
            'models_trained': [],
            'performance_metrics': {},
            'data_quality': {},
            'errors': []
        }
        
        self.logger.info("🚀 Complete Training Test Initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging for the training test."""
        log_file = self.test_dir / f"training_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def setup_configuration(self):
        """Setup configuration for training test."""
        self.logger.info("🔧 Setting up configuration...")
        
        try:
            # Create test configuration
            config_dir = self.test_dir / "config"
            self.config_manager = init_config(str(config_dir), environment='development', create_defaults=True)
            
            # Override for testing
            self.config_manager.set('training.use_gpu', False)  # Use CPU for testing
            self.config_manager.set('training.sample_size', 5000)  # Smaller sample for testing
            self.config_manager.set('training.epochs', 5)  # Quick training
            self.config_manager.set('data.batch_size', 1000)  # Smaller batches
            self.config_manager.set('model.rf_n_estimators', 20)  # Faster training
            
            self.logger.info("✅ Configuration setup completed")
            self.log_phase("Configuration Setup", True, "Test configuration created successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration setup failed: {e}")
            self.log_phase("Configuration Setup", False, str(e))
            raise
    
    def log_phase(self, phase_name: str, success: bool, details: str = ""):
        """Log test phase results."""
        status = "✅ PASSED" if success else "❌ FAILED"
        self.logger.info(f"{status}: {phase_name}")
        
        if details:
            self.logger.info(f"   Details: {details}")
        
        self.results['test_phases'].append({
            'phase': phase_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
        if not success:
            self.results['errors'].append(f"{phase_name}: {details}")
    
    def create_comprehensive_test_data(self):
        """Create comprehensive test dataset simulating real ColO-RAN data."""
        self.logger.info("📊 Creating comprehensive test dataset...")
        
        try:
            # Set seed for reproducible test data
            np.random.seed(42)
            
            # Create realistic ColO-RAN dataset with various scenarios
            n_samples = 10000
            
            # Time-based features
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
            
            # Network configuration features
            base_stations = [1, 8, 15, 22, 29, 36, 43]
            scheduling_policies = ['sched0', 'sched1', 'sched2']
            training_configs = [f'tr{i}' for i in range(10)]
            
            # Create dataset with realistic distributions
            data = pd.DataFrame({
                # Temporal features
                'Timestamp': [int(ts.timestamp() * 1000) for ts in timestamps],
                
                # Network configuration
                'sched_policy': np.random.choice(scheduling_policies, n_samples),
                'training_config': np.random.choice(training_configs, n_samples),
                'bs_id': np.random.choice(base_stations, n_samples),
                'exp_id': np.random.choice([1, 2, 3, 4, 5], n_samples),
                'imsi': [f'user_{i%1000}' for i in range(n_samples)],
                
                # Traffic and performance metrics (realistic distributions)
                'tx_brate downlink [Mbps]': np.random.lognormal(2.5, 0.5, n_samples),  # Log-normal distribution
                'sum_requested_prbs': np.random.poisson(12, n_samples),  # Poisson distribution
                'sum_granted_prbs': np.random.poisson(10, n_samples),   # Usually less than requested
                
                # Error rates (beta distribution for realistic error patterns)
                'tx_errors downlink (%)': np.random.beta(1, 20, n_samples) * 10,  # Low error rates
                'rx_errors uplink (%)': np.random.beta(1, 15, n_samples) * 8,     # Slightly higher
                
                # Channel quality (normal distribution around good quality)
                'dl_cqi': np.random.normal(11, 2, n_samples).clip(1, 15),
                
                # User equipment count (Poisson distribution)
                'num_ues': np.random.poisson(6, n_samples).clip(1, 42)
            })
            
            # Add some realistic missing values (5% missing rate)
            missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
            for col in ['tx_errors downlink (%)', 'rx_errors uplink (%)', 'dl_cqi', 'num_ues']:
                missing_col_indices = np.random.choice(missing_indices, size=len(missing_indices)//4, replace=False)
                data.loc[missing_col_indices, col] = np.nan
            
            # Add some network load scenarios
            # Peak hours (higher traffic)
            peak_hours = (timestamps.hour >= 8) & (timestamps.hour <= 22)
            data.loc[peak_hours, 'sum_requested_prbs'] *= 1.5
            data.loc[peak_hours, 'num_ues'] *= 1.3
            
            # Weekend patterns (different usage)
            weekend = timestamps.weekday >= 5
            data.loc[weekend, 'tx_brate downlink [Mbps]'] *= 1.2
            
            self.logger.info(f"✅ Created test dataset: {len(data):,} records, {len(data.columns)} features")
            self.logger.info(f"   Missing values: {data.isnull().sum().sum():,} ({data.isnull().sum().sum()/(len(data)*len(data.columns))*100:.1f}%)")
            
            self.log_phase("Test Data Creation", True, f"Created {len(data):,} records with realistic distributions")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Test data creation failed: {e}")
            self.log_phase("Test Data Creation", False, str(e))
            raise
    
    def test_data_processing_pipeline(self, raw_data):
        """Test the complete data processing pipeline."""
        self.logger.info("🔄 Testing data processing pipeline...")
        
        try:
            # Initialize data processor
            slice_configs = {f'tr{i}': [5, 6, 6] for i in range(10)}  # 17 RBGs total
            processor = MemoryOptimizedProcessor(slice_configs, batch_size=self.config_manager.get('data.batch_size', 1000))
            
            # Test data processing
            processed_data = processor.process_data(raw_data)
            
            if processed_data is None or len(processed_data) == 0:
                raise ValueError("Data processing returned empty result")
            
            # Validate processed data
            required_features = [
                'slice_id', 'sched_policy_num', 'allocated_rbgs',
                'sum_requested_prbs', 'sum_granted_prbs',
                'prb_utilization', 'throughput_efficiency', 'qos_score',
                'network_load', 'hour', 'minute', 'day_of_week',
                'allocation_efficiency'
            ]
            
            missing_features = [f for f in required_features if f not in processed_data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Test data quality validation
            quality_report = processor.validate_data_quality(processed_data)
            self.results['data_quality'] = quality_report
            
            # Test deterministic processing
            processed_data_2 = processor.process_data(raw_data.copy())
            consistency_check = processed_data['allocation_efficiency'].equals(processed_data_2['allocation_efficiency'])
            
            if not consistency_check:
                raise ValueError("Data processing is not deterministic")
            
            # Test outlier cleaning
            cleaned_data = processor.clean_outliers(processed_data)
            
            self.logger.info(f"✅ Data processing completed successfully:")
            self.logger.info(f"   Input records: {len(raw_data):,}")
            self.logger.info(f"   Processed records: {len(processed_data):,}")
            self.logger.info(f"   Features created: {len(processed_data.columns)}")
            self.logger.info(f"   Data quality score: {quality_report['quality_score']:.1f}/100")
            self.logger.info(f"   Deterministic processing: ✅")
            
            self.log_phase("Data Processing Pipeline", True, 
                         f"Processed {len(processed_data):,} records, quality score: {quality_report['quality_score']:.1f}/100")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Data processing failed: {e}")
            self.log_phase("Data Processing Pipeline", False, str(e))
            raise
    
    def test_ml_training_pipeline(self, processed_data):
        """Test the complete ML training pipeline."""
        self.logger.info("🧠 Testing ML training pipeline...")
        
        if not SKLEARN_AVAILABLE:
            self.log_phase("ML Training Pipeline", False, "Scikit-learn not available")
            return None
        
        try:
            # Prepare training data
            features = [
                'slice_id', 'sched_policy_num', 'allocated_rbgs',
                'sum_requested_prbs', 'sum_granted_prbs',
                'prb_utilization', 'throughput_efficiency', 'qos_score',
                'network_load', 'hour', 'minute', 'day_of_week'
            ]
            target = 'allocation_efficiency'
            
            # Validate features exist
            missing_features = [f for f in features if f not in processed_data.columns]
            if missing_features:
                raise ValueError(f"Missing features for training: {missing_features}")
            
            # Prepare data
            X = processed_data[features].fillna(0).astype(np.float32)
            y = processed_data[target].fillna(0).astype(np.float32)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.logger.info(f"   Training data prepared: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
            
            # Train Random Forest model
            self.logger.info("   Training Random Forest model...")
            rf_model = RandomForestRegressor(
                n_estimators=self.config_manager.get('model.rf_n_estimators', 20),
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            rf_predictions = rf_model.predict(X_test_scaled)
            
            # Calculate metrics
            rf_metrics = {
                'r2': r2_score(y_test, rf_predictions),
                'mae': mean_absolute_error(y_test, rf_predictions),
                'mse': mean_squared_error(y_test, rf_predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, rf_predictions))
            }
            
            self.logger.info(f"   Random Forest Results:")
            self.logger.info(f"     R² Score: {rf_metrics['r2']:.4f}")
            self.logger.info(f"     MAE: {rf_metrics['mae']:.4f}")
            self.logger.info(f"     RMSE: {rf_metrics['rmse']:.4f}")
            
            # Feature importance analysis
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"   Top 5 Important Features:")
            for idx, row in feature_importance.head().iterrows():
                self.logger.info(f"     {row['feature']}: {row['importance']:.4f}")
            
            # Save model and results
            model_save_dir = self.test_dir / "models"
            model_save_dir.mkdir(exist_ok=True)
            
            import joblib
            model_path = model_save_dir / "test_rf_model.pkl"
            scaler_path = model_save_dir / "test_scaler.pkl"
            
            joblib.dump(rf_model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Store results
            self.results['models_trained'].append('RandomForest')
            self.results['performance_metrics']['RandomForest'] = rf_metrics
            
            # Test model loading
            loaded_model = joblib.load(model_path)
            loaded_scaler = joblib.load(scaler_path)
            
            # Verify loaded model works
            test_predictions = loaded_model.predict(loaded_scaler.transform(X_test))
            if not np.allclose(rf_predictions, test_predictions):
                raise ValueError("Model loading verification failed")
            
            self.logger.info(f"✅ ML training pipeline completed successfully")
            self.logger.info(f"   Model saved to: {model_path}")
            self.logger.info(f"   Model loading verification: ✅")
            
            self.log_phase("ML Training Pipeline", True, 
                         f"RF model trained, R²: {rf_metrics['r2']:.4f}, MAE: {rf_metrics['mae']:.4f}")
            
            return {
                'model': rf_model,
                'scaler': scaler,
                'metrics': rf_metrics,
                'feature_importance': feature_importance,
                'model_path': model_path,
                'scaler_path': scaler_path
            }
            
        except Exception as e:
            self.logger.error(f"❌ ML training pipeline failed: {e}")
            self.log_phase("ML Training Pipeline", False, str(e))
            raise
    
    def test_resource_allocation_optimization(self, ml_results):
        """Test resource allocation optimization system."""
        self.logger.info("⚡ Testing resource allocation optimization...")
        
        try:
            # Import allocator
            from optimization.allocator import SliceResourceAllocator
            
            # Initialize allocator with trained model
            if ml_results:
                allocator = SliceResourceAllocator(
                    model_path=str(ml_results['model_path']),
                    scaler_path=str(ml_results['scaler_path']),
                    features=ml_results['feature_importance']['feature'].tolist(),
                    config_manager=self.config_manager
                )
            else:
                # Use fallback mode
                allocator = SliceResourceAllocator(config_manager=self.config_manager)
            
            # Test different network scenarios
            test_scenarios = [
                {
                    'name': 'Low Load',
                    'state': {
                        'num_ues': 5, 'sched_policy_num': 0, 'bs_id': 1, 'exp_id': 1,
                        'sum_requested_prbs': 8, 'hour': 10, 'minute': 30, 'day_of_week': 2,
                        'qos_score': 0.9, 'network_load': 0.3, 'throughput_efficiency': 0.8
                    }
                },
                {
                    'name': 'Medium Load',
                    'state': {
                        'num_ues': 15, 'sched_policy_num': 1, 'bs_id': 8, 'exp_id': 2,
                        'sum_requested_prbs': 14, 'hour': 14, 'minute': 15, 'day_of_week': 3,
                        'qos_score': 0.75, 'network_load': 0.6, 'throughput_efficiency': 0.65
                    }
                },
                {
                    'name': 'High Load',
                    'state': {
                        'num_ues': 25, 'sched_policy_num': 2, 'bs_id': 15, 'exp_id': 3,
                        'sum_requested_prbs': 18, 'hour': 18, 'minute': 45, 'day_of_week': 5,
                        'qos_score': 0.6, 'network_load': 0.85, 'throughput_efficiency': 0.5
                    }
                }
            ]
            
            optimization_results = []
            
            for scenario in test_scenarios:
                self.logger.info(f"   Testing {scenario['name']} scenario...")
                
                # Test exhaustive search
                best_exhaustive, efficiency_exhaustive = allocator.optimize_exhaustive(scenario['state'])
                
                # Test genetic algorithm
                best_genetic, efficiency_genetic = allocator.optimize_genetic(scenario['state'])
                
                scenario_result = {
                    'scenario': scenario['name'],
                    'exhaustive_allocation': best_exhaustive,
                    'exhaustive_efficiency': efficiency_exhaustive,
                    'genetic_allocation': best_genetic,
                    'genetic_efficiency': efficiency_genetic,
                    'rbg_total_check': sum(best_exhaustive) == allocator.total_rbgs
                }
                
                optimization_results.append(scenario_result)
                
                self.logger.info(f"     Exhaustive: {best_exhaustive} (efficiency: {efficiency_exhaustive:.4f})")
                self.logger.info(f"     Genetic: {best_genetic} (efficiency: {efficiency_genetic:.4f})")
                self.logger.info(f"     RBG total check: {'✅' if scenario_result['rbg_total_check'] else '❌'}")
            
            # Test simulation with deterministic scenarios
            self.logger.info("   Testing scenario-based simulation...")
            simulation_results = allocator.simulate(steps=20, method="genetic", random_seed=42)
            
            if simulation_results is None or len(simulation_results) == 0:
                raise ValueError("Simulation returned empty results")
            
            # Test reproducibility
            simulation_results_2 = allocator.simulate(steps=20, method="genetic", random_seed=42)
            reproducible = simulation_results['improvement'].equals(simulation_results_2['improvement'])
            
            if not reproducible:
                raise ValueError("Simulation is not reproducible")
            
            avg_improvement = simulation_results['improvement'].mean()
            
            self.logger.info(f"✅ Resource allocation optimization completed successfully")
            self.logger.info(f"   Scenarios tested: {len(test_scenarios)}")
            self.logger.info(f"   Simulation steps: {len(simulation_results)}")
            self.logger.info(f"   Average improvement: {avg_improvement:.4f}")
            self.logger.info(f"   Reproducibility check: ✅")
            
            self.log_phase("Resource Allocation Optimization", True,
                         f"Tested {len(test_scenarios)} scenarios, avg improvement: {avg_improvement:.4f}")
            
            return {
                'scenarios': optimization_results,
                'simulation': simulation_results,
                'average_improvement': avg_improvement,
                'reproducible': reproducible
            }
            
        except Exception as e:
            self.logger.error(f"❌ Resource allocation optimization failed: {e}")
            self.log_phase("Resource Allocation Optimization", False, str(e))
            raise
    
    def test_end_to_end_integration(self):
        """Test complete end-to-end integration."""
        self.logger.info("🔗 Testing end-to-end integration...")
        
        try:
            # Test configuration system
            training_config = self.config_manager.get_training_config()
            data_config = self.config_manager.get_data_config()
            
            self.logger.info(f"   Configuration system: ✅")
            self.logger.info(f"   Training config keys: {list(training_config.keys())}")
            self.logger.info(f"   Data config keys: {list(data_config.keys())}")
            
            # Test data pipeline reusability
            slice_configs = {f'tr{i}': [5, 6, 6] for i in range(5)}
            processor = MemoryOptimizedProcessor(slice_configs, batch_size=500)
            
            # Create small test dataset
            small_test_data = self.create_minimal_test_data()
            processed_small = processor.process_data(small_test_data)
            
            if processed_small is None or len(processed_small) == 0:
                raise ValueError("Small dataset processing failed")
            
            # Test memory optimization
            original_memory = processed_small.memory_usage(deep=True).sum()
            optimized_data = processor.optimize_datatypes(processed_small)
            optimized_memory = optimized_data.memory_usage(deep=True).sum()
            
            memory_savings = (original_memory - optimized_memory) / original_memory * 100
            
            self.logger.info(f"✅ End-to-end integration completed successfully")
            self.logger.info(f"   Small dataset processing: ✅")
            self.logger.info(f"   Memory optimization: {memory_savings:.1f}% savings")
            self.logger.info(f"   Configuration integration: ✅")
            
            self.log_phase("End-to-End Integration", True,
                         f"Integration successful, {memory_savings:.1f}% memory savings")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ End-to-end integration failed: {e}")
            self.log_phase("End-to-End Integration", False, str(e))
            raise
    
    def create_minimal_test_data(self):
        """Create minimal test data for quick integration testing."""
        n_samples = 100
        return pd.DataFrame({
            'Timestamp': range(1000000, 1000000 + n_samples),
            'sched_policy': ['sched0'] * n_samples,
            'training_config': ['tr0'] * n_samples,
            'bs_id': [1] * n_samples,
            'exp_id': [1] * n_samples,
            'imsi': [f'user_{i}' for i in range(n_samples)],
            'tx_brate downlink [Mbps]': np.random.uniform(10, 20, n_samples),
            'sum_requested_prbs': np.random.randint(5, 15, n_samples),
            'sum_granted_prbs': np.random.randint(5, 15, n_samples),
            'tx_errors downlink (%)': np.random.uniform(0, 2, n_samples),
            'rx_errors uplink (%)': np.random.uniform(0, 2, n_samples),
            'dl_cqi': np.random.randint(8, 12, n_samples),
            'num_ues': np.random.randint(3, 8, n_samples)
        })
    
    def generate_training_test_report(self):
        """Generate comprehensive training test report."""
        self.logger.info("📋 Generating training test report...")
        
        # Calculate summary statistics
        total_phases = len(self.results['test_phases'])
        successful_phases = len([p for p in self.results['test_phases'] if p['success']])
        success_rate = (successful_phases / total_phases * 100) if total_phases > 0 else 0
        
        # Create report
        report = {
            'test_summary': {
                'timestamp': self.results['timestamp'],
                'total_phases': total_phases,
                'successful_phases': successful_phases,
                'success_rate_percent': round(success_rate, 2),
                'overall_status': 'PASS' if success_rate >= 90 and len(self.results['errors']) == 0 else 'FAIL'
            },
            'test_phases': self.results['test_phases'],
            'models_trained': self.results['models_trained'],
            'performance_metrics': self.results['performance_metrics'],
            'data_quality': self.results['data_quality'],
            'errors': self.results['errors']
        }
        
        # Save detailed report
        report_file = self.test_dir / f"training_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎯 COLORAN DYNAMIC SLICE OPTIMIZER - COMPLETE TRAINING TEST REPORT")
        print("=" * 80)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Test Phases: {successful_phases}/{total_phases} successful ({success_rate:.1f}%)")
        print(f"🏆 Overall Status: {report['test_summary']['overall_status']}")
        
        if self.results['models_trained']:
            print(f"\n🤖 MODELS TRAINED:")
            for model in self.results['models_trained']:
                print(f"   ✓ {model}")
                if model in self.results['performance_metrics']:
                    metrics = self.results['performance_metrics'][model]
                    print(f"     R²: {metrics.get('r2', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}")
        
        if self.results['data_quality']:
            print(f"\n📈 DATA QUALITY:")
            quality = self.results['data_quality']
            print(f"   Quality Score: {quality.get('quality_score', 0):.1f}/100")
            print(f"   Total Records: {quality.get('total_records', 0):,}")
            print(f"   Duplicates: {quality.get('duplicates', 0):,}")
        
        print(f"\n🔍 TEST PHASES DETAILS:")
        for phase in self.results['test_phases']:
            status = "✅ PASS" if phase['success'] else "❌ FAIL"
            print(f"   {status} {phase['phase']}")
            if phase['details']:
                print(f"       {phase['details']}")
        
        if self.results['errors']:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        print(f"\n📁 Detailed report saved to: {report_file}")
        
        return report
    
    def run_complete_training_test(self):
        """Run the complete training test pipeline."""
        self.logger.info("🚀 Starting Complete Training Test for ColO-RAN Dynamic Slice Optimizer")
        self.logger.info("=" * 80)
        
        try:
            # Phase 1: Create test data
            raw_data = self.create_comprehensive_test_data()
            
            # Phase 2: Test data processing
            processed_data = self.test_data_processing_pipeline(raw_data)
            
            # Phase 3: Test ML training
            ml_results = self.test_ml_training_pipeline(processed_data)
            
            # Phase 4: Test resource allocation
            optimization_results = self.test_resource_allocation_optimization(ml_results)
            
            # Phase 5: Test end-to-end integration
            integration_success = self.test_end_to_end_integration()
            
            # Generate final report
            report = self.generate_training_test_report()
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Complete training test failed: {e}")
            
            # Generate report even if test failed
            report = self.generate_training_test_report()
            
            return report


def main():
    """Main training test entry point."""
    print("🔬 ColO-RAN Dynamic Slice Optimizer - Complete Training Test")
    print("=" * 70)
    print("This test demonstrates the entire ML pipeline from data to deployment")
    print("")
    
    try:
        # Run complete training test
        test = CompleteTrainingTest()
        report = test.run_complete_training_test()
        
        # Determine success
        success_rate = report['test_summary']['success_rate_percent']
        overall_status = report['test_summary']['overall_status']
        
        if overall_status == 'PASS':
            print(f"\n🎉 TRAINING TEST SUCCESSFUL!")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   All critical components validated")
            return 0
        else:
            print(f"\n⚠️ TRAINING TEST COMPLETED WITH ISSUES")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Check the detailed report for issues")
            return 1
            
    except Exception as e:
        print(f"\n❌ TRAINING TEST FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)