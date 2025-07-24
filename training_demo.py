#!/usr/bin/env python3
# training_demo.py

"""
Complete Training Demonstration for ColO-RAN Dynamic Slice Optimizer
This script demonstrates the entire ML pipeline from data loading to model deployment.
"""

import os
import sys
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
    print("PASS: Core dependencies (numpy, pandas) loaded successfully")
except ImportError as e:
    print(f"FAIL: Failed to import core dependencies: {e}")
    print("Please install: pip install numpy pandas")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    print("PASS: Scikit-learn loaded successfully")
    SKLEARN_AVAILABLE = True
except ImportError:
    print("SKIP: Scikit-learn not available - will demonstrate with basic implementation")
    SKLEARN_AVAILABLE = False

# Import project modules
try:
    from data.data_loader import ColoRANDataLoader
    from data.data_processor import MemoryOptimizedProcessor
    print("PASS: Data modules loaded successfully")
except ImportError as e:
    print(f"FAIL: Failed to import data modules: {e}")
    sys.exit(1)

try:
    from coloran_optimizer.config import ConfigurationManager, init_config
    print("PASS: Configuration system loaded successfully")
except ImportError as e:
    print(f"FAIL: Failed to import configuration system: {e}")
    sys.exit(1)


class TrainingDemo:
    """Complete training demonstration for ColO-RAN Dynamic Slice Optimizer."""
    
    def __init__(self):
        self.demo_dir = Path(__file__).parent / "training_demo_results"
        self.demo_dir.mkdir(exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'phases_completed': [],
            'performance_metrics': {},
            'models_saved': []
        }
        
        print("\nStarting ColO-RAN Dynamic Slice Optimizer Training Demo")
        print("=" * 60)
    
    def setup_configuration(self):
        """Setup configuration for training demo."""
        print("\n[PHASE 1] Setting up configuration...")
        
        try:
            # Create demo configuration
            config_dir = self.demo_dir / "config"
            self.config_manager = init_config(str(config_dir), environment='development', create_defaults=True)
            
            # Override for demo
            self.config_manager.set('training.use_gpu', False)  # Use CPU for demo
            self.config_manager.set('training.sample_size', 2000)  # Small sample
            self.config_manager.set('training.epochs', 3)  # Quick training
            self.config_manager.set('data.batch_size', 500)  # Small batches
            self.config_manager.set('model.rf_n_estimators', 10)  # Fast training
            
            print("PASS: Configuration system setup completed")
            print(f"      Config files created in: {config_dir}")
            
            self.results['phases_completed'].append('Configuration Setup')
            return True
            
        except Exception as e:
            print(f"FAIL: Configuration setup failed: {e}")
            return False
    
    def create_demo_data(self):
        """Create demonstration dataset simulating real ColO-RAN data."""
        print("\n[PHASE 2] Creating demonstration dataset...")
        
        try:
            # Set seed for reproducible demo data
            np.random.seed(42)
            
            # Create realistic ColO-RAN dataset
            n_samples = 5000
            
            # Time-based features
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
            
            # Create dataset with realistic distributions
            data = pd.DataFrame({
                # Temporal features
                'Timestamp': [int(ts.timestamp() * 1000) for ts in timestamps],
                
                # Network configuration
                'sched_policy': np.random.choice(['sched0', 'sched1', 'sched2'], n_samples),
                'training_config': np.random.choice([f'tr{i}' for i in range(5)], n_samples),
                'bs_id': np.random.choice([1, 8, 15, 22], n_samples),
                'exp_id': np.random.choice([1, 2, 3], n_samples),
                'imsi': [f'user_{i%500}' for i in range(n_samples)],
                
                # Performance metrics with realistic distributions
                'tx_brate downlink [Mbps]': np.random.lognormal(2.3, 0.4, n_samples),
                'sum_requested_prbs': np.random.poisson(10, n_samples),
                'sum_granted_prbs': np.random.poisson(8, n_samples),
                
                # Error rates (low and realistic)
                'tx_errors downlink (%)': np.random.exponential(1.5, n_samples),
                'rx_errors uplink (%)': np.random.exponential(1.2, n_samples),
                
                # Channel quality
                'dl_cqi': np.random.normal(10, 2, n_samples).clip(1, 15),
                
                # User equipment count
                'num_ues': np.random.poisson(5, n_samples).clip(1, 30)
            })
            
            # Add some missing values to test data processing
            missing_indices = np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)
            for col in ['tx_errors downlink (%)', 'rx_errors uplink (%)', 'dl_cqi']:
                col_missing = np.random.choice(missing_indices, size=len(missing_indices)//3, replace=False)
                data.loc[col_missing, col] = np.nan
            
            print(f"PASS: Demo dataset created successfully")
            print(f"      Records: {len(data):,}")
            print(f"      Features: {len(data.columns)}")
            print(f"      Missing values: {data.isnull().sum().sum():,}")
            
            self.results['phases_completed'].append('Demo Data Creation')
            return data
            
        except Exception as e:
            print(f"FAIL: Demo data creation failed: {e}")
            return None
    
    def demonstrate_data_processing(self, raw_data):
        """Demonstrate the data processing pipeline."""
        print("\n[PHASE 3] Demonstrating data processing pipeline...")
        
        try:
            # Initialize data processor
            slice_configs = {f'tr{i}': [5, 6, 6] for i in range(5)}  # 17 RBGs total
            processor = MemoryOptimizedProcessor(slice_configs, batch_size=500)
            
            # Process data
            print("         Processing raw data...")
            processed_data = processor.process_data(raw_data)
            
            if processed_data is None or len(processed_data) == 0:
                raise ValueError("Data processing returned empty result")
            
            # Validate processed data
            required_features = [
                'slice_id', 'sched_policy_num', 'allocated_rbgs',
                'prb_utilization', 'throughput_efficiency', 'qos_score',
                'allocation_efficiency'
            ]
            
            missing_features = [f for f in required_features if f not in processed_data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Test data quality validation
            print("         Validating data quality...")
            quality_report = processor.validate_data_quality(processed_data)
            
            # Test deterministic processing
            print("         Testing deterministic processing...")
            processed_data_2 = processor.process_data(raw_data.copy())
            consistency_check = processed_data['allocation_efficiency'].equals(processed_data_2['allocation_efficiency'])
            
            if not consistency_check:
                raise ValueError("Data processing is not deterministic")
            
            print(f"PASS: Data processing pipeline completed successfully")
            print(f"      Input records: {len(raw_data):,}")
            print(f"      Output records: {len(processed_data):,}")
            print(f"      Features created: {len(processed_data.columns)}")
            print(f"      Data quality score: {quality_report['quality_score']:.1f}/100")
            print(f"      Deterministic processing: VERIFIED")
            
            self.results['phases_completed'].append('Data Processing')
            return processed_data
            
        except Exception as e:
            print(f"FAIL: Data processing failed: {e}")
            return None
    
    def demonstrate_ml_training(self, processed_data):
        """Demonstrate ML model training."""
        print("\n[PHASE 4] Demonstrating ML model training...")
        
        if not SKLEARN_AVAILABLE:
            print("SKIP: Scikit-learn not available")
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
            
            # Prepare data
            X = processed_data[features].fillna(0).astype(np.float32)
            y = processed_data[target].fillna(0).astype(np.float32)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            print("         Scaling features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            print(f"         Training data: {X_train.shape[0]:,} samples")
            print(f"         Test data: {X_test.shape[0]:,} samples")
            
            # Train Random Forest model
            print("         Training Random Forest model...")
            rf_model = RandomForestRegressor(
                n_estimators=10,  # Small for demo
                max_depth=8,
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
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"PASS: ML model training completed successfully")
            print(f"      Model type: Random Forest")
            print(f"      R2 Score: {rf_metrics['r2']:.4f}")
            print(f"      MAE: {rf_metrics['mae']:.4f}")
            print(f"      RMSE: {rf_metrics['rmse']:.4f}")
            print(f"      Top feature: {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.4f})")
            
            # Save model
            import joblib
            model_path = self.demo_dir / "demo_rf_model.pkl"
            scaler_path = self.demo_dir / "demo_scaler.pkl"
            
            joblib.dump(rf_model, model_path)
            joblib.dump(scaler, scaler_path)
            
            print(f"      Model saved to: {model_path}")
            
            self.results['phases_completed'].append('ML Training')
            self.results['performance_metrics'] = rf_metrics
            self.results['models_saved'].append(str(model_path))
            
            return {
                'model': rf_model,
                'scaler': scaler,
                'metrics': rf_metrics,
                'feature_importance': feature_importance,
                'model_path': model_path,
                'scaler_path': scaler_path
            }
            
        except Exception as e:
            print(f"FAIL: ML training failed: {e}")
            return None
    
    def demonstrate_resource_allocation(self, ml_results):
        """Demonstrate resource allocation optimization."""
        print("\n[PHASE 5] Demonstrating resource allocation optimization...")
        
        try:
            # Import allocator
            from optimization.allocator import SliceResourceAllocator
            
            # Initialize allocator
            if ml_results:
                allocator = SliceResourceAllocator(
                    model_path=str(ml_results['model_path']),
                    scaler_path=str(ml_results['scaler_path']),
                    features=ml_results['feature_importance']['feature'].tolist(),
                    config_manager=self.config_manager
                )
                print("         Using trained ML model for optimization")
            else:
                # Use fallback mode
                allocator = SliceResourceAllocator(config_manager=self.config_manager)
                print("         Using fallback heuristic optimization")
            
            # Test network scenarios
            test_scenarios = [
                {
                    'name': 'Peak Hours',
                    'state': {
                        'num_ues': 20, 'sched_policy_num': 1, 'bs_id': 8, 'exp_id': 2,
                        'sum_requested_prbs': 16, 'hour': 18, 'minute': 30, 'day_of_week': 3,
                        'qos_score': 0.65, 'network_load': 0.8, 'throughput_efficiency': 0.55
                    }
                },
                {
                    'name': 'Off-Peak',
                    'state': {
                        'num_ues': 8, 'sched_policy_num': 0, 'bs_id': 1, 'exp_id': 1,
                        'sum_requested_prbs': 6, 'hour': 3, 'minute': 15, 'day_of_week': 2,
                        'qos_score': 0.9, 'network_load': 0.25, 'throughput_efficiency': 0.85
                    }
                }
            ]
            
            print("         Testing optimization scenarios...")
            for scenario in test_scenarios:
                print(f"           {scenario['name']} scenario:")
                
                # Test exhaustive search
                best_exhaustive, efficiency_exhaustive = allocator.optimize_exhaustive(scenario['state'])
                print(f"             Exhaustive: {best_exhaustive} (efficiency: {efficiency_exhaustive:.4f})")
                
                # Test genetic algorithm
                best_genetic, efficiency_genetic = allocator.optimize_genetic(scenario['state'])
                print(f"             Genetic: {best_genetic} (efficiency: {efficiency_genetic:.4f})")
                
                # Verify RBG total
                if sum(best_exhaustive) != allocator.total_rbgs:
                    raise ValueError(f"RBG total mismatch: {sum(best_exhaustive)} != {allocator.total_rbgs}")
            
            # Test simulation
            print("         Testing scenario-based simulation...")
            simulation_results = allocator.simulate(steps=10, method="genetic", random_seed=42)
            
            if simulation_results is None or len(simulation_results) == 0:
                raise ValueError("Simulation returned empty results")
            
            # Test reproducibility
            simulation_results_2 = allocator.simulate(steps=10, method="genetic", random_seed=42)
            reproducible = simulation_results['improvement'].equals(simulation_results_2['improvement'])
            
            if not reproducible:
                raise ValueError("Simulation is not reproducible")
            
            avg_improvement = simulation_results['improvement'].mean()
            
            print(f"PASS: Resource allocation optimization completed successfully")
            print(f"      Scenarios tested: {len(test_scenarios)}")
            print(f"      Simulation steps: {len(simulation_results)}")
            print(f"      Average improvement: {avg_improvement:.4f}")
            print(f"      Reproducibility: VERIFIED")
            
            self.results['phases_completed'].append('Resource Allocation')
            return True
            
        except Exception as e:
            print(f"FAIL: Resource allocation failed: {e}")
            return False
    
    def demonstrate_configuration_flexibility(self):
        """Demonstrate configuration system flexibility."""
        print("\n[PHASE 6] Demonstrating configuration flexibility...")
        
        try:
            # Test different environment configurations
            print("         Testing environment-specific configurations...")
            
            # Test development config
            dev_config = self.config_manager.get_training_config()
            print(f"           Development - GPU: {dev_config.get('use_gpu')}, Epochs: {dev_config.get('epochs')}")
            
            # Test environment variable override
            original_gpu = self.config_manager.get('training.use_gpu')
            self.config_manager.set('training.use_gpu', True)
            updated_gpu = self.config_manager.get('training.use_gpu')
            
            if updated_gpu != True:
                raise ValueError("Configuration override failed")
            
            # Reset
            self.config_manager.set('training.use_gpu', original_gpu)
            
            # Test configuration validation
            data_config = self.config_manager.get_data_config()
            model_config = self.config_manager.get_model_config()
            
            required_sections = ['data', 'training', 'model']
            available_sections = []
            
            if data_config:
                available_sections.append('data')
            if dev_config:
                available_sections.append('training')
            if model_config:
                available_sections.append('model')
            
            if len(available_sections) != len(required_sections):
                raise ValueError(f"Missing configuration sections: {set(required_sections) - set(available_sections)}")
            
            print(f"PASS: Configuration flexibility demonstrated successfully")
            print(f"      Configuration sections: {available_sections}")
            print(f"      Environment override: VERIFIED")
            print(f"      Configuration validation: VERIFIED")
            
            self.results['phases_completed'].append('Configuration Flexibility')
            return True
            
        except Exception as e:
            print(f"FAIL: Configuration flexibility test failed: {e}")
            return False
    
    def generate_demo_report(self):
        """Generate demonstration report."""
        print("\n[FINAL] Generating demonstration report...")
        
        # Calculate summary
        total_phases = 6
        completed_phases = len(self.results['phases_completed'])
        success_rate = (completed_phases / total_phases) * 100
        
        # Create report
        report = {
            'demo_summary': {
                'timestamp': self.results['timestamp'],
                'total_phases': total_phases,
                'completed_phases': completed_phases,
                'success_rate_percent': round(success_rate, 2),
                'overall_status': 'SUCCESS' if success_rate >= 90 else 'PARTIAL'
            },
            'phases_completed': self.results['phases_completed'],
            'performance_metrics': self.results['performance_metrics'],
            'models_saved': self.results['models_saved']
        }
        
        # Save report
        report_file = self.demo_dir / f"training_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 70)
        print("COLORAN DYNAMIC SLICE OPTIMIZER - TRAINING DEMONSTRATION REPORT")
        print("=" * 70)
        print(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Phases Completed: {completed_phases}/{total_phases} ({success_rate:.1f}%)")
        print(f"Overall Status: {report['demo_summary']['overall_status']}")
        
        if self.results['performance_metrics']:
            metrics = self.results['performance_metrics']
            print(f"\nML Model Performance:")
            print(f"  R2 Score: {metrics.get('r2', 0):.4f}")
            print(f"  Mean Absolute Error: {metrics.get('mae', 0):.4f}")
            print(f"  Root Mean Square Error: {metrics.get('rmse', 0):.4f}")
        
        print(f"\nPhases Completed:")
        for i, phase in enumerate(self.results['phases_completed'], 1):
            print(f"  {i}. {phase}")
        
        if self.results['models_saved']:
            print(f"\nModels Saved:")
            for model in self.results['models_saved']:
                print(f"  - {model}")
        
        print(f"\nDetailed report saved to: {report_file}")
        
        return report
    
    def run_complete_demo(self):
        """Run the complete training demonstration."""
        print("Starting Complete Training Demonstration...")
        
        try:
            # Phase 1: Setup configuration
            if not self.setup_configuration():
                return False
            
            # Phase 2: Create demo data
            raw_data = self.create_demo_data()
            if raw_data is None:
                return False
            
            # Phase 3: Demonstrate data processing
            processed_data = self.demonstrate_data_processing(raw_data)
            if processed_data is None:
                return False
            
            # Phase 4: Demonstrate ML training
            ml_results = self.demonstrate_ml_training(processed_data)
            
            # Phase 5: Demonstrate resource allocation
            if not self.demonstrate_resource_allocation(ml_results):
                return False
            
            # Phase 6: Demonstrate configuration flexibility
            if not self.demonstrate_configuration_flexibility():
                return False
            
            # Generate final report
            report = self.generate_demo_report()
            
            success_rate = report['demo_summary']['success_rate_percent']
            return success_rate >= 90
            
        except Exception as e:
            print(f"\nDEMO FAILED: {e}")
            return False


def main():
    """Main demonstration entry point."""
    print("ColO-RAN Dynamic Slice Optimizer - Complete Training Demonstration")
    print("=" * 70)
    print("This demonstration shows the entire ML pipeline in action")
    print("")
    
    try:
        # Run complete demo
        demo = TrainingDemo()
        success = demo.run_complete_demo()
        
        if success:
            print(f"\nTRAINING DEMONSTRATION SUCCESSFUL!")
            print(f"All critical components validated and working correctly")
            return 0
        else:
            print(f"\nTRAINING DEMONSTRATION COMPLETED WITH ISSUES")
            print(f"Check the detailed report for any problems")
            return 1
            
    except Exception as e:
        print(f"\nTRAINING DEMONSTRATION FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)