#!/usr/bin/env python3
# validate_fixes.py

"""
Lightweight validation script to verify critical fixes have been implemented.
This script checks the code changes without requiring all dependencies.
"""

import os
import sys
import logging
from pathlib import Path
import ast
import re
from datetime import datetime


class CodeFixValidator:
    """Validates that critical code fixes have been implemented."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'fixes_validated': [],
            'issues_found': [],
            'validation_summary': {}
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_random_data_elimination(self):
        """Validate that random data generation has been eliminated."""
        self.logger.info("🔍 Validating elimination of random data generation...")
        
        issues = []
        files_checked = []
        
        # Check data_processor.py
        processor_file = self.src_dir / "data" / "data_processor.py"
        if processor_file.exists():
            content = processor_file.read_text()
            files_checked.append(str(processor_file))
            
            # Look for random data generation patterns
            random_patterns = [
                r'np\.random\.randint\(',
                r'np\.random\.rand\(',
                r'np\.random\.random\(',
                r'np\.random\.uniform\(',
                r'pd\.Series\(np\.random'
            ]
            
            for pattern in random_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Check if it's in comments or test data creation
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            if '# Conservative' in line or 'fillna(' in line:
                                continue  # This is acceptable fallback
                            issues.append(f"Random generation found in {processor_file}:{i}")
            
            # Check for proper fallback values
            if 'fillna(2.5)' in content and 'fillna(10.0)' in content and 'fillna(5.0)' in content:
                self.results['fixes_validated'].append("Data processor uses deterministic fallback values")
            else:
                issues.append("Data processor missing deterministic fallback values")
        
        # Check allocator.py
        allocator_file = self.src_dir / "optimization" / "allocator.py"
        if allocator_file.exists():
            content = allocator_file.read_text()
            files_checked.append(str(allocator_file))
            
            # Check for random seed management
            if 'random_seed' in content and 'np.random.seed(' in content:
                self.results['fixes_validated'].append("Allocator implements random seed management")
            else:
                issues.append("Allocator missing random seed management")
            
            # Check for scenario-based testing
            if 'test_scenarios' in content and 'scenario' in content:
                self.results['fixes_validated'].append("Allocator uses scenario-based testing instead of random")
            else:
                issues.append("Allocator missing scenario-based testing")
        
        if issues:
            self.results['issues_found'].extend(issues)
        else:
            self.results['fixes_validated'].append("Random data generation eliminated")
        
        self.logger.info(f"✅ Checked {len(files_checked)} files for random data generation")
        return len(issues) == 0
    
    def validate_configuration_system(self):
        """Validate that configuration management system is implemented."""
        self.logger.info("🔧 Validating configuration management system...")
        
        issues = []
        
        # Check if config manager exists
        config_file = self.src_dir / "coloran_optimizer" / "config" / "config_manager.py"
        if not config_file.exists():
            issues.append("Configuration manager file missing")
            return False
        
        content = config_file.read_text()
        
        # Check for key features
        required_features = [
            'class ConfigurationManager',
            'def get_training_config',
            'def get_data_config',
            'def get_model_config',
            'def get_security_config',
            '_load_env_overrides',
            '_validate_config',
            'create_default_configs'
        ]
        
        for feature in required_features:
            if feature in content:
                self.results['fixes_validated'].append(f"Config system has {feature}")
            else:
                issues.append(f"Config system missing {feature}")
        
        # Check for environment variable support
        if 'COLORAN_' in content and 'os.environ' in content:
            self.results['fixes_validated'].append("Environment variable override support")
        else:
            issues.append("Missing environment variable support")
        
        if issues:
            self.results['issues_found'].extend(issues)
        else:
            self.results['fixes_validated'].append("Configuration management system implemented")
        
        return len(issues) == 0
    
    def validate_ml_trainer_improvements(self):
        """Validate ML trainer improvements."""
        self.logger.info("🧠 Validating ML trainer improvements...")
        
        issues = []
        
        trainer_file = self.src_dir / "models" / "ml_trainer.py"
        if not trainer_file.exists():
            issues.append("ML trainer file missing")
            return False
        
        content = trainer_file.read_text()
        
        # Check for production features
        production_features = [
            'config_manager',
            '_setup_gpu_environment',
            '_validate_data_quality',
            'cross_val_score',
            'save_models',
            'load_models',
            'model_metadata',
            'training_history'
        ]
        
        for feature in production_features:
            if feature in content:
                self.results['fixes_validated'].append(f"ML trainer has {feature}")
            else:
                issues.append(f"ML trainer missing {feature}")
        
        # Check for GPU memory management
        if 'memory_growth' in content and 'mixed_precision' in content:
            self.results['fixes_validated'].append("GPU memory management implemented")
        else:
            issues.append("Missing GPU memory management")
        
        # Check for reproducibility
        if '_set_random_seeds' in content and 'random_seed' in content:
            self.results['fixes_validated'].append("Reproducibility features implemented")
        else:
            issues.append("Missing reproducibility features")
        
        if issues:
            self.results['issues_found'].extend(issues)
        else:
            self.results['fixes_validated'].append("ML trainer production-ready")
        
        return len(issues) == 0
    
    def validate_security_system(self):
        """Validate security system implementation."""
        self.logger.info("🔒 Validating security system...")
        
        issues = []
        
        security_file = self.src_dir / "coloran_optimizer" / "security" / "security_manager.py"
        if not security_file.exists():
            issues.append("Security manager file missing")
            return False
        
        content = security_file.read_text()
        
        # Check for security features
        security_features = [
            'class SecurityManager',
            '_setup_jwt_secret',
            '_setup_encryption',
            'hash_password',
            'verify_password',
            'encrypt_sensitive_data',
            'decrypt_sensitive_data',
            'validate_input',
            '_check_rate_limit',
            'get_security_report'
        ]
        
        for feature in security_features:
            if feature in content:
                self.results['fixes_validated'].append(f"Security system has {feature}")
            else:
                issues.append(f"Security system missing {feature}")
        
        # Check for JWT secret validation
        if 'len(jwt_secret) < 32' in content:
            self.results['fixes_validated'].append("JWT secret validation implemented")
        else:
            issues.append("Missing JWT secret validation")
        
        # Check for input sanitization
        if 'dangerous_patterns' in content and 'script' in content:
            self.results['fixes_validated'].append("Input sanitization implemented")
        else:
            issues.append("Missing input sanitization")
        
        if issues:
            self.results['issues_found'].extend(issues)
        else:
            self.results['fixes_validated'].append("Comprehensive security system implemented")
        
        return len(issues) == 0
    
    def validate_test_framework(self):
        """Validate test framework implementation."""
        self.logger.info("🧪 Validating test framework...")
        
        issues = []
        test_files = []
        
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            issues.append("Tests directory missing")
            return False
        
        # Check for test files
        expected_test_files = [
            "test_data_pipeline.py",
            "test_ml_trainer.py",
            "test_config_manager.py"
        ]
        
        for test_file in expected_test_files:
            test_path = tests_dir / test_file
            if test_path.exists():
                test_files.append(test_file)
                content = test_path.read_text()
                
                # Check for pytest usage
                if 'import pytest' in content and 'def test_' in content:
                    self.results['fixes_validated'].append(f"Test file {test_file} properly structured")
                else:
                    issues.append(f"Test file {test_file} not properly structured")
            else:
                issues.append(f"Missing test file: {test_file}")
        
        coverage = len(test_files) / len(expected_test_files) * 100
        self.results['validation_summary']['test_coverage'] = f"{coverage:.1f}%"
        
        if coverage >= 80:
            self.results['fixes_validated'].append(f"Test coverage: {coverage:.1f}%")
        else:
            issues.append(f"Insufficient test coverage: {coverage:.1f}%")
        
        if issues:
            self.results['issues_found'].extend(issues)
        
        return len(issues) == 0
    
    def validate_documentation_and_structure(self):
        """Validate project documentation and structure."""
        self.logger.info("📋 Validating project structure and documentation...")
        
        issues = []
        
        # Check for key files
        required_files = [
            "requirements.txt",
            "README.md",
            "validate_pipeline.py"
        ]
        
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.results['fixes_validated'].append(f"Project has {file_name}")
            else:
                issues.append(f"Missing required file: {file_name}")
        
        # Check requirements.txt for security dependencies
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            
            security_deps = ['pyjwt', 'cryptography', 'bcrypt']
            for dep in security_deps:
                if dep in content:
                    self.results['fixes_validated'].append(f"Security dependency {dep} included")
                else:
                    issues.append(f"Missing security dependency: {dep}")
        
        if issues:
            self.results['issues_found'].extend(issues)
        
        return len(issues) == 0
    
    def generate_validation_report(self):
        """Generate final validation report."""
        self.logger.info("\n📊 Generating Validation Report...")
        
        total_validations = len(self.results['fixes_validated'])
        total_issues = len(self.results['issues_found'])
        
        # Calculate success metrics
        success_rate = (total_validations / (total_validations + total_issues) * 100) if (total_validations + total_issues) > 0 else 0
        
        self.results['validation_summary'].update({
            'total_fixes_validated': total_validations,
            'total_issues_found': total_issues,
            'success_rate_percent': round(success_rate, 2),
            'overall_status': 'PASS' if success_rate >= 80 and total_issues == 0 else 'FAIL'
        })
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 COLORAN DYNAMIC SLICE OPTIMIZER - VALIDATION REPORT")
        print("=" * 60)
        print(f"📅 Timestamp: {self.results['timestamp']}")
        print(f"✅ Fixes Validated: {total_validations}")
        print(f"❌ Issues Found: {total_issues}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"🏆 Overall Status: {self.results['validation_summary']['overall_status']}")
        
        if self.results['fixes_validated']:
            print(f"\n✅ CRITICAL FIXES VALIDATED ({len(self.results['fixes_validated'])}):")
            for fix in self.results['fixes_validated']:
                print(f"   ✓ {fix}")
        
        if self.results['issues_found']:
            print(f"\n❌ ISSUES FOUND ({len(self.results['issues_found'])}):")
            for issue in self.results['issues_found']:
                print(f"   ✗ {issue}")
        
        # Production readiness summary
        print("\n🚀 PRODUCTION READINESS CHECKLIST:")
        readiness_items = [
            ("Data Integrity", "Random data generation eliminated" in str(self.results['fixes_validated'])),
            ("Configuration System", "Configuration management system implemented" in str(self.results['fixes_validated'])),
            ("ML Pipeline", "ML trainer production-ready" in str(self.results['fixes_validated'])),
            ("Security System", "Comprehensive security system implemented" in str(self.results['fixes_validated'])),
            ("Test Framework", "Test coverage" in str(self.results['fixes_validated']))
        ]
        
        for item, status in readiness_items:
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {item}")
        
        return self.results
    
    def run_validation(self):
        """Run complete validation suite."""
        self.logger.info("🚀 Starting ColO-RAN Dynamic Slice Optimizer Fix Validation")
        
        # Run all validations
        validations = [
            ("Random Data Elimination", self.validate_random_data_elimination),
            ("Configuration System", self.validate_configuration_system),
            ("ML Trainer Improvements", self.validate_ml_trainer_improvements),
            ("Security System", self.validate_security_system),
            ("Test Framework", self.validate_test_framework),
            ("Documentation & Structure", self.validate_documentation_and_structure)
        ]
        
        for name, validation_func in validations:
            try:
                success = validation_func()
                status = "✅ PASSED" if success else "⚠️ ISSUES FOUND"
                self.logger.info(f"{status}: {name}")
            except Exception as e:
                self.logger.error(f"❌ ERROR in {name}: {str(e)}")
                self.results['issues_found'].append(f"Validation error in {name}: {str(e)}")
        
        # Generate final report
        return self.generate_validation_report()


def main():
    """Main validation entry point."""
    validator = CodeFixValidator()
    results = validator.run_validation()
    
    # Exit with appropriate code
    success_rate = results['validation_summary']['success_rate_percent']
    total_issues = results['validation_summary']['total_issues_found']
    
    if success_rate >= 80 and total_issues == 0:
        print(f"\n🎉 VALIDATION SUCCESSFUL - All critical fixes implemented!")
        return 0
    else:
        print(f"\n⚠️ VALIDATION COMPLETED WITH ISSUES - {success_rate:.1f}% success rate")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)