#!/usr/bin/env python3
"""
Validate Sandbox - Quick validation of testing sandbox setup
Verifies all components are properly configured before autonomous testing
"""

import sys
import json
import importlib
from pathlib import Path
from datetime import datetime

class SandboxValidator:
    def __init__(self):
        self.sandbox_root = Path(__file__).parent
        self.validation_results = {}
        
    def validate_directory_structure(self) -> bool:
        """Validate all required directories exist"""
        print("🗂️  Validating directory structure...")
        
        required_dirs = [
            'baseline_metrics',
            'qmk_integration', 
            'stress_scenarios',
            'sync_validation',
            'monitoring_logs',
            'results_summary',
            'scripts',
            'safety_monitoring'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.sandbox_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"   ❌ Missing directories: {', '.join(missing_dirs)}")
            return False
        
        print("   ✅ All required directories present")
        return True
    
    def validate_core_scripts(self) -> bool:
        """Validate core scripts exist and are executable"""
        print("📝 Validating core scripts...")
        
        required_scripts = [
            'launch_autonomous_testing.py',
            'safety_monitoring/system_guardian.py',
            'safety_monitoring/emergency_stop.py',
            'scripts/autonomous_test_runner.py',
            'scripts/resource_monitor.py',
            'baseline_metrics/baseline_collector.py',
            'stress_scenarios/concurrent_load_test.py'
        ]
        
        missing_scripts = []
        non_executable = []
        
        for script_path in required_scripts:
            full_path = self.sandbox_root / script_path
            
            if not full_path.exists():
                missing_scripts.append(script_path)
            else:
                # Check if executable
                import os
                if not os.access(full_path, os.X_OK):
                    non_executable.append(script_path)
        
        if missing_scripts:
            print(f"   ❌ Missing scripts: {', '.join(missing_scripts)}")
            return False
        
        if non_executable:
            print(f"   ⚠️  Non-executable scripts: {', '.join(non_executable)}")
        
        print("   ✅ All core scripts present")
        return True
    
    def validate_python_dependencies(self) -> bool:
        """Validate required Python modules"""
        print("🐍 Validating Python dependencies...")
        
        required_modules = [
            'psutil',
            'asyncio', 
            'json',
            'pathlib',
            'datetime',
            'logging',
            'threading',
            'concurrent.futures',
            'dataclasses',
            'traceback'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"   ❌ Missing modules: {', '.join(missing_modules)}")
            print(f"   💡 Install with: pip install {' '.join(missing_modules)}")
            return False
        
        print("   ✅ All required modules available")
        return True
    
    def validate_configuration_files(self) -> bool:
        """Validate configuration files are present and valid"""
        print("⚙️  Validating configuration files...")
        
        config_files = [
            'safety_monitoring/safety_config.json'
        ]
        
        for config_file in config_files:
            config_path = self.sandbox_root / config_file
            
            if not config_path.exists():
                print(f"   ❌ Missing config file: {config_file}")
                return False
            
            # Validate JSON format
            try:
                with open(config_path, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON in {config_file}: {e}")
                return False
        
        print("   ✅ All configuration files valid")
        return True
    
    def validate_system_resources(self) -> bool:
        """Validate system has sufficient resources"""
        print("💾 Validating system resources...")
        
        try:
            import psutil
            import shutil
            
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 4.0:
                print(f"   ⚠️  Low memory: {available_gb:.1f}GB available (4GB+ recommended)")
            
            # Check disk space
            _, _, free_bytes = shutil.disk_usage(self.sandbox_root)
            free_gb = free_bytes / (1024**3)
            
            if free_gb < 1.0:
                print(f"   ❌ Insufficient disk space: {free_gb:.1f}GB free (1GB+ required)")
                return False
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                print(f"   ⚠️  Limited CPU cores: {cpu_count} (2+ recommended)")
            
            print(f"   ✅ System resources adequate")
            print(f"      Memory: {available_gb:.1f}GB available")
            print(f"      Disk: {free_gb:.1f}GB free")
            print(f"      CPU: {cpu_count} cores")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error checking system resources: {e}")
            return False
    
    def test_import_modules(self) -> bool:
        """Test importing key modules"""
        print("📦 Testing module imports...")
        
        test_imports = [
            ('scripts.autonomous_test_runner', 'AutonomousTestRunner'),
            ('safety_monitoring.system_guardian', 'SystemGuardian'), 
            ('safety_monitoring.emergency_stop', 'EmergencyStop'),
            ('scripts.resource_monitor', 'ResourceMonitor'),
            ('baseline_metrics.baseline_collector', 'BaselineCollector')
        ]
        
        # Add sandbox root to path for testing
        sys.path.insert(0, str(self.sandbox_root))
        
        failed_imports = []
        
        for module_path, class_name in test_imports:
            try:
                module = importlib.import_module(module_path)
                getattr(module, class_name)
            except Exception as e:
                failed_imports.append(f"{module_path}.{class_name}: {e}")
        
        # Remove from path
        sys.path.pop(0)
        
        if failed_imports:
            print("   ❌ Failed imports:")
            for failure in failed_imports:
                print(f"      {failure}")
            return False
        
        print("   ✅ All module imports successful")
        return True
    
    def run_comprehensive_validation(self) -> bool:
        """Run all validation checks"""
        print("🔍 TESTING SANDBOX VALIDATION")
        print("=" * 50)
        print()
        
        validation_checks = [
            ("Directory Structure", self.validate_directory_structure),
            ("Core Scripts", self.validate_core_scripts),
            ("Python Dependencies", self.validate_python_dependencies),
            ("Configuration Files", self.validate_configuration_files),
            ("System Resources", self.validate_system_resources),
            ("Module Imports", self.test_import_modules)
        ]
        
        all_passed = True
        results = {}
        
        for check_name, check_function in validation_checks:
            try:
                result = check_function()
                results[check_name] = result
                
                if not result:
                    all_passed = False
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error during {check_name}: {e}")
                results[check_name] = False
                all_passed = False
                print()
        
        # Save validation results
        self.save_validation_results(results, all_passed)
        
        # Display final result
        print("=" * 50)
        if all_passed:
            print("🎉 VALIDATION PASSED - Sandbox ready for autonomous testing!")
            print()
            print("Next steps:")
            print("  1. python3 launch_autonomous_testing.py --check-only")
            print("  2. python3 launch_autonomous_testing.py")
        else:
            print("❌ VALIDATION FAILED - Please resolve issues above")
            print()
            print("Common solutions:")
            print("  • pip install psutil")
            print("  • Free up disk space")
            print("  • Close unnecessary applications")
        
        return all_passed
    
    def save_validation_results(self, results: dict, overall_success: bool):
        """Save validation results to file"""
        try:
            results_dir = self.sandbox_root / "results_summary"
            results_dir.mkdir(exist_ok=True)
            
            validation_report = {
                "validation_timestamp": datetime.now().isoformat(),
                "sandbox_path": str(self.sandbox_root),
                "overall_success": overall_success,
                "individual_results": results,
                "python_version": sys.version,
                "validation_script": str(Path(__file__).name)
            }
            
            report_file = results_dir / f"sandbox_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            print(f"📊 Validation report saved to: {report_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save validation report: {e}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""
Sandbox Validation Tool

This script validates that the testing sandbox is properly configured
and ready for autonomous stress testing.

Usage:
    python3 validate_sandbox.py

The validation checks:
    • Directory structure completeness
    • Core script availability and permissions
    • Python module dependencies
    • Configuration file validity
    • System resource sufficiency
    • Module import functionality

A validation report is saved to results_summary/ for reference.
        """)
        return
    
    validator = SandboxValidator()
    success = validator.run_comprehensive_validation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()