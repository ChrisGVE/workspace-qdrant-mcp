#!/usr/bin/env python3
"""
Quick Demonstration of Comprehensive Testing Infrastructure
===========================================================

This script provides a quick demonstration of the testing infrastructure
capabilities without requiring the full workspace-qdrant-mcp dependencies.

It demonstrates:
1. Infrastructure setup and validation
2. Test data generation
3. Safety monitoring
4. Performance monitoring  
5. Validation framework
6. Report generation

Usage:
    python 20250107-0904_demo_infrastructure.py
"""

import asyncio
import sys
import time
from pathlib import Path
import importlib.util

def load_module_from_file(module_name: str, file_path: str):
    """Dynamically load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

async def demo_infrastructure():
    """Demonstrate the comprehensive testing infrastructure."""
    
    print("🎯 COMPREHENSIVE TESTING INFRASTRUCTURE DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Load infrastructure modules
        current_dir = Path(__file__).parent
        
        print("📦 Loading testing infrastructure modules...")
        infra_module = load_module_from_file(
            "test_infrastructure",
            str(current_dir / "20250107-0900_comprehensive_testing_infrastructure.py")
        )
        
        validation_module = load_module_from_file(
            "test_validation_framework", 
            str(current_dir / "20250107-0901_test_validation_framework.py")
        )
        
        print("✅ Modules loaded successfully")
        
        # Extract required classes
        ComprehensiveTestingInfrastructure = infra_module.ComprehensiveTestingInfrastructure
        TestEnvironmentManager = infra_module.TestEnvironmentManager
        SafetyMonitor = infra_module.SafetyMonitor
        PerformanceMonitor = infra_module.PerformanceMonitor
        TestDataGenerator = infra_module.TestDataGenerator
        ValidationFramework = validation_module.ValidationFramework
        ValidationLevel = validation_module.ValidationLevel
        TestPhase = validation_module.TestPhase
        
        print("\n🔧 INFRASTRUCTURE SETUP")
        print("-" * 30)
        
        # 1. Test Environment Manager
        print("Setting up test environment...")
        env_manager = TestEnvironmentManager()
        print(f"✅ Test environment created at: {env_manager.base_dir}")
        print(f"   📁 Test data dir: {env_manager.test_data_dir}")
        print(f"   📊 Results dir: {env_manager.results_dir}")
        print(f"   📝 Logs dir: {env_manager.logs_dir}")
        
        # 2. Safety Monitor
        print("\nSetting up safety monitoring...")
        safety_monitor = SafetyMonitor(
            max_cpu_percent=95.0,  # Higher threshold for demo
            max_memory_percent=90.0,
            check_interval=0.5
        )
        safety_monitor.start_monitoring()
        print("✅ Safety monitoring active")
        
        # 3. Performance Monitor
        print("\nSetting up performance monitoring...")
        perf_monitor = PerformanceMonitor(collection_interval=0.2)
        perf_monitor.start_monitoring()
        print("✅ Performance monitoring active")
        
        print("\n📊 TEST DATA GENERATION")
        print("-" * 30)
        
        # 4. Test Data Generation
        print("Generating test data...")
        data_generator = TestDataGenerator(env_manager.test_data_dir)
        
        # Generate different types of files
        source_files = data_generator.generate_source_code_files(count=20, languages=['python', 'javascript', 'markdown'])
        print(f"✅ Generated {len(source_files)} source code files")
        
        doc_files = data_generator.generate_documentation_files(count=10)
        print(f"✅ Generated {len(doc_files)} documentation files")
        
        config_files = data_generator.generate_configuration_files(count=5)
        print(f"✅ Generated {len(config_files)} configuration files")
        
        # Check file content samples
        if source_files:
            sample_file = source_files[0]
            content_preview = sample_file.read_text()[:200] + "..."
            print(f"📄 Sample content from {sample_file.name}:")
            print(f"   {content_preview}")
        
        print("\n🔍 VALIDATION FRAMEWORK")
        print("-" * 30)
        
        # 5. Validation Framework
        print("Setting up validation framework...")
        validator = ValidationFramework(
            env_manager=env_manager,
            validation_level=ValidationLevel.STANDARD
        )
        print("✅ Validation framework ready")
        
        # Test validation capabilities
        print("Running sample validations...")
        
        # Simulate LSP integration validation
        lsp_results = await validator.validate_phase(TestPhase.LSP_INTEGRATION)
        print(f"✅ LSP Integration: {len(lsp_results)} validations completed")
        for result in lsp_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"   {status}: {result.test_id} (accuracy: {result.accuracy_score:.2f}, {result.performance_ms:.1f}ms)")
        
        # Simulate ingestion validation
        test_data = {"generated_files": {"source_code": source_files, "documentation": doc_files}}
        ingestion_results = await validator.validate_phase(TestPhase.INGESTION_CAPABILITIES, test_data)
        print(f"✅ Ingestion: {len(ingestion_results)} validations completed")
        for result in ingestion_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"   {status}: {result.test_id} (accuracy: {result.accuracy_score:.2f}, {result.performance_ms:.1f}ms)")
        
        print("\n⚡ PERFORMANCE MONITORING")
        print("-" * 30)
        
        # Simulate some work to collect performance metrics
        print("Simulating workload for performance measurement...")
        for i in range(5):
            # Simulate some processing work
            await asyncio.sleep(0.1)
            print(f"   Processing batch {i+1}/5...")
        
        # Stop monitoring and collect results
        metrics = perf_monitor.stop_monitoring()
        safety_monitor.stop_monitoring()
        
        print(f"✅ Performance monitoring completed:")
        print(f"   Duration: {metrics.duration:.2f}s")
        print(f"   Average CPU: {metrics.avg_cpu_usage:.1f}%")
        print(f"   Peak Memory: {metrics.peak_memory_usage:.1f}%")
        print(f"   Samples collected: {len(metrics.cpu_usage)} CPU, {len(metrics.memory_usage)} memory")
        
        print("\n📋 REPORTING")
        print("-" * 30)
        
        # 6. Report Generation
        print("Generating comprehensive validation report...")
        report = validator.generate_validation_report()
        
        if "validation_summary" in report:
            summary = report["validation_summary"]
            print(f"✅ Validation Report Generated:")
            print(f"   Total Tests: {summary['total_tests']}")
            print(f"   Passed: {summary['passed_tests']}")
            print(f"   Success Rate: {summary['success_rate']:.1f}%")
            print(f"   Overall Score: {summary['overall_score']:.1f}/100")
            print(f"   Average Accuracy: {summary['overall_accuracy']:.3f}")
            print(f"   Average Performance: {summary['overall_performance_ms']:.1f}ms")
            
        if "recommendations" in report and report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        # Save a demonstration report
        report_file = env_manager.results_dir / "demo_infrastructure_report.json"
        import json
        with open(report_file, 'w') as f:
            json.dump({
                "demo_info": {
                    "timestamp": time.time(),
                    "test_environment": str(env_manager.base_dir),
                    "files_generated": len(source_files) + len(doc_files) + len(config_files)
                },
                "performance_metrics": {
                    "duration": metrics.duration,
                    "avg_cpu_usage": metrics.avg_cpu_usage,
                    "peak_memory_usage": metrics.peak_memory_usage
                },
                "validation_report": report
            }, f, indent=2)
        print(f"📄 Demo report saved to: {report_file}")
        
        print("\n🧹 CLEANUP")
        print("-" * 30)
        
        # 7. Cleanup
        print("Performing cleanup...")
        
        # Register files for cleanup
        for file_list in [source_files, doc_files, config_files]:
            for file_path in file_list:
                env_manager.add_temp_file(file_path)
                
        print(f"   📝 Registered {len(source_files + doc_files + config_files)} files for cleanup")
        
        # Note: In real usage, you might call env_manager.cleanup(force=True)
        # For demo, we'll preserve the files so you can inspect them
        print(f"   📁 Test files preserved at: {env_manager.base_dir}")
        print(f"   💡 To cleanup: rm -rf {env_manager.base_dir}")
        
        print("\n🎉 DEMONSTRATION COMPLETE")
        print("=" * 50)
        print("Infrastructure components successfully demonstrated:")
        print("✅ Environment Management")
        print("✅ Safety Monitoring") 
        print("✅ Performance Monitoring")
        print("✅ Test Data Generation")
        print("✅ Validation Framework")
        print("✅ Report Generation")
        print("✅ Cleanup Procedures")
        
        print(f"\n📁 Explore generated files at: {env_manager.base_dir}")
        print(f"📊 View results at: {env_manager.results_dir}")
        print(f"📝 Check logs at: {env_manager.logs_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

async def main():
    """Main entry point for the demonstration."""
    try:
        exit_code = await demo_infrastructure()
        return exit_code
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)