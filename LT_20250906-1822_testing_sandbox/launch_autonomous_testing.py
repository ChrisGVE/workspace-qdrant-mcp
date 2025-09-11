#!/usr/bin/env python3
"""
Launch Autonomous Testing - Main entry point for overnight stress testing
Orchestrates the complete autonomous testing campaign with safety monitoring
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.autonomous_test_runner import AutonomousTestRunner, CampaignConfig
from safety_monitoring.system_guardian import SystemGuardian
from safety_monitoring.emergency_stop import EmergencyStop
from baseline_metrics.baseline_collector import BaselineCollector

class TestingLauncher:
    def __init__(self):
        self.testing_sandbox_root = Path(__file__).parent
        self.setup_logging()
        
    def setup_logging(self):
        """Setup launcher logging"""
        log_dir = self.testing_sandbox_root / "monitoring_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def display_banner(self):
        """Display launch banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    AUTONOMOUS STRESS TESTING CAMPAIGN                         ║
║                        Workspace-Qdrant MCP Testing                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Duration: 8-12 hours overnight operation                                    ║
║  Safety: Continuous monitoring with emergency stops                          ║
║  Scope: Progressive load testing with autonomous escalation                  ║
║  Report: Comprehensive analysis and recommendations                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Testing Sandbox: {self.testing_sandbox_root}")
        print()
    
    def check_prerequisites(self) -> Dict[str, Any]:
        """Check system prerequisites for autonomous testing"""
        self.logger.info("Checking system prerequisites")
        
        prerequisites = {
            "python_version": True,
            "required_modules": True,
            "directory_structure": True,
            "safety_systems": True,
            "disk_space": True,
            "memory": True,
            "all_checks_passed": False
        }
        
        try:
            # Check Python version
            if sys.version_info < (3, 7):
                prerequisites["python_version"] = False
                self.logger.error(f"Python 3.7+ required, found {sys.version}")
            
            # Check required modules
            required_modules = ['psutil', 'asyncio', 'json', 'pathlib', 'datetime']
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    prerequisites["required_modules"] = False
                    self.logger.error(f"Required module missing: {module}")
                    break
            
            # Check directory structure
            required_dirs = [
                'baseline_metrics', 'qmk_integration', 'stress_scenarios',
                'sync_validation', 'monitoring_logs', 'results_summary',
                'scripts', 'safety_monitoring'
            ]
            
            for dir_name in required_dirs:
                dir_path = self.testing_sandbox_root / dir_name
                if not dir_path.exists():
                    prerequisites["directory_structure"] = False
                    self.logger.error(f"Required directory missing: {dir_path}")
                    break
            
            # Check safety systems
            safety_guardian_path = self.testing_sandbox_root / "safety_monitoring" / "system_guardian.py"
            emergency_stop_path = self.testing_sandbox_root / "safety_monitoring" / "emergency_stop.py"
            
            if not safety_guardian_path.exists() or not emergency_stop_path.exists():
                prerequisites["safety_systems"] = False
                self.logger.error("Safety monitoring systems not found")
            
            # Check disk space (minimum 1GB free)
            import shutil
            _, _, free_bytes = shutil.disk_usage(self.testing_sandbox_root)
            free_gb = free_bytes / (1024**3)
            
            if free_gb < 1.0:
                prerequisites["disk_space"] = False
                self.logger.error(f"Insufficient disk space: {free_gb:.1f}GB free, need at least 1GB")
            
            # Check available memory (minimum 4GB)
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 4.0:
                prerequisites["memory"] = False
                self.logger.error(f"Insufficient memory: {available_gb:.1f}GB available, recommend at least 4GB")
            
            # Overall check
            prerequisites["all_checks_passed"] = all(
                prerequisites[key] for key in prerequisites.keys() 
                if key != "all_checks_passed"
            )
            
            if prerequisites["all_checks_passed"]:
                self.logger.info("All prerequisite checks passed")
            else:
                self.logger.error("Some prerequisite checks failed")
            
        except Exception as e:
            self.logger.error(f"Error checking prerequisites: {e}")
            prerequisites["all_checks_passed"] = False
        
        return prerequisites
    
    def get_user_confirmation(self) -> bool:
        """Get user confirmation for autonomous testing"""
        print("AUTONOMOUS TESTING CONFIRMATION")
        print("=" * 50)
        print()
        print("This will start an 8-12 hour autonomous stress testing campaign.")
        print("The system will:")
        print("  • Run progressive load tests automatically")
        print("  • Monitor system resources continuously") 
        print("  • Stop if safety thresholds are exceeded")
        print("  • Generate comprehensive reports")
        print("  • Require minimal human intervention")
        print()
        print("Safety features enabled:")
        print("  ✓ Real-time resource monitoring")
        print("  ✓ Emergency stop mechanisms")
        print("  ✓ Progressive test escalation")
        print("  ✓ Automatic recovery procedures")
        print()
        
        while True:
            response = input("Proceed with autonomous testing? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print("Please enter 'yes' or 'no'")
    
    def create_campaign_config(self) -> CampaignConfig:
        """Create campaign configuration based on system capabilities"""
        self.logger.info("Creating campaign configuration")
        
        # Get system info for configuration
        import psutil
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        # Adjust configuration based on system resources
        if memory.total / (1024**3) >= 16 and cpu_count >= 8:
            # High-end system
            duration_hours = 12
            max_failures = 5
        elif memory.total / (1024**3) >= 8 and cpu_count >= 4:
            # Mid-range system
            duration_hours = 8
            max_failures = 3
        else:
            # Conservative for lower-end systems
            duration_hours = 6
            max_failures = 2
        
        config = CampaignConfig(
            total_duration_hours=duration_hours,
            safety_check_interval_minutes=15,
            baseline_health_requirement=70,
            max_consecutive_failures=max_failures,
            emergency_cooldown_minutes=30,
            progressive_escalation=True,
            auto_recovery=True
        )
        
        self.logger.info(f"Campaign configured for {duration_hours} hours with {max_failures} max failures")
        return config
    
    def save_launch_info(self, prerequisites: Dict[str, Any], config: CampaignConfig) -> str:
        """Save launch information for reference"""
        launch_info = {
            "launch_timestamp": datetime.now().isoformat(),
            "testing_sandbox_path": str(self.testing_sandbox_root),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "prerequisites_check": prerequisites,
            "campaign_config": config.__dict__,
            "expected_completion": (
                datetime.now().timestamp() + (config.total_duration_hours * 3600)
            )
        }
        
        results_dir = self.testing_sandbox_root / "results_summary"
        results_dir.mkdir(exist_ok=True)
        
        launch_file = results_dir / f"launch_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(launch_file, 'w') as f:
                json.dump(launch_info, f, indent=2)
            
            self.logger.info(f"Launch information saved to {launch_file}")
            return str(launch_file)
            
        except Exception as e:
            self.logger.error(f"Error saving launch info: {e}")
            return ""
    
    async def launch_autonomous_testing(self) -> Dict[str, Any]:
        """Launch the autonomous testing campaign"""
        try:
            self.display_banner()
            
            # Check prerequisites
            print("Checking system prerequisites...")
            prerequisites = self.check_prerequisites()
            
            if not prerequisites["all_checks_passed"]:
                print("❌ Prerequisite checks failed!")
                print("Please resolve the issues above before launching autonomous testing.")
                return {"success": False, "error": "Prerequisites not met"}
            
            print("✅ All prerequisite checks passed!")
            print()
            
            # Get user confirmation
            if not self.get_user_confirmation():
                print("Autonomous testing cancelled by user.")
                return {"success": False, "error": "User cancelled"}
            
            print()
            print("🚀 Launching autonomous testing campaign...")
            
            # Create campaign configuration
            config = self.create_campaign_config()
            
            # Save launch information
            launch_file = self.save_launch_info(prerequisites, config)
            
            # Create and start the autonomous test runner
            runner = AutonomousTestRunner(config)
            
            print(f"Campaign Duration: {config.total_duration_hours} hours")
            print(f"Expected Completion: {datetime.fromtimestamp(datetime.now().timestamp() + config.total_duration_hours * 3600).strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("🔄 Campaign starting...")
            print("💡 Use emergency_stop.py to halt testing if needed")
            print("📊 Monitor progress in monitoring_logs/")
            print()
            
            # Run the campaign
            campaign_result = await runner.run_autonomous_campaign()
            
            # Display completion summary
            print("\n" + "="*60)
            print("🏁 AUTONOMOUS TESTING CAMPAIGN COMPLETED")
            print("="*60)
            
            campaign_info = campaign_result.get("campaign_info", {})
            summary = campaign_result.get("campaign_summary", {})
            
            print(f"Duration: {campaign_info.get('duration_hours', 0):.1f} hours")
            print(f"Phases: {summary.get('successful_phases', 0)}/{summary.get('total_phases_attempted', 0)} successful")
            print(f"Operations: {summary.get('total_successful_operations', 0)}/{summary.get('total_operations', 0)} successful")
            print(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
            
            # Show key recommendations
            recommendations = campaign_result.get("recommendations", [])
            if recommendations:
                print("\nKey Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec}")
            
            print(f"\n📁 Detailed reports available in: {self.testing_sandbox_root}/results_summary/")
            
            return {
                "success": True,
                "campaign_result": campaign_result,
                "launch_file": launch_file
            }
            
        except KeyboardInterrupt:
            print("\n⚠️  Testing interrupted by user (Ctrl+C)")
            self.logger.warning("Testing interrupted by user")
            
            # Trigger emergency stop
            emergency = EmergencyStop()
            emergency.stop_all_tests(force=True)
            
            return {"success": False, "error": "Interrupted by user"}
            
        except Exception as e:
            print(f"\n❌ Error during autonomous testing: {e}")
            self.logger.error(f"Launch error: {e}")
            
            # Trigger emergency stop on error
            try:
                emergency = EmergencyStop()
                emergency.stop_all_tests(force=True)
            except:
                pass
            
            return {"success": False, "error": str(e)}


def show_help():
    """Show help information"""
    help_text = """
Autonomous Stress Testing Campaign Launcher

This script launches an 8-12 hour autonomous stress testing campaign for the
workspace-qdrant MCP system. The testing includes:

• Progressive load testing with automatic escalation
• Continuous safety monitoring and resource tracking  
• Emergency stop mechanisms for system protection
• Comprehensive reporting and recommendations

USAGE:
    python3 launch_autonomous_testing.py [options]

OPTIONS:
    --help, -h          Show this help message
    --check-only        Only run prerequisite checks, don't start testing
    --emergency-stop    Stop any currently running tests

SAFETY:
    The system includes multiple safety mechanisms:
    - Real-time resource monitoring with automatic thresholds
    - Emergency stop script (emergency_stop.py) for manual intervention
    - Progressive test escalation with automatic backoff
    - System resource reservation (20% CPU/memory kept free)

MONITORING:
    During testing, monitor progress via:
    - monitoring_logs/ directory for real-time logs
    - results_summary/ directory for reports
    - Use 'python3 safety_monitoring/emergency_stop.py' to halt if needed

REQUIREMENTS:
    - Python 3.7+
    - At least 4GB available memory
    - At least 1GB free disk space
    - psutil module for system monitoring
    """
    print(help_text)


async def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            show_help()
            return
        elif sys.argv[1] == '--check-only':
            launcher = TestingLauncher()
            prerequisites = launcher.check_prerequisites()
            
            if prerequisites["all_checks_passed"]:
                print("✅ All prerequisite checks passed! System ready for autonomous testing.")
                sys.exit(0)
            else:
                print("❌ Some prerequisite checks failed. See logs for details.")
                sys.exit(1)
        
        elif sys.argv[1] == '--emergency-stop':
            print("🛑 Triggering emergency stop...")
            emergency = EmergencyStop()
            result = emergency.stop_all_tests(force=True)
            print(f"Emergency stop result: {result['message']}")
            return
    
    # Launch autonomous testing
    launcher = TestingLauncher()
    result = await launcher.launch_autonomous_testing()
    
    if result["success"]:
        print("\n✅ Autonomous testing campaign completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Autonomous testing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())