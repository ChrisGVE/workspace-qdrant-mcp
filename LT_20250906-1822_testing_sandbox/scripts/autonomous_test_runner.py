#!/usr/bin/env python3
"""
Autonomous Test Runner - Orchestrates 8-12 hour overnight stress testing campaign
Runs completely autonomously with progressive escalation and safety monitoring
"""

import asyncio
import json
import logging
import time
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import traceback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.resource_monitor import ResourceMonitor
from safety_monitoring.system_guardian import SystemGuardian
from baseline_metrics.baseline_collector import BaselineCollector
from stress_scenarios.concurrent_load_test import ConcurrentLoadTest, LoadTestConfig

@dataclass
class TestPhase:
    """Configuration for a test phase"""
    name: str
    duration_minutes: int
    max_concurrent_connections: int
    operations_per_connection: int
    stress_level: str
    safety_threshold_multiplier: float = 1.0
    required_baseline_health: int = 70

@dataclass
class CampaignConfig:
    """Configuration for the entire testing campaign"""
    total_duration_hours: int = 8
    safety_check_interval_minutes: int = 15
    baseline_health_requirement: int = 70
    max_consecutive_failures: int = 3
    emergency_cooldown_minutes: int = 30
    progressive_escalation: bool = True
    auto_recovery: bool = True

class AutonomousTestRunner:
    def __init__(self, campaign_config: CampaignConfig = None):
        self.config = campaign_config or CampaignConfig()
        self.setup_logging()
        
        # Initialize monitoring systems
        self.system_guardian = SystemGuardian()
        self.resource_monitor = ResourceMonitor(monitoring_interval=10)
        self.baseline_collector = BaselineCollector()
        
        # Test state
        self.campaign_running = False
        self.campaign_start_time = None
        self.campaign_end_time = None
        self.current_phase = None
        self.phase_results = []
        self.emergency_stops = 0
        self.consecutive_failures = 0
        
        # Safety state
        self.emergency_stop = False
        self.last_safety_check = None
        self.baseline_health_score = None
        
        # Define test phases for progressive escalation
        self.test_phases = [
            TestPhase("warmup", 30, 2, 10, "low", 0.8, 60),
            TestPhase("light_load", 60, 3, 25, "low", 0.9, 65),
            TestPhase("moderate_load_1", 90, 5, 40, "moderate", 1.0, 70),
            TestPhase("moderate_load_2", 90, 7, 50, "moderate", 1.0, 75),
            TestPhase("high_load_1", 120, 10, 75, "high", 1.1, 80),
            TestPhase("high_load_2", 120, 12, 100, "high", 1.1, 80),
            TestPhase("stress_test", 150, 15, 150, "extreme", 1.2, 85),
            TestPhase("endurance_test", 180, 8, 200, "high", 1.0, 75)
        ]
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """Setup autonomous test runner logging"""
        log_dir = Path(__file__).parent.parent / "monitoring_logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"autonomous_test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown")
        self.emergency_stop = True
        self.campaign_running = False
    
    async def validate_system_readiness(self) -> Dict[str, Any]:
        """Validate system is ready for autonomous testing"""
        self.logger.info("Validating system readiness for autonomous testing")
        
        try:
            # Collect baseline
            baseline_report = self.baseline_collector.generate_full_baseline()
            
            health_score = baseline_report.get("health_score", {})
            self.baseline_health_score = health_score.get("overall_score", 0)
            
            readiness_check = {
                "timestamp": datetime.now().isoformat(),
                "baseline_health_score": self.baseline_health_score,
                "meets_minimum_requirement": self.baseline_health_score >= self.config.baseline_health_requirement,
                "baseline_report": baseline_report,
                "system_resources": {
                    "memory_available": baseline_report.get("system_baseline", {}).get("system_metrics", {}).get("memory", {}).get("available_gb", 0),
                    "cpu_baseline": baseline_report.get("system_baseline", {}).get("system_metrics", {}).get("cpu", {}).get("percent_used", 0),
                    "disk_free": baseline_report.get("system_baseline", {}).get("system_metrics", {}).get("disk", {}).get("free_gb", 0)
                },
                "recommended_phases": []
            }
            
            # Adjust test phases based on system capacity
            if self.baseline_health_score >= 90:
                readiness_check["recommended_phases"] = self.test_phases  # All phases
            elif self.baseline_health_score >= 80:
                readiness_check["recommended_phases"] = self.test_phases[:6]  # Skip most intensive
            elif self.baseline_health_score >= 70:
                readiness_check["recommended_phases"] = self.test_phases[:4]  # Conservative approach
            else:
                readiness_check["recommended_phases"] = self.test_phases[:2]  # Very conservative
            
            # Save readiness report
            self.save_readiness_report(readiness_check)
            
            return readiness_check
            
        except Exception as e:
            self.logger.error(f"Error validating system readiness: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "meets_minimum_requirement": False,
                "baseline_health_score": 0
            }
    
    def save_readiness_report(self, readiness_check: Dict[str, Any]):
        """Save system readiness report"""
        results_dir = Path(__file__).parent.parent / "results_summary"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = results_dir / f"system_readiness_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(readiness_check, f, indent=2)
            
            self.logger.info(f"Readiness report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving readiness report: {e}")
    
    async def perform_safety_check(self) -> Dict[str, Any]:
        """Perform comprehensive safety check"""
        self.logger.info("Performing safety check")
        
        try:
            # Get current system metrics
            current_metrics = self.resource_monitor.collect_current_metrics()
            
            # Check against safety thresholds (adjusted for current phase)
            safety_multiplier = self.current_phase.safety_threshold_multiplier if self.current_phase else 1.0
            
            memory_threshold = 80 * safety_multiplier
            cpu_threshold = 85 * safety_multiplier
            
            memory_percent = current_metrics.get("memory", {}).get("percent_used", 0)
            cpu_percent = current_metrics.get("cpu", {}).get("percent_used", 0)
            
            safety_violations = []
            safety_warnings = []
            
            if memory_percent > memory_threshold:
                safety_violations.append(f"Memory usage {memory_percent:.1f}% exceeds threshold {memory_threshold:.1f}%")
            elif memory_percent > memory_threshold - 10:
                safety_warnings.append(f"Memory usage {memory_percent:.1f}% approaching threshold")
            
            if cpu_percent > cpu_threshold:
                safety_violations.append(f"CPU usage {cpu_percent:.1f}% exceeds threshold {cpu_threshold:.1f}%")
            elif cpu_percent > cpu_threshold - 10:
                safety_warnings.append(f"CPU usage {cpu_percent:.1f}% approaching threshold")
            
            safety_check = {
                "timestamp": datetime.now().isoformat(),
                "current_phase": self.current_phase.name if self.current_phase else "none",
                "safety_multiplier": safety_multiplier,
                "system_metrics": current_metrics,
                "thresholds": {
                    "memory_threshold": memory_threshold,
                    "cpu_threshold": cpu_threshold
                },
                "violations": safety_violations,
                "warnings": safety_warnings,
                "safe_to_continue": len(safety_violations) == 0,
                "recommend_phase_downgrade": len(safety_violations) > 0 or len(safety_warnings) > 2
            }
            
            self.last_safety_check = datetime.now()
            
            return safety_check
            
        except Exception as e:
            self.logger.error(f"Error performing safety check: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "safe_to_continue": False,
                "recommend_emergency_stop": True
            }
    
    async def run_test_phase(self, phase: TestPhase) -> Dict[str, Any]:
        """Run a single test phase"""
        self.logger.info(f"Starting test phase: {phase.name}")
        self.current_phase = phase
        
        phase_start_time = time.time()
        
        try:
            # Perform safety check before starting phase
            safety_check = await self.perform_safety_check()
            
            if not safety_check.get("safe_to_continue", False):
                raise Exception(f"Safety check failed before phase {phase.name}: {safety_check.get('violations', [])}")
            
            # Configure load test for this phase
            load_config = LoadTestConfig(
                max_concurrent_connections=phase.max_concurrent_connections,
                operations_per_connection=phase.operations_per_connection,
                test_duration_minutes=phase.duration_minutes,
                stress_level=phase.stress_level,
                ramp_up_duration_seconds=min(60, phase.duration_minutes * 6)  # 10% of phase duration
            )
            
            # Create and run load test
            load_test = ConcurrentLoadTest(load_config)
            
            # Register load test process with system guardian
            self.system_guardian.register_test_process(os.getpid(), f"phase_{phase.name}")
            
            # Run the load test
            test_result = await load_test.run_load_test()
            
            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time
            
            # Analyze phase results
            phase_analysis = self.analyze_phase_results(phase, test_result, phase_duration)
            
            phase_report = {
                "phase_info": asdict(phase),
                "start_time": datetime.fromtimestamp(phase_start_time).isoformat(),
                "end_time": datetime.fromtimestamp(phase_end_time).isoformat(),
                "duration_seconds": phase_duration,
                "test_result": test_result,
                "phase_analysis": phase_analysis,
                "success": phase_analysis.get("phase_successful", False)
            }
            
            # Save phase report
            self.save_phase_report(phase_report)
            
            # Reset consecutive failures on success
            if phase_report["success"]:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            self.logger.info(f"Phase {phase.name} completed: {'SUCCESS' if phase_report['success'] else 'FAILURE'}")
            
            return phase_report
            
        except Exception as e:
            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time
            
            self.logger.error(f"Phase {phase.name} failed: {e}")
            self.consecutive_failures += 1
            
            return {
                "phase_info": asdict(phase),
                "start_time": datetime.fromtimestamp(phase_start_time).isoformat(),
                "end_time": datetime.fromtimestamp(phase_end_time).isoformat(),
                "duration_seconds": phase_duration,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        
        finally:
            self.current_phase = None
    
    def analyze_phase_results(self, phase: TestPhase, test_result: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Analyze the results of a test phase"""
        summary = test_result.get("summary", {})
        
        # Define success criteria for each stress level
        success_criteria = {
            "low": {"min_success_rate": 95, "max_avg_response_time": 2.0},
            "moderate": {"min_success_rate": 90, "max_avg_response_time": 3.0},
            "high": {"min_success_rate": 85, "max_avg_response_time": 5.0},
            "extreme": {"min_success_rate": 80, "max_avg_response_time": 8.0}
        }
        
        criteria = success_criteria.get(phase.stress_level, success_criteria["moderate"])
        
        success_rate = summary.get("success_rate", 0)
        avg_response_time = summary.get("avg_response_time", float('inf'))
        operations_per_second = summary.get("operations_per_second", 0)
        
        # Evaluate success
        meets_success_rate = success_rate >= criteria["min_success_rate"]
        meets_response_time = avg_response_time <= criteria["max_avg_response_time"]
        
        phase_successful = meets_success_rate and meets_response_time
        
        # Performance rating
        if success_rate >= 95 and avg_response_time <= 1.0:
            performance_rating = "EXCELLENT"
        elif success_rate >= 90 and avg_response_time <= 2.0:
            performance_rating = "GOOD"
        elif success_rate >= 80 and avg_response_time <= 5.0:
            performance_rating = "ACCEPTABLE"
        else:
            performance_rating = "POOR"
        
        return {
            "phase_successful": phase_successful,
            "performance_rating": performance_rating,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "operations_per_second": operations_per_second,
            "meets_success_rate_requirement": meets_success_rate,
            "meets_response_time_requirement": meets_response_time,
            "criteria_used": criteria,
            "recommendations": self.generate_phase_recommendations(phase, summary)
        }
    
    def generate_phase_recommendations(self, phase: TestPhase, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on phase results"""
        recommendations = []
        
        success_rate = summary.get("success_rate", 0)
        avg_response_time = summary.get("avg_response_time", 0)
        failed_operations = summary.get("failed_operations", 0)
        
        if success_rate < 85:
            recommendations.append("Consider reducing concurrent connections for stability")
        
        if avg_response_time > 3.0:
            recommendations.append("Response times are high - system may be under stress")
        
        if failed_operations > 10:
            recommendations.append("High failure count - investigate error patterns")
        
        if success_rate > 95 and avg_response_time < 1.0:
            recommendations.append("System performing well - can potentially handle higher load")
        
        return recommendations
    
    def save_phase_report(self, phase_report: Dict[str, Any]):
        """Save individual phase report"""
        results_dir = Path(__file__).parent.parent / "results_summary"
        results_dir.mkdir(exist_ok=True)
        
        phase_name = phase_report.get("phase_info", {}).get("name", "unknown")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = results_dir / f"phase_report_{phase_name}_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(phase_report, f, indent=2)
            
            self.logger.info(f"Phase report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving phase report: {e}")
    
    def should_continue_campaign(self) -> tuple[bool, str]:
        """Determine if the campaign should continue"""
        if self.emergency_stop:
            return False, "Emergency stop triggered"
        
        if not self.campaign_running:
            return False, "Campaign stopped"
        
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            return False, f"Maximum consecutive failures ({self.config.max_consecutive_failures}) reached"
        
        # Check campaign duration
        if self.campaign_start_time:
            elapsed_hours = (time.time() - self.campaign_start_time) / 3600
            if elapsed_hours >= self.config.total_duration_hours:
                return False, f"Campaign duration ({self.config.total_duration_hours} hours) completed"
        
        return True, "Continue campaign"
    
    async def run_autonomous_campaign(self) -> Dict[str, Any]:
        """Run the complete autonomous testing campaign"""
        self.logger.info("Starting autonomous stress testing campaign")
        
        try:
            # Initialize monitoring
            self.system_guardian.start_monitoring()
            self.resource_monitor.start_monitoring()
            
            self.campaign_running = True
            self.campaign_start_time = time.time()
            
            # Phase 1: System readiness validation
            self.logger.info("Phase 1: System readiness validation")
            readiness_check = await self.validate_system_readiness()
            
            if not readiness_check.get("meets_minimum_requirement", False):
                raise Exception(f"System does not meet minimum readiness requirements. Health score: {readiness_check.get('baseline_health_score', 0)}")
            
            # Use recommended phases based on system capacity
            recommended_phases = readiness_check.get("recommended_phases", self.test_phases[:2])
            
            self.logger.info(f"System ready. Running {len(recommended_phases)} test phases")
            
            # Phase 2: Execute test phases
            for i, phase in enumerate(recommended_phases):
                if not self.campaign_running or self.emergency_stop:
                    break
                
                should_continue, reason = self.should_continue_campaign()
                if not should_continue:
                    self.logger.warning(f"Stopping campaign: {reason}")
                    break
                
                self.logger.info(f"Executing phase {i+1}/{len(recommended_phases)}: {phase.name}")
                
                # Run the phase
                phase_result = await self.run_test_phase(phase)
                self.phase_results.append(phase_result)
                
                # Check if we should continue after this phase
                if not phase_result.get("success", False):
                    self.logger.warning(f"Phase {phase.name} failed")
                    
                    if self.consecutive_failures >= self.config.max_consecutive_failures:
                        self.logger.error("Too many consecutive failures, stopping campaign")
                        break
                    
                    # Apply cooldown on failure
                    if self.config.auto_recovery:
                        cooldown_minutes = self.config.emergency_cooldown_minutes
                        self.logger.info(f"Applying {cooldown_minutes} minute cooldown after failure")
                        await asyncio.sleep(cooldown_minutes * 60)
                else:
                    # Short break between successful phases
                    self.logger.info("Phase successful, taking 5-minute break")
                    await asyncio.sleep(300)  # 5 minute break
                
                # Periodic safety check
                if (time.time() - self.campaign_start_time) % (self.config.safety_check_interval_minutes * 60) < 300:
                    safety_check = await self.perform_safety_check()
                    if not safety_check.get("safe_to_continue", True):
                        self.logger.warning("Safety check failed, stopping campaign")
                        break
            
            self.campaign_end_time = time.time()
            
            # Generate final campaign report
            campaign_report = self.generate_campaign_report(readiness_check)
            
            # Save campaign report
            self.save_campaign_report(campaign_report)
            
            self.logger.info("Autonomous testing campaign completed")
            
            return campaign_report
            
        except Exception as e:
            self.logger.error(f"Autonomous campaign failed: {e}")
            self.campaign_end_time = time.time()
            
            return {
                "campaign_info": {
                    "start_time": datetime.fromtimestamp(self.campaign_start_time).isoformat() if self.campaign_start_time else None,
                    "end_time": datetime.fromtimestamp(self.campaign_end_time).isoformat() if self.campaign_end_time else None,
                    "error": str(e),
                    "success": False
                },
                "phase_results": self.phase_results
            }
        
        finally:
            self.campaign_running = False
            
            # Clean up monitoring
            self.system_guardian.stop_monitoring()
            self.resource_monitor.stop_monitoring()
    
    def generate_campaign_report(self, readiness_check: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive campaign report"""
        campaign_duration = self.campaign_end_time - self.campaign_start_time if self.campaign_start_time and self.campaign_end_time else 0
        
        # Analyze overall campaign results
        successful_phases = sum(1 for phase in self.phase_results if phase.get("success", False))
        total_phases = len(self.phase_results)
        
        # Aggregate statistics
        total_operations = sum(
            phase.get("test_result", {}).get("summary", {}).get("total_operations", 0)
            for phase in self.phase_results
        )
        
        total_successful_operations = sum(
            phase.get("test_result", {}).get("summary", {}).get("successful_operations", 0)
            for phase in self.phase_results
        )
        
        overall_success_rate = (total_successful_operations / total_operations * 100) if total_operations > 0 else 0
        
        # Performance trends
        response_times = []
        for phase in self.phase_results:
            avg_time = phase.get("test_result", {}).get("summary", {}).get("avg_response_time", 0)
            if avg_time > 0:
                response_times.append(avg_time)
        
        performance_trend = "STABLE"
        if len(response_times) > 2:
            if response_times[-1] > response_times[0] * 1.5:
                performance_trend = "DEGRADING"
            elif response_times[-1] < response_times[0] * 0.8:
                performance_trend = "IMPROVING"
        
        report = {
            "campaign_info": {
                "start_time": datetime.fromtimestamp(self.campaign_start_time).isoformat() if self.campaign_start_time else None,
                "end_time": datetime.fromtimestamp(self.campaign_end_time).isoformat() if self.campaign_end_time else None,
                "duration_hours": campaign_duration / 3600,
                "config": asdict(self.config),
                "baseline_health_score": self.baseline_health_score
            },
            "campaign_summary": {
                "total_phases_attempted": total_phases,
                "successful_phases": successful_phases,
                "failed_phases": total_phases - successful_phases,
                "phase_success_rate": (successful_phases / total_phases * 100) if total_phases > 0 else 0,
                "total_operations": total_operations,
                "total_successful_operations": total_successful_operations,
                "overall_success_rate": overall_success_rate,
                "performance_trend": performance_trend,
                "emergency_stops": self.emergency_stops,
                "consecutive_failures": self.consecutive_failures
            },
            "readiness_check": readiness_check,
            "phase_results": self.phase_results,
            "resource_impact_summary": self.resource_monitor.get_summary_report(),
            "recommendations": self.generate_campaign_recommendations()
        }
        
        return report
    
    def generate_campaign_recommendations(self) -> List[str]:
        """Generate recommendations for future testing"""
        recommendations = []
        
        successful_phases = sum(1 for phase in self.phase_results if phase.get("success", False))
        total_phases = len(self.phase_results)
        
        if total_phases == 0:
            recommendations.append("No phases completed - investigate system issues")
        elif successful_phases == total_phases:
            recommendations.append("All phases successful - system is highly stable")
            recommendations.append("Consider increasing load levels for future testing")
        elif successful_phases / total_phases > 0.8:
            recommendations.append("Most phases successful - system is stable under load")
        else:
            recommendations.append("Multiple phase failures - investigate system bottlenecks")
        
        if self.consecutive_failures > 0:
            recommendations.append(f"Address issues causing {self.consecutive_failures} consecutive failures")
        
        if self.baseline_health_score and self.baseline_health_score < 80:
            recommendations.append("Improve baseline system health before future testing")
        
        return recommendations
    
    def save_campaign_report(self, report: Dict[str, Any]):
        """Save final campaign report"""
        results_dir = Path(__file__).parent.parent / "results_summary"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = results_dir / f"autonomous_campaign_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Campaign report saved to {report_file}")
            
            # Also create a human-readable summary
            summary_file = results_dir / f"campaign_summary_{timestamp}.txt"
            self.create_human_readable_summary(report, summary_file)
            
        except Exception as e:
            self.logger.error(f"Error saving campaign report: {e}")
    
    def create_human_readable_summary(self, report: Dict[str, Any], summary_file: Path):
        """Create human-readable summary of the campaign"""
        try:
            campaign_info = report.get("campaign_info", {})
            summary = report.get("campaign_summary", {})
            
            with open(summary_file, 'w') as f:
                f.write("AUTONOMOUS STRESS TESTING CAMPAIGN SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("CAMPAIGN OVERVIEW:\n")
                f.write(f"Start Time: {campaign_info.get('start_time', 'Unknown')}\n")
                f.write(f"End Time: {campaign_info.get('end_time', 'Unknown')}\n")
                f.write(f"Duration: {campaign_info.get('duration_hours', 0):.1f} hours\n")
                f.write(f"Baseline Health Score: {campaign_info.get('baseline_health_score', 0)}/100\n\n")
                
                f.write("RESULTS SUMMARY:\n")
                f.write(f"Phases Attempted: {summary.get('total_phases_attempted', 0)}\n")
                f.write(f"Phases Successful: {summary.get('successful_phases', 0)}\n")
                f.write(f"Phase Success Rate: {summary.get('phase_success_rate', 0):.1f}%\n")
                f.write(f"Total Operations: {summary.get('total_operations', 0)}\n")
                f.write(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.1f}%\n")
                f.write(f"Performance Trend: {summary.get('performance_trend', 'Unknown')}\n\n")
                
                if summary.get('emergency_stops', 0) > 0:
                    f.write(f"Emergency Stops: {summary.get('emergency_stops', 0)}\n")
                
                if summary.get('consecutive_failures', 0) > 0:
                    f.write(f"Consecutive Failures: {summary.get('consecutive_failures', 0)}\n")
                
                recommendations = report.get("recommendations", [])
                if recommendations:
                    f.write("\nRECOMMENDATIONS:\n")
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                
                f.write(f"\nDetailed reports available in JSON format.\n")
            
            self.logger.info(f"Human-readable summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating human-readable summary: {e}")


async def main():
    """Main entry point"""
    print("Starting Autonomous Stress Testing Campaign...")
    
    # Configure campaign for overnight operation
    campaign_config = CampaignConfig(
        total_duration_hours=8,  # 8-hour overnight run
        safety_check_interval_minutes=15,
        baseline_health_requirement=70,
        max_consecutive_failures=3,
        emergency_cooldown_minutes=30,
        progressive_escalation=True,
        auto_recovery=True
    )
    
    runner = AutonomousTestRunner(campaign_config)
    
    try:
        # Run the autonomous campaign
        campaign_report = await runner.run_autonomous_campaign()
        
        # Display final summary
        campaign_info = campaign_report.get("campaign_info", {})
        summary = campaign_report.get("campaign_summary", {})
        
        print(f"\nAutonomous Campaign Complete!")
        print(f"Duration: {campaign_info.get('duration_hours', 0):.1f} hours")
        print(f"Phases Completed: {summary.get('successful_phases', 0)}/{summary.get('total_phases_attempted', 0)}")
        print(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
        print(f"Performance Trend: {summary.get('performance_trend', 'Unknown')}")
        
        recommendations = campaign_report.get("recommendations", [])
        if recommendations:
            print(f"\nKey Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"  - {rec}")
        
        return campaign_report
        
    except Exception as e:
        print(f"Error during autonomous campaign: {e}")
        return None


if __name__ == "__main__":
    # Run the autonomous campaign
    asyncio.run(main())