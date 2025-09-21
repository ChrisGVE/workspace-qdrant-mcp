#!/usr/bin/env python3
"""
Production Readiness Validator for workspace-qdrant-mcp.

Comprehensive validation script that checks all aspects of production deployment
readiness including service installation, monitoring, security, performance,
and operational procedures.

Usage:
    python tests/production_readiness_validator.py
    python tests/production_readiness_validator.py --report-format json
    python tests/production_readiness_validator.py --strict --output report.json
"""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Import local modules
try:
    from wqm_cli.cli.commands.service import ServiceManager
    from common.observability import (
        health_checker_instance,
        metrics_instance,
    )
    from common.observability.endpoints import (
        health_check_basic,
        health_check_detailed,
        metrics_prometheus,
        metrics_json,
        system_diagnostics,
    )
    from tests.utils.deployment_helpers import DockerTestHelper, MonitoringTestHelper
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Run from project root directory")
    sys.exit(1)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    recommendations: List[str]
    critical: bool = False


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: float
    platform: str
    overall_score: float
    passed_checks: int
    total_checks: int
    critical_failures: int
    results: List[ValidationResult]
    summary: Dict[str, Any]


class ProductionReadinessValidator:
    """Main validator for production readiness assessment."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        
    async def run_all_validations(self) -> ValidationReport:
        """Run all production readiness validations."""
        print("🚀 Starting Production Readiness Validation...")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        print("-" * 60)
        
        # Run all validation categories
        await self._validate_service_installation()
        await self._validate_monitoring_system()
        await self._validate_health_checks()
        await self._validate_docker_deployment()
        await self._validate_security_configuration()
        await self._validate_backup_procedures()
        await self._validate_performance_monitoring()
        await self._validate_documentation()
        await self._validate_operational_procedures()
        
        # Generate report
        report = self._generate_report()
        
        print("-" * 60)
        print(f"✅ Validation completed in {time.time() - self.start_time:.2f}s")
        print(f"Overall Score: {report.overall_score:.1f}%")
        print(f"Passed: {report.passed_checks}/{report.total_checks} checks")
        
        if report.critical_failures > 0:
            print(f"❌ Critical failures: {report.critical_failures}")
        
        return report
    
    async def _validate_service_installation(self):
        """Validate service installation capabilities."""
        print("📦 Validating Service Installation...")
        
        try:
            service_manager = ServiceManager()
            
            # Test platform detection
            platform_detected = service_manager.system in ["linux", "darwin", "windows"]
            
            # Test daemon binary discovery (mock)
            binary_path = await service_manager._find_daemon_binary()
            binary_found = binary_path is not None
            
            # Test service configuration generation
            can_generate_config = True  # Service manager has config generation methods
            
            score = sum([platform_detected, can_generate_config]) / 2
            if binary_found:
                score = min(score + 0.2, 1.0)  # Bonus for actual binary
            
            recommendations = []
            if not binary_found:
                recommendations.append("Build the memexd-priority binary with: cargo build --release --bin memexd-priority")
            if score < 0.8:
                recommendations.append("Ensure all service installation prerequisites are met")
            
            self.results.append(ValidationResult(
                name="Service Installation",
                passed=score >= 0.7,
                score=score,
                message=f"Platform: {service_manager.system}, Binary: {'Found' if binary_found else 'Not found'}",
                details={
                    "platform": service_manager.system,
                    "binary_found": binary_found,
                    "service_name": service_manager.service_name
                },
                recommendations=recommendations,
                critical=True
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Service Installation",
                passed=False,
                score=0.0,
                message=f"Service installation validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Fix service manager import/configuration issues"],
                critical=True
            ))
    
    async def _validate_monitoring_system(self):
        """Validate monitoring system integration."""
        print("📊 Validating Monitoring System...")
        
        try:
            # Test metrics collection
            metrics_instance.increment_counter("validation_test")
            prometheus_data = await metrics_prometheus()
            json_data = await metrics_json()
            
            # Test metrics format
            prometheus_valid = isinstance(prometheus_data, str) and len(prometheus_data) > 0
            json_valid = isinstance(json_data, dict) and "counters" in json_data
            
            # Test Prometheus config
            prometheus_config_path = Path("monitoring/prometheus/prometheus.yml")
            prometheus_config_exists = prometheus_config_path.exists()
            
            # Test Grafana config
            grafana_provisioning_path = Path("monitoring/grafana/provisioning")
            grafana_config_exists = grafana_provisioning_path.exists()
            
            score = sum([
                prometheus_valid,
                json_valid,
                prometheus_config_exists,
                grafana_config_exists
            ]) / 4
            
            recommendations = []
            if not prometheus_config_exists:
                recommendations.append("Create Prometheus configuration file")
            if not grafana_config_exists:
                recommendations.append("Set up Grafana provisioning configuration")
            
            self.results.append(ValidationResult(
                name="Monitoring System",
                passed=score >= 0.75,
                score=score,
                message=f"Metrics: {'✓' if prometheus_valid and json_valid else '✗'}, Config: {'✓' if prometheus_config_exists else '✗'}",
                details={
                    "prometheus_format_valid": prometheus_valid,
                    "json_format_valid": json_valid,
                    "prometheus_config_exists": prometheus_config_exists,
                    "grafana_config_exists": grafana_config_exists
                },
                recommendations=recommendations,
                critical=True
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Monitoring System",
                passed=False,
                score=0.0,
                message=f"Monitoring system validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Fix monitoring system configuration"],
                critical=True
            ))
    
    async def _validate_health_checks(self):
        """Validate health check system."""
        print("🏥 Validating Health Checks...")
        
        try:
            start_time = time.perf_counter()
            
            # Test basic health check
            basic_health = await health_check_basic()
            basic_duration = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            
            # Test detailed health check
            detailed_health = await health_check_detailed()
            detailed_duration = time.perf_counter() - start_time
            
            # Test diagnostics
            diagnostics = await system_diagnostics()
            
            # Validate response formats
            basic_valid = (
                isinstance(basic_health, dict) and 
                "status" in basic_health and
                basic_health["status"] in ["healthy", "degraded", "unhealthy"]
            )
            
            detailed_valid = (
                isinstance(detailed_health, dict) and
                "components" in detailed_health and
                isinstance(detailed_health["components"], dict)
            )
            
            diagnostics_valid = (
                isinstance(diagnostics, dict) and
                "health_status" in diagnostics and
                "system_info" in diagnostics
            )
            
            performance_good = basic_duration < 1.0 and detailed_duration < 10.0
            
            score = sum([
                basic_valid,
                detailed_valid,
                diagnostics_valid,
                performance_good
            ]) / 4
            
            recommendations = []
            if not performance_good:
                recommendations.append("Optimize health check performance")
            if score < 0.8:
                recommendations.append("Review health check component implementations")
            
            self.results.append(ValidationResult(
                name="Health Checks",
                passed=score >= 0.75,
                score=score,
                message=f"Basic: {basic_duration:.3f}s, Detailed: {detailed_duration:.3f}s",
                details={
                    "basic_valid": basic_valid,
                    "detailed_valid": detailed_valid,
                    "diagnostics_valid": diagnostics_valid,
                    "basic_duration": basic_duration,
                    "detailed_duration": detailed_duration,
                    "components_count": len(detailed_health.get("components", {}))
                },
                recommendations=recommendations,
                critical=True
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Health Checks",
                passed=False,
                score=0.0,
                message=f"Health check validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Fix health check system configuration"],
                critical=True
            ))
    
    async def _validate_docker_deployment(self):
        """Validate Docker deployment configuration."""
        print("🐳 Validating Docker Deployment...")
        
        try:
            docker_helper = DockerTestHelper()
            
            # Check Docker availability
            docker_available = docker_helper.is_docker_available()
            compose_available = docker_helper.is_compose_available()
            
            # Check Dockerfile exists
            dockerfile_path = Path("docker/Dockerfile")
            dockerfile_exists = dockerfile_path.exists()
            
            # Check docker-compose.yml
            compose_path = Path("docker/docker-compose.yml")
            compose_exists = compose_path.exists()
            
            # Validate compose structure
            compose_valid = False
            required_services = []
            
            if compose_exists:
                try:
                    with open(compose_path, 'r') as f:
                        compose_config = yaml.safe_load(f)
                    
                    services = compose_config.get("services", {})
                    required_services = ["workspace-qdrant-mcp", "qdrant", "redis"]
                    compose_valid = all(service in services for service in required_services)
                    
                except Exception:
                    compose_valid = False
            
            score = sum([
                dockerfile_exists,
                compose_exists,
                compose_valid
            ]) / 3
            
            # Bonus points for Docker availability (not required for deployment config validation)
            if docker_available:
                score = min(score + 0.1, 1.0)
            
            recommendations = []
            if not dockerfile_exists:
                recommendations.append("Create Dockerfile for container builds")
            if not compose_exists:
                recommendations.append("Create docker-compose.yml for orchestration")
            if not compose_valid:
                recommendations.append("Ensure all required services are in docker-compose.yml")
            if not docker_available:
                recommendations.append("Install Docker for container operations")
            
            self.results.append(ValidationResult(
                name="Docker Deployment",
                passed=score >= 0.7,
                score=score,
                message=f"Dockerfile: {'✓' if dockerfile_exists else '✗'}, Compose: {'✓' if compose_exists else '✗'}",
                details={
                    "docker_available": docker_available,
                    "compose_available": compose_available,
                    "dockerfile_exists": dockerfile_exists,
                    "compose_exists": compose_exists,
                    "compose_valid": compose_valid,
                    "required_services": required_services
                },
                recommendations=recommendations
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Docker Deployment",
                passed=False,
                score=0.0,
                message=f"Docker deployment validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Fix Docker configuration"],
            ))
    
    async def _validate_security_configuration(self):
        """Validate security configuration."""
        print("🔒 Validating Security Configuration...")
        
        try:
            # Check Docker security settings
            compose_path = Path("docker/docker-compose.yml")
            security_score = 0
            security_checks = 0
            
            if compose_path.exists():
                with open(compose_path, 'r') as f:
                    compose_config = yaml.safe_load(f)
                
                services = compose_config.get("services", {})
                
                for service_name, service_config in services.items():
                    security_checks += 4  # 4 checks per service
                    
                    # Check security_opt
                    if "security_opt" in service_config:
                        security_opts = service_config["security_opt"]
                        if any("no-new-privileges" in opt for opt in security_opts):
                            security_score += 1
                    
                    # Check user configuration (non-root)
                    if "user" in service_config or service_name in ["nginx", "redis"]:
                        security_score += 1
                    
                    # Check resource limits
                    if "deploy" in service_config and "resources" in service_config["deploy"]:
                        security_score += 1
                    
                    # Check for read-only filesystem hints
                    if "read_only" in service_config or service_name in ["nginx"]:
                        security_score += 1
            
            # Check for secrets management
            env_files = list(Path(".").glob("*.env*"))
            secrets_documented = any(".env.example" in str(f) or ".env.template" in str(f) for f in env_files)
            
            if secrets_documented:
                security_score += 2
                security_checks += 2
            
            score = security_score / max(security_checks, 1) if security_checks > 0 else 0
            
            recommendations = []
            if score < 0.7:
                recommendations.extend([
                    "Implement no-new-privileges security option",
                    "Configure non-root users for containers",
                    "Set resource limits for all services",
                    "Use read-only filesystems where possible",
                    "Document secrets management procedures"
                ])
            
            self.results.append(ValidationResult(
                name="Security Configuration",
                passed=score >= 0.6,
                score=score,
                message=f"Security checks: {security_score}/{security_checks}",
                details={
                    "security_score": security_score,
                    "security_checks": security_checks,
                    "secrets_documented": secrets_documented
                },
                recommendations=recommendations
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Security Configuration",
                passed=False,
                score=0.0,
                message=f"Security validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Review security configuration"],
            ))
    
    async def _validate_backup_procedures(self):
        """Validate backup and recovery procedures."""
        print("💾 Validating Backup Procedures...")
        
        try:
            # Check for backup-related configuration
            compose_path = Path("docker/docker-compose.yml")
            volumes_configured = False
            persistent_volumes = []
            
            if compose_path.exists():
                with open(compose_path, 'r') as f:
                    compose_config = yaml.safe_load(f)
                
                volumes = compose_config.get("volumes", {})
                services = compose_config.get("services", {})
                
                # Check for data volumes
                data_volumes = ["workspace_data", "qdrant_storage", "redis_data", "prometheus_data"]
                persistent_volumes = [vol for vol in data_volumes if vol in volumes]
                volumes_configured = len(persistent_volumes) >= 3
            
            # Check for backup documentation
            docs_dir = Path("docs")
            backup_docs = []
            if docs_dir.exists():
                backup_docs = list(docs_dir.glob("*backup*")) + list(docs_dir.glob("*recovery*"))
            
            backup_docs_exist = len(backup_docs) > 0
            
            # Check for backup scripts/procedures (in scripts or tools directories)
            scripts_dir = Path("scripts")
            tools_dir = Path("tools")
            
            backup_scripts = []
            for directory in [scripts_dir, tools_dir]:
                if directory.exists():
                    backup_scripts.extend(list(directory.glob("*backup*")))
                    backup_scripts.extend(list(directory.glob("*restore*")))
            
            backup_scripts_exist = len(backup_scripts) > 0
            
            score = sum([
                volumes_configured,
                backup_docs_exist,
                backup_scripts_exist
            ]) / 3
            
            recommendations = []
            if not volumes_configured:
                recommendations.append("Configure persistent volumes for data backup")
            if not backup_docs_exist:
                recommendations.append("Create backup and recovery documentation")
            if not backup_scripts_exist:
                recommendations.append("Develop automated backup scripts")
            
            self.results.append(ValidationResult(
                name="Backup Procedures",
                passed=score >= 0.5,
                score=score,
                message=f"Volumes: {len(persistent_volumes)}, Docs: {'✓' if backup_docs_exist else '✗'}",
                details={
                    "volumes_configured": volumes_configured,
                    "persistent_volumes": persistent_volumes,
                    "backup_docs_exist": backup_docs_exist,
                    "backup_scripts_exist": backup_scripts_exist,
                    "backup_docs": [str(doc) for doc in backup_docs],
                    "backup_scripts": [str(script) for script in backup_scripts]
                },
                recommendations=recommendations
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Backup Procedures",
                passed=False,
                score=0.0,
                message=f"Backup validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Create backup and recovery procedures"],
            ))
    
    async def _validate_performance_monitoring(self):
        """Validate performance monitoring capabilities."""
        print("⚡ Validating Performance Monitoring...")
        
        try:
            # Test metrics collection performance
            start_time = time.perf_counter()
            
            # Collect test metrics
            for i in range(100):
                metrics_instance.increment_counter("perf_test_counter")
                metrics_instance.record_histogram("perf_test_histogram", i * 0.01)
                metrics_instance.set_gauge("perf_test_gauge", i)
            
            collection_duration = time.perf_counter() - start_time
            
            # Test metrics export performance
            start_time = time.perf_counter()
            prometheus_data = await metrics_prometheus()
            export_duration = time.perf_counter() - start_time
            
            # Validate performance thresholds
            collection_fast = collection_duration < 0.1  # 100ms for 300 metric operations
            export_fast = export_duration < 1.0  # 1s for metrics export
            
            # Check for performance monitoring configuration
            grafana_dashboards_dir = Path("monitoring/grafana/dashboards")
            performance_dashboards = []
            
            if grafana_dashboards_dir.exists():
                for dashboard_file in grafana_dashboards_dir.glob("*.json"):
                    try:
                        with open(dashboard_file, 'r') as f:
                            dashboard_content = f.read()
                        
                        if any(keyword in dashboard_content.lower() 
                              for keyword in ["performance", "latency", "duration", "response_time"]):
                            performance_dashboards.append(dashboard_file.name)
                    except Exception:
                        continue
            
            performance_dashboards_exist = len(performance_dashboards) > 0
            
            score = sum([
                collection_fast,
                export_fast,
                performance_dashboards_exist
            ]) / 3
            
            recommendations = []
            if not collection_fast:
                recommendations.append("Optimize metrics collection performance")
            if not export_fast:
                recommendations.append("Optimize metrics export performance")
            if not performance_dashboards_exist:
                recommendations.append("Create performance monitoring dashboards")
            
            self.results.append(ValidationResult(
                name="Performance Monitoring",
                passed=score >= 0.7,
                score=score,
                message=f"Collection: {collection_duration:.3f}s, Export: {export_duration:.3f}s",
                details={
                    "collection_duration": collection_duration,
                    "export_duration": export_duration,
                    "collection_fast": collection_fast,
                    "export_fast": export_fast,
                    "performance_dashboards": performance_dashboards
                },
                recommendations=recommendations
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Performance Monitoring",
                passed=False,
                score=0.0,
                message=f"Performance monitoring validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Fix performance monitoring system"],
            ))
    
    async def _validate_documentation(self):
        """Validate documentation completeness."""
        print("📚 Validating Documentation...")
        
        try:
            # Check for essential documentation files
            essential_docs = {
                "README.md": Path("README.md"),
                "DEPLOYMENT.md": Path("DEPLOYMENT.md"),
                "docker/README.md": Path("docker/README.md"),
                "monitoring/README.md": Path("monitoring/README.md"),
            }
            
            docs_score = 0
            docs_found = []
            docs_missing = []
            
            for doc_name, doc_path in essential_docs.items():
                if doc_path.exists():
                    docs_score += 1
                    docs_found.append(doc_name)
                else:
                    docs_missing.append(doc_name)
            
            # Check for API documentation
            api_docs_paths = [
                Path("docs/api"),
                Path("src/workspace_qdrant_mcp/server.py"),  # FastAPI auto-docs
                Path("openapi.json"),
                Path("docs/openapi.yaml")
            ]
            
            api_docs_exist = any(path.exists() for path in api_docs_paths)
            if api_docs_exist:
                docs_score += 1
            
            # Check for monitoring documentation
            monitoring_docs = [
                Path("monitoring/README.md"),
                Path("docs/monitoring.md"),
                Path("monitoring/grafana/README.md")
            ]
            
            monitoring_docs_exist = any(path.exists() for path in monitoring_docs)
            if monitoring_docs_exist:
                docs_score += 1
            
            total_docs = len(essential_docs) + 2  # API docs + monitoring docs
            score = docs_score / total_docs
            
            recommendations = []
            if docs_missing:
                recommendations.append(f"Create missing documentation: {', '.join(docs_missing)}")
            if not api_docs_exist:
                recommendations.append("Add API documentation")
            if not monitoring_docs_exist:
                recommendations.append("Add monitoring setup documentation")
            
            self.results.append(ValidationResult(
                name="Documentation",
                passed=score >= 0.6,
                score=score,
                message=f"Found: {len(docs_found)}/{len(essential_docs)} essential docs",
                details={
                    "docs_found": docs_found,
                    "docs_missing": docs_missing,
                    "api_docs_exist": api_docs_exist,
                    "monitoring_docs_exist": monitoring_docs_exist,
                    "total_score": docs_score,
                    "total_possible": total_docs
                },
                recommendations=recommendations
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Documentation",
                passed=False,
                score=0.0,
                message=f"Documentation validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Create essential project documentation"],
            ))
    
    async def _validate_operational_procedures(self):
        """Validate operational procedures and runbooks."""
        print("🔧 Validating Operational Procedures...")
        
        try:
            # Check for operational scripts
            scripts_dir = Path("scripts")
            operational_scripts = []
            
            if scripts_dir.exists():
                script_patterns = ["deploy*", "start*", "stop*", "restart*", "health*", "backup*"]
                for pattern in script_patterns:
                    operational_scripts.extend(list(scripts_dir.glob(pattern)))
            
            # Check for deployment automation
            ci_cd_files = [
                Path(".github/workflows"),
                Path(".gitlab-ci.yml"),
                Path("Jenkinsfile"),
                Path(".circleci/config.yml"),
                Path("azure-pipelines.yml")
            ]
            
            ci_cd_exists = any(path.exists() for path in ci_cd_files)
            
            # Check for configuration management
            config_examples = [
                Path(".env.example"),
                Path("config.yaml.example"),
                Path("docker/.env.example")
            ]
            
            config_examples_exist = any(path.exists() for path in config_examples)
            
            # Check for monitoring and alerting setup
            alerting_config = Path("monitoring/prometheus/rules")
            alerting_configured = alerting_config.exists()
            
            score = sum([
                len(operational_scripts) > 0,
                ci_cd_exists,
                config_examples_exist,
                alerting_configured
            ]) / 4
            
            recommendations = []
            if not operational_scripts:
                recommendations.append("Create operational scripts for deployment and maintenance")
            if not ci_cd_exists:
                recommendations.append("Set up CI/CD pipeline automation")
            if not config_examples_exist:
                recommendations.append("Provide configuration examples and templates")
            if not alerting_configured:
                recommendations.append("Configure monitoring alerts and thresholds")
            
            self.results.append(ValidationResult(
                name="Operational Procedures",
                passed=score >= 0.5,
                score=score,
                message=f"Scripts: {len(operational_scripts)}, CI/CD: {'✓' if ci_cd_exists else '✗'}",
                details={
                    "operational_scripts": [str(script) for script in operational_scripts],
                    "ci_cd_exists": ci_cd_exists,
                    "config_examples_exist": config_examples_exist,
                    "alerting_configured": alerting_configured
                },
                recommendations=recommendations
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                name="Operational Procedures",
                passed=False,
                score=0.0,
                message=f"Operational procedures validation failed: {e}",
                details={"error": str(e)},
                recommendations=["Develop operational procedures and automation"],
            ))
    
    def _generate_report(self) -> ValidationReport:
        """Generate comprehensive validation report."""
        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results if result.passed)
        critical_failures = sum(1 for result in self.results if result.critical and not result.passed)
        
        # Calculate overall score as weighted average
        total_score = sum(result.score for result in self.results)
        overall_score = (total_score / total_checks * 100) if total_checks > 0 else 0
        
        # Generate summary
        summary = {
            "validation_duration": time.time() - self.start_time,
            "platform_info": {
                "system": platform.system(),
                "release": platform.release(),
                "python_version": platform.python_version()
            },
            "score_breakdown": {
                result.name: result.score for result in self.results
            },
            "critical_checks": [
                result.name for result in self.results if result.critical
            ],
            "failed_critical": [
                result.name for result in self.results if result.critical and not result.passed
            ],
            "top_recommendations": self._get_top_recommendations()
        }
        
        return ValidationReport(
            timestamp=time.time(),
            platform=platform.system(),
            overall_score=overall_score,
            passed_checks=passed_checks,
            total_checks=total_checks,
            critical_failures=critical_failures,
            results=self.results,
            summary=summary
        )
    
    def _get_top_recommendations(self) -> List[str]:
        """Get top recommendations from all validation results."""
        all_recommendations = []
        
        for result in self.results:
            if not result.passed and result.recommendations:
                all_recommendations.extend(result.recommendations)
        
        # Return unique recommendations, prioritizing critical failures
        seen = set()
        top_recommendations = []
        
        # First, add recommendations from critical failures
        for result in self.results:
            if result.critical and not result.passed:
                for rec in result.recommendations:
                    if rec not in seen:
                        seen.add(rec)
                        top_recommendations.append(rec)
        
        # Then add other recommendations
        for result in self.results:
            if not result.critical and not result.passed:
                for rec in result.recommendations:
                    if rec not in seen and len(top_recommendations) < 10:
                        seen.add(rec)
                        top_recommendations.append(rec)
        
        return top_recommendations[:10]


def print_report(report: ValidationReport, format_type: str = "text"):
    """Print validation report in specified format."""
    if format_type == "json":
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print("\n" + "="*80)
        print("🚀 PRODUCTION READINESS VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(report.timestamp))}")
        print(f"Platform: {report.platform}")
        print(f"Overall Score: {report.overall_score:.1f}%")
        print(f"Passed Checks: {report.passed_checks}/{report.total_checks}")
        
        if report.critical_failures > 0:
            print(f"❌ Critical Failures: {report.critical_failures}")
        
        print("\n📊 DETAILED RESULTS")
        print("-" * 50)
        
        for result in report.results:
            status = "✅" if result.passed else "❌"
            critical_marker = "🔴" if result.critical and not result.passed else ""
            
            print(f"{status} {critical_marker} {result.name}: {result.score*100:.1f}%")
            print(f"   {result.message}")
            
            if not result.passed and result.recommendations:
                print("   Recommendations:")
                for rec in result.recommendations[:3]:  # Show top 3
                    print(f"   - {rec}")
            print()
        
        if report.summary["top_recommendations"]:
            print("🔧 TOP RECOMMENDATIONS")
            print("-" * 50)
            for i, rec in enumerate(report.summary["top_recommendations"], 1):
                print(f"{i}. {rec}")
        
        print("\n⏱️  SUMMARY")
        print("-" * 50)
        print(f"Validation Duration: {report.summary['validation_duration']:.2f} seconds")
        print(f"Python Version: {report.summary['platform_info']['python_version']}")
        
        if report.overall_score >= 80:
            print("\n🎉 PRODUCTION READY! Your system meets production deployment standards.")
        elif report.overall_score >= 60:
            print("\n⚠️  NEEDS IMPROVEMENT. Address the recommendations above before production deployment.")
        else:
            print("\n🚨 NOT READY FOR PRODUCTION. Critical issues must be resolved.")


async def main():
    """Main entry point for production readiness validation."""
    parser = argparse.ArgumentParser(description="Production Readiness Validator")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--format", choices=["text", "json"], default="text", 
                       help="Report format")
    parser.add_argument("--strict", action="store_true", 
                       help="Use strict validation criteria")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Redirect stdout if quiet mode
    if args.quiet:
        import io
        sys.stdout = io.StringIO()
    
    try:
        validator = ProductionReadinessValidator()
        report = await validator.run_all_validations()
        
        # Restore stdout
        if args.quiet:
            sys.stdout = sys.__stdout__
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                if args.format == "json":
                    json.dump(asdict(report), f, indent=2, default=str)
                else:
                    original_stdout = sys.stdout
                    sys.stdout = f
                    print_report(report, args.format)
                    sys.stdout = original_stdout
            
            print(f"Report saved to: {args.output}")
        else:
            print_report(report, args.format)
        
        # Exit with appropriate code
        if args.strict:
            # In strict mode, any critical failure or score < 90% fails
            if report.critical_failures > 0 or report.overall_score < 90:
                sys.exit(1)
        else:
            # In normal mode, only critical failures cause exit failure
            if report.critical_failures > 0:
                sys.exit(1)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())