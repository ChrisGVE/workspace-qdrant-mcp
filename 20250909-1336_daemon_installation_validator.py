#!/usr/bin/env python3
"""
Daemon Service Installation Documentation Validator

This script validates that the daemon service installation guide and related files
are comprehensive, accurate, and complete.
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class DaemonInstallationValidator:
    """Validates daemon service installation documentation and files."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> Dict[str, any]:
        """Run all validation checks."""
        print("🔍 Validating daemon service installation documentation...")
        print("=" * 60)
        
        # Core validation checks
        self.validate_documentation_structure()
        self.validate_service_configurations()
        self.validate_installation_scripts()
        self.validate_platform_coverage()
        self.validate_security_considerations()
        self.validate_troubleshooting_coverage()
        
        # Generate summary
        self.generate_validation_summary()
        
        return self.validation_results
    
    def validate_documentation_structure(self):
        """Validate the main documentation structure."""
        print("📚 Validating documentation structure...")
        
        guide_path = self.project_root / "20250909-1336_daemon_service_installation_guide.md"
        
        if not guide_path.exists():
            self.errors.append("Main installation guide not found")
            return
            
        content = guide_path.read_text()
        
        # Required sections
        required_sections = [
            "# Daemon Service Installation and Startup Guide",
            "## Overview",
            "## Prerequisites", 
            "## Quick Installation",
            "## Platform-Specific Installation",
            "### Linux (systemd)",
            "### macOS (launchd)",
            "### Windows Service",
            "## Service Management Commands",
            "## Post-Installation Verification",
            "## Troubleshooting",
            "## Security Considerations",
            "## Auto-Start Configuration"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            self.errors.append(f"Missing documentation sections: {missing_sections}")
        else:
            print("✅ All required documentation sections present")
            
        # Check for platform-specific details
        platforms = ["Linux", "macOS", "Windows"]
        for platform_name in platforms:
            if platform_name.lower() not in content.lower():
                self.warnings.append(f"Limited {platform_name} coverage in documentation")
                
        # Check for troubleshooting depth
        troubleshooting_topics = [
            "Service Won't Start",
            "High Memory Usage", 
            "Permission Errors",
            "IPC Connection Failed"
        ]
        
        missing_troubleshooting = []
        for topic in troubleshooting_topics:
            if topic not in content:
                missing_troubleshooting.append(topic)
                
        if missing_troubleshooting:
            self.warnings.append(f"Missing troubleshooting topics: {missing_troubleshooting}")
            
        self.validation_results["documentation_structure"] = {
            "guide_exists": guide_path.exists(),
            "sections_complete": len(missing_sections) == 0,
            "troubleshooting_comprehensive": len(missing_troubleshooting) == 0
        }
        
    def validate_service_configurations(self):
        """Validate service configuration files."""
        print("⚙️  Validating service configurations...")
        
        configs_dir = self.project_root / "20250909-1336_service_configs"
        
        # Expected configuration files
        expected_configs = {
            "systemd/memexd.service": self.validate_systemd_config,
            "launchd/com.workspace-qdrant.memexd.plist": self.validate_launchd_config,
            "config/memexd.toml": self.validate_toml_config
        }
        
        config_results = {}
        
        for config_path, validator in expected_configs.items():
            full_path = configs_dir / config_path
            if full_path.exists():
                try:
                    result = validator(full_path)
                    config_results[config_path] = result
                    if result["valid"]:
                        print(f"✅ {config_path} is valid")
                    else:
                        print(f"❌ {config_path} has issues: {result['issues']}")
                except Exception as e:
                    self.errors.append(f"Failed to validate {config_path}: {e}")
                    config_results[config_path] = {"valid": False, "error": str(e)}
            else:
                self.errors.append(f"Missing configuration file: {config_path}")
                config_results[config_path] = {"valid": False, "error": "File not found"}
                
        self.validation_results["service_configurations"] = config_results
        
    def validate_systemd_config(self, path: Path) -> Dict[str, any]:
        """Validate systemd service file."""
        content = path.read_text()
        issues = []
        
        # Required sections
        required_sections = ["[Unit]", "[Service]", "[Install]"]
        for section in required_sections:
            if section not in content:
                issues.append(f"Missing section: {section}")
                
        # Security settings
        security_settings = [
            "NoNewPrivileges=true",
            "ProtectSystem=strict",
            "ProtectHome=true",
            "PrivateTmp=true"
        ]
        
        missing_security = []
        for setting in security_settings:
            if setting not in content:
                missing_security.append(setting)
                
        if missing_security:
            issues.append(f"Missing security settings: {missing_security}")
            
        # Resource limits
        if "MemoryMax=" not in content:
            issues.append("Missing memory limits")
            
        # Restart policy
        if "Restart=always" not in content:
            issues.append("Missing restart policy")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "has_security": len(missing_security) == 0,
            "has_resource_limits": "MemoryMax=" in content
        }
        
    def validate_launchd_config(self, path: Path) -> Dict[str, any]:
        """Validate macOS LaunchDaemon plist."""
        content = path.read_text()
        issues = []
        
        # Required keys
        required_keys = [
            "<key>Label</key>",
            "<key>ProgramArguments</key>",
            "<key>RunAtLoad</key>",
            "<key>KeepAlive</key>"
        ]
        
        for key in required_keys:
            if key not in content:
                issues.append(f"Missing key: {key}")
                
        # Security considerations
        if "<key>UserName</key>" not in content:
            issues.append("Missing dedicated user configuration")
            
        # Resource limits
        if "SoftResourceLimits" not in content:
            issues.append("Missing resource limits")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "has_user": "<key>UserName</key>" in content,
            "has_keepalive": "<key>KeepAlive</key>" in content
        }
        
    def validate_toml_config(self, path: Path) -> Dict[str, any]:
        """Validate TOML configuration file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return {"valid": False, "error": "TOML library not available"}
                
        content = path.read_text()
        issues = []
        
        try:
            config = tomllib.loads(content)
            
            # Required sections
            required_sections = [
                "daemon", "qdrant", "embedding", "workspace", "logging", "security", "performance"
            ]
            
            for section in required_sections:
                if section not in config:
                    issues.append(f"Missing section: {section}")
                    
            # Validate daemon section
            if "daemon" in config:
                daemon_config = config["daemon"]
                required_daemon_keys = ["port", "log_level", "worker_threads"]
                for key in required_daemon_keys:
                    if key not in daemon_config:
                        issues.append(f"Missing daemon.{key}")
                        
            # Validate security section
            if "security" in config:
                security_config = config["security"]
                if security_config.get("bind_address") not in ["127.0.0.1", "localhost"]:
                    issues.append("Security risk: bind_address should be localhost")
                    
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "sections_complete": len([s for s in required_sections if s in config]) == len(required_sections)
            }
            
        except Exception as e:
            return {"valid": False, "error": f"TOML parsing error: {e}"}
            
    def validate_installation_scripts(self):
        """Validate installation scripts."""
        print("📜 Validating installation scripts...")
        
        scripts_dir = self.project_root / "20250909-1336_service_configs" / "scripts"
        
        expected_scripts = [
            "install-linux.sh",
            "install-macos.sh"
        ]
        
        windows_script = self.project_root / "20250909-1336_service_configs" / "windows" / "install-service.ps1"
        
        script_results = {}
        
        # Validate Unix scripts
        for script_name in expected_scripts:
            script_path = scripts_dir / script_name
            if script_path.exists():
                result = self.validate_shell_script(script_path)
                script_results[script_name] = result
                if result["valid"]:
                    print(f"✅ {script_name} is valid")
                else:
                    print(f"❌ {script_name} has issues: {result['issues']}")
            else:
                self.errors.append(f"Missing installation script: {script_name}")
                script_results[script_name] = {"valid": False, "error": "File not found"}
                
        # Validate Windows script
        if windows_script.exists():
            result = self.validate_powershell_script(windows_script)
            script_results["install-service.ps1"] = result
            if result["valid"]:
                print("✅ install-service.ps1 is valid")
            else:
                print(f"❌ install-service.ps1 has issues: {result['issues']}")
        else:
            self.errors.append("Missing Windows installation script")
            script_results["install-service.ps1"] = {"valid": False, "error": "File not found"}
            
        self.validation_results["installation_scripts"] = script_results
        
    def validate_shell_script(self, path: Path) -> Dict[str, any]:
        """Validate shell script."""
        content = path.read_text()
        issues = []
        
        # Check for shebang
        if not content.startswith("#!/bin/bash"):
            issues.append("Missing proper shebang")
            
        # Check for error handling
        if "set -e" not in content:
            issues.append("Missing 'set -e' for error handling")
            
        # Check for root check
        if "root" not in content.lower() or "sudo" not in content.lower():
            issues.append("Missing root/sudo permission checks")
            
        # Check for service user creation (Linux: useradd/adduser, macOS: dscl)
        if "useradd" not in content and "adduser" not in content and "dscl" not in content:
            issues.append("Missing service user creation")
            
        # Check for directory creation
        if "mkdir" not in content:
            issues.append("Missing directory creation")
            
        # Check for permission setting
        if "chmod" not in content or "chown" not in content:
            issues.append("Missing permission configuration")
            
        # Check for service management
        if "systemctl" not in content and "launchctl" not in content:
            issues.append("Missing service management commands")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "has_error_handling": "set -e" in content,
            "has_permission_checks": "root" in content.lower()
        }
        
    def validate_powershell_script(self, path: Path) -> Dict[str, any]:
        """Validate PowerShell script."""
        content = path.read_text()
        issues = []
        
        # Check for parameter declaration
        if "param(" not in content:
            issues.append("Missing parameter declaration")
            
        # Check for admin check
        if "Administrator" not in content:
            issues.append("Missing administrator permission check")
            
        # Check for error handling
        if "try" not in content or "catch" not in content:
            issues.append("Missing try/catch error handling")
            
        # Check for service commands
        if "sc.exe" not in content:
            issues.append("Missing service management commands")
            
        # Check for directory creation
        if "New-Item" not in content:
            issues.append("Missing directory creation")
            
        # Check for permission setting
        if "AccessRule" not in content:
            issues.append("Missing permission configuration")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "has_error_handling": "try" in content and "catch" in content,
            "has_admin_check": "Administrator" in content
        }
        
    def validate_platform_coverage(self):
        """Validate platform coverage completeness."""
        print("🖥️  Validating platform coverage...")
        
        # Check if all major platforms are covered
        platforms = {
            "linux": ["systemd", "Ubuntu", "CentOS", "RHEL"],
            "macos": ["launchd", "macOS", "Darwin"],
            "windows": ["Service", "PowerShell", "Windows"]
        }
        
        guide_path = self.project_root / "20250909-1336_daemon_service_installation_guide.md"
        if guide_path.exists():
            content = guide_path.read_text().lower()
            
            platform_coverage = {}
            for platform, keywords in platforms.items():
                coverage_score = sum(1 for keyword in keywords if keyword.lower() in content)
                platform_coverage[platform] = {
                    "score": coverage_score,
                    "total_keywords": len(keywords),
                    "percentage": (coverage_score / len(keywords)) * 100
                }
                
            # Check for comprehensive coverage
            well_covered = sum(1 for p in platform_coverage.values() if p["percentage"] >= 75)
            
            self.validation_results["platform_coverage"] = {
                "platforms": platform_coverage,
                "well_covered_count": well_covered,
                "comprehensive": well_covered >= 3
            }
            
            if well_covered >= 3:
                print("✅ Comprehensive platform coverage")
            else:
                self.warnings.append("Incomplete platform coverage")
                
    def validate_security_considerations(self):
        """Validate security considerations coverage."""
        print("🔒 Validating security considerations...")
        
        security_topics = [
            "user permissions",
            "file permissions", 
            "dedicated service user",
            "resource limits",
            "bind address",
            "localhost",
            "firewall",
            "api key",
            "configuration security"
        ]
        
        guide_path = self.project_root / "20250909-1336_daemon_service_installation_guide.md"
        if guide_path.exists():
            content = guide_path.read_text().lower()
            
            covered_topics = []
            for topic in security_topics:
                if topic in content:
                    covered_topics.append(topic)
                    
            coverage_percentage = (len(covered_topics) / len(security_topics)) * 100
            
            self.validation_results["security_coverage"] = {
                "covered_topics": covered_topics,
                "total_topics": len(security_topics),
                "percentage": coverage_percentage
            }
            
            if coverage_percentage >= 80:
                print("✅ Comprehensive security coverage")
            else:
                self.warnings.append(f"Security coverage only {coverage_percentage:.1f}%")
                
    def validate_troubleshooting_coverage(self):
        """Validate troubleshooting section completeness."""
        print("🔧 Validating troubleshooting coverage...")
        
        troubleshooting_areas = [
            "service won't start",
            "memory usage",
            "permission errors", 
            "connection failed",
            "log analysis",
            "configuration errors",
            "port conflicts",
            "firewall issues"
        ]
        
        guide_path = self.project_root / "20250909-1336_daemon_service_installation_guide.md"
        if guide_path.exists():
            content = guide_path.read_text().lower()
            
            covered_areas = []
            for area in troubleshooting_areas:
                if area in content:
                    covered_areas.append(area)
                    
            coverage_percentage = (len(covered_areas) / len(troubleshooting_areas)) * 100
            
            self.validation_results["troubleshooting_coverage"] = {
                "covered_areas": covered_areas,
                "total_areas": len(troubleshooting_areas),
                "percentage": coverage_percentage
            }
            
            if coverage_percentage >= 75:
                print("✅ Comprehensive troubleshooting coverage")
            else:
                self.warnings.append(f"Troubleshooting coverage only {coverage_percentage:.1f}%")
                
    def generate_validation_summary(self):
        """Generate comprehensive validation summary."""
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)
        
        # Count issues
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        
        if total_errors == 0 and total_warnings == 0:
            print("✅ ALL VALIDATIONS PASSED!")
            overall_status = "EXCELLENT"
        elif total_errors == 0:
            print(f"⚠️  PASSED WITH {total_warnings} WARNINGS")
            overall_status = "GOOD"
        else:
            print(f"❌ FAILED WITH {total_errors} ERRORS AND {total_warnings} WARNINGS")
            overall_status = "NEEDS_IMPROVEMENT"
            
        print()
        
        # Detailed results
        if self.errors:
            print("🔴 ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
            print()
            
        if self.warnings:
            print("🟡 WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
            print()
            
        # Summary statistics
        print("📊 COVERAGE STATISTICS:")
        
        if "platform_coverage" in self.validation_results:
            pc = self.validation_results["platform_coverage"]
            print(f"  Platform Coverage: {pc['well_covered_count']}/3 platforms well covered")
            
        if "security_coverage" in self.validation_results:
            sc = self.validation_results["security_coverage"]
            print(f"  Security Coverage: {sc['percentage']:.1f}%")
            
        if "troubleshooting_coverage" in self.validation_results:
            tc = self.validation_results["troubleshooting_coverage"]
            print(f"  Troubleshooting Coverage: {tc['percentage']:.1f}%")
            
        print()
        
        # Component status
        print("🔧 COMPONENT STATUS:")
        components = [
            ("Documentation Structure", "documentation_structure"),
            ("Service Configurations", "service_configurations"), 
            ("Installation Scripts", "installation_scripts")
        ]
        
        for name, key in components:
            if key in self.validation_results:
                component_data = self.validation_results[key]
                if isinstance(component_data, dict):
                    if key == "service_configurations" or key == "installation_scripts":
                        # Count valid configurations/scripts
                        valid_count = sum(1 for v in component_data.values() if isinstance(v, dict) and v.get("valid", False))
                        total_count = len(component_data)
                        status = "✅ GOOD" if valid_count == total_count else f"⚠️  {valid_count}/{total_count}"
                    else:
                        # Boolean check
                        all_good = all(v for v in component_data.values() if isinstance(v, bool))
                        status = "✅ GOOD" if all_good else "⚠️  ISSUES"
                else:
                    status = "❓ UNKNOWN"
            else:
                status = "❌ MISSING"
                
            print(f"  {name}: {status}")
            
        print()
        print(f"🎯 OVERALL STATUS: {overall_status}")
        
        # Save results
        self.validation_results["summary"] = {
            "overall_status": overall_status,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
def main():
    """Main validation execution."""
    validator = DaemonInstallationValidator()
    
    try:
        results = validator.validate_all()
        
        # Save validation report
        report_path = validator.project_root / "20250909-1336_daemon_installation_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n📄 Detailed validation report saved to: {report_path}")
        
        # Exit with appropriate code
        if results["summary"]["total_errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()