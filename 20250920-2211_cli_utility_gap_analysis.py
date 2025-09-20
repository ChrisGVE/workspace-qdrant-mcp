#!/usr/bin/env python3
"""
CLI Utility Gap Analysis for workspace-qdrant-mcp
Task 266.4 - Agent 4: CLI Utility Analysis Agent

Assessment of current CLI fragmentation vs unified wqm interface
according to PRD v3.0 Component 3 requirements.

PRD v3.0 Component 3 Requirements:
- Role: User Control and Administration
- Responsibilities: System administration, library management, configuration, daemon lifecycle
- Interface: Single unified CLI with domain-specific subcommands
- Scope: Complete system control without requiring MCP server
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
import re

@dataclass
class CLIComponent:
    """Represents a CLI component"""
    name: str
    location: str
    functionality: List[str]
    interface_type: str  # "script", "module", "binary"
    dependencies: List[str]
    consolidation_target: str

@dataclass
class CLIGap:
    """Represents a gap in CLI functionality"""
    gap_type: str
    current_state: str
    required_state: str
    implementation_effort: str
    dependencies: List[str]

class CLIUtilityGapAnalyzer:
    """Analyze CLI utility gaps against PRD v3.0 unified wqm interface"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cli_components = []
        self.gaps = []

    def analyze_cli_gaps(self) -> Dict:
        """Main CLI gap analysis method"""
        print("🖥️ Starting CLI utility gap analysis...")

        # Phase 1: Current CLI component inventory
        cli_inventory = self._inventory_cli_components()

        # Phase 2: Target wqm interface mapping
        target_interface = self._map_target_wqm_interface()

        # Phase 3: Gap identification
        gap_analysis = self._identify_cli_gaps(cli_inventory, target_interface)

        # Phase 4: Consolidation planning
        consolidation_plan = self._plan_cli_consolidation()

        # Phase 5: Daemon lifecycle integration analysis
        daemon_integration = self._analyze_daemon_integration()

        # Phase 6: Configuration management analysis
        config_management = self._analyze_config_management()

        # Phase 7: Migration strategy
        migration_strategy = self._design_cli_migration_strategy()

        # Generate comprehensive report
        report = self._generate_cli_gap_report(
            cli_inventory, target_interface, gap_analysis,
            consolidation_plan, daemon_integration, config_management,
            migration_strategy
        )

        return report

    def _inventory_cli_components(self) -> Dict:
        """Inventory all current CLI components"""
        print("📋 Inventorying current CLI components...")

        cli_components = []

        # Find Python CLI modules
        python_cli_files = list(self.project_root.rglob("*cli*.py"))
        for cli_file in python_cli_files:
            if '__pycache__' in str(cli_file) or '.venv' in str(cli_file):
                continue

            component = self._analyze_python_cli_file(cli_file)
            if component:
                cli_components.append(component)

        # Find shell scripts
        shell_scripts = list(self.project_root.rglob("*.sh"))
        for script in shell_scripts:
            if '.venv' in str(script) or '.git' in str(script):
                continue

            component = self._analyze_shell_script(script)
            if component:
                cli_components.append(component)

        # Find entry points in pyproject.toml
        entry_points = self._find_entry_points()

        # Find CLI wrapper and main entry points
        cli_wrappers = self._find_cli_wrappers()

        return {
            "python_cli_components": [asdict(c) for c in cli_components if c.interface_type == "module"],
            "shell_scripts": [asdict(c) for c in cli_components if c.interface_type == "script"],
            "entry_points": entry_points,
            "cli_wrappers": cli_wrappers,
            "total_components": len(cli_components),
            "fragmentation_level": self._assess_fragmentation_level(cli_components)
        }

    def _analyze_python_cli_file(self, file_path: Path) -> CLIComponent:
        """Analyze a Python CLI file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Extract functionality from imports and function names
            functionality = self._extract_cli_functionality(content)

            # Determine dependencies
            dependencies = self._extract_dependencies(content)

            rel_path = str(file_path.relative_to(self.project_root))

            return CLIComponent(
                name=file_path.stem,
                location=rel_path,
                functionality=functionality,
                interface_type="module",
                dependencies=dependencies,
                consolidation_target=self._determine_consolidation_target(functionality)
            )

        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return None

    def _analyze_shell_script(self, file_path: Path) -> CLIComponent:
        """Analyze a shell script"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            functionality = ["shell_script", "automation"]
            if "test" in content:
                functionality.append("testing")
            if "docker" in content:
                functionality.append("containerization")

            rel_path = str(file_path.relative_to(self.project_root))

            return CLIComponent(
                name=file_path.stem,
                location=rel_path,
                functionality=functionality,
                interface_type="script",
                dependencies=["bash"],
                consolidation_target="wqm_admin"
            )

        except Exception as e:
            print(f"Warning: Could not analyze shell script {file_path}: {e}")
            return None

    def _extract_cli_functionality(self, content: str) -> List[str]:
        """Extract CLI functionality from Python content"""
        functionality = []

        content_lower = content.lower()

        # Check for specific functionality patterns
        if any(term in content_lower for term in ['argparse', 'click', 'typer']):
            functionality.append("argument_parsing")

        if any(term in content_lower for term in ['collection', 'qdrant']):
            functionality.append("collection_management")

        if any(term in content_lower for term in ['daemon', 'service']):
            functionality.append("daemon_control")

        if any(term in content_lower for term in ['config', 'configuration']):
            functionality.append("configuration")

        if any(term in content_lower for term in ['watch', 'monitor']):
            functionality.append("monitoring")

        if any(term in content_lower for term in ['ingest', 'document', 'parse']):
            functionality.append("document_processing")

        if any(term in content_lower for term in ['admin', 'manage']):
            functionality.append("administration")

        if any(term in content_lower for term in ['status', 'health']):
            functionality.append("status_reporting")

        return functionality if functionality else ["general_utility"]

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from Python content"""
        dependencies = []

        # Look for imports
        import_lines = [line for line in content.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]

        for line in import_lines:
            if 'qdrant' in line:
                dependencies.append("qdrant-client")
            if 'click' in line:
                dependencies.append("click")
            if 'typer' in line:
                dependencies.append("typer")
            if 'fastapi' in line:
                dependencies.append("fastapi")

        return list(set(dependencies))

    def _determine_consolidation_target(self, functionality: List[str]) -> str:
        """Determine which wqm subcommand this should consolidate to"""
        if any(f in functionality for f in ['daemon_control', 'service']):
            return "wqm_service"
        elif any(f in functionality for f in ['collection_management', 'administration']):
            return "wqm_admin"
        elif any(f in functionality for f in ['configuration']):
            return "wqm_config"
        elif any(f in functionality for f in ['document_processing', 'ingest']):
            return "wqm_ingest"
        elif any(f in functionality for f in ['monitoring', 'status_reporting']):
            return "wqm_status"
        else:
            return "wqm_utility"

    def _find_entry_points(self) -> Dict:
        """Find CLI entry points in pyproject.toml"""
        pyproject_file = self.project_root / "pyproject.toml"
        entry_points = {}

        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r') as f:
                    content = f.read()

                # Simple parsing for entry points
                in_scripts = False
                for line in content.split('\n'):
                    if '[project.scripts]' in line:
                        in_scripts = True
                    elif line.startswith('[') and in_scripts:
                        in_scripts = False
                    elif in_scripts and '=' in line:
                        name, target = line.split('=', 1)
                        entry_points[name.strip()] = target.strip().strip('"')

            except Exception as e:
                print(f"Warning: Could not parse pyproject.toml: {e}")

        return entry_points

    def _find_cli_wrappers(self) -> List[str]:
        """Find CLI wrapper files"""
        wrappers = []

        # Look for specific wrapper patterns
        wrapper_patterns = [
            "*cli_wrapper*",
            "*main.py",
            "*__main__.py"
        ]

        for pattern in wrapper_patterns:
            for file_path in self.project_root.rglob(pattern):
                if '.venv' not in str(file_path) and '__pycache__' not in str(file_path):
                    rel_path = str(file_path.relative_to(self.project_root))
                    wrappers.append(rel_path)

        return wrappers

    def _assess_fragmentation_level(self, components: List[CLIComponent]) -> str:
        """Assess the level of CLI fragmentation"""
        if len(components) > 10:
            return "high"
        elif len(components) > 5:
            return "medium"
        elif len(components) > 2:
            return "low"
        else:
            return "minimal"

    def _map_target_wqm_interface(self) -> Dict:
        """Map target unified wqm interface from PRD v3.0"""
        return {
            "wqm": {
                "description": "Unified CLI interface for workspace-qdrant-mcp",
                "subcommands": {
                    "service": {
                        "description": "Daemon lifecycle management",
                        "commands": ["start", "stop", "restart", "status", "install", "uninstall"],
                        "responsibilities": [
                            "Start/stop daemon",
                            "Service installation",
                            "Daemon status monitoring",
                            "Log management"
                        ]
                    },
                    "admin": {
                        "description": "System administration",
                        "commands": ["collections", "users", "backup", "restore", "migrate"],
                        "responsibilities": [
                            "Collection management",
                            "User administration",
                            "Data backup/restore",
                            "System migration"
                        ]
                    },
                    "config": {
                        "description": "Configuration management",
                        "commands": ["set", "get", "list", "validate", "reset"],
                        "responsibilities": [
                            "Configuration validation",
                            "Settings management",
                            "Environment setup",
                            "Config file management"
                        ]
                    },
                    "ingest": {
                        "description": "Document ingestion and processing",
                        "commands": ["add", "batch", "watch", "status"],
                        "responsibilities": [
                            "Single document ingestion",
                            "Batch processing",
                            "Watch folder setup",
                            "Ingestion monitoring"
                        ]
                    },
                    "status": {
                        "description": "System status and monitoring",
                        "commands": ["health", "metrics", "logs", "diagnostics"],
                        "responsibilities": [
                            "Health monitoring",
                            "Performance metrics",
                            "Log access",
                            "System diagnostics"
                        ]
                    }
                }
            }
        }

    def _identify_cli_gaps(self, inventory: Dict, target: Dict) -> List[CLIGap]:
        """Identify gaps between current and target CLI"""
        print("🔍 Identifying CLI gaps...")

        gaps = []

        # Unified interface gap
        gaps.append(CLIGap(
            gap_type="Unified Interface",
            current_state=f"Fragmented CLI with {inventory['total_components']} components",
            required_state="Single 'wqm' command with subcommands",
            implementation_effort="medium",
            dependencies=["CLI framework", "Command routing"]
        ))

        # Daemon lifecycle gap
        daemon_components = [c for c in inventory['python_cli_components'] + inventory['shell_scripts']
                           if 'daemon_control' in c['functionality']]
        if not daemon_components:
            gaps.append(CLIGap(
                gap_type="Daemon Lifecycle",
                current_state="No daemon lifecycle management",
                required_state="Full daemon control via 'wqm service'",
                implementation_effort="high",
                dependencies=["Daemon communication", "Service integration"]
            ))

        # Configuration management gap
        config_components = [c for c in inventory['python_cli_components']
                           if 'configuration' in c['functionality']]
        if len(config_components) < 2:
            gaps.append(CLIGap(
                gap_type="Configuration Management",
                current_state="Limited configuration CLI",
                required_state="Comprehensive config management via 'wqm config'",
                implementation_effort="medium",
                dependencies=["Config validation", "Settings persistence"]
            ))

        # Administration gap
        admin_components = [c for c in inventory['python_cli_components']
                          if 'administration' in c['functionality']]
        if not admin_components:
            gaps.append(CLIGap(
                gap_type="System Administration",
                current_state="No unified admin interface",
                required_state="Complete admin capabilities via 'wqm admin'",
                implementation_effort="high",
                dependencies=["Collection management", "User management"]
            ))

        return gaps

    def _plan_cli_consolidation(self) -> Dict:
        """Plan CLI consolidation strategy"""
        print("📋 Planning CLI consolidation...")

        return {
            "consolidation_approach": "Progressive migration to unified wqm interface",
            "implementation_phases": {
                "phase_1": {
                    "name": "Core Framework",
                    "duration": "1 week",
                    "deliverables": [
                        "Basic wqm CLI structure",
                        "Subcommand routing",
                        "Common argument parsing"
                    ]
                },
                "phase_2": {
                    "name": "Essential Commands",
                    "duration": "2 weeks",
                    "deliverables": [
                        "wqm service implementation",
                        "wqm admin basic functionality",
                        "wqm config implementation"
                    ]
                },
                "phase_3": {
                    "name": "Advanced Features",
                    "duration": "1 week",
                    "deliverables": [
                        "wqm ingest implementation",
                        "wqm status implementation",
                        "Advanced admin features"
                    ]
                },
                "phase_4": {
                    "name": "Migration and Cleanup",
                    "duration": "1 week",
                    "deliverables": [
                        "Migrate existing CLI functionality",
                        "Remove fragmented components",
                        "Update documentation"
                    ]
                }
            },
            "backward_compatibility": {
                "strategy": "Wrapper scripts for existing commands",
                "deprecation_timeline": "3 months",
                "migration_support": "Automatic detection and guidance"
            }
        }

    def _analyze_daemon_integration(self) -> Dict:
        """Analyze daemon lifecycle integration needs"""
        print("🔄 Analyzing daemon integration needs...")

        return {
            "current_daemon_control": "Limited or missing",
            "required_capabilities": [
                "Start/stop daemon process",
                "Service installation (systemd/launchd)",
                "Process monitoring and health checks",
                "Log file management",
                "Configuration reload",
                "Graceful shutdown handling"
            ],
            "implementation_requirements": [
                "Process management library",
                "Service definition templates",
                "Inter-process communication",
                "System integration scripts",
                "Health monitoring endpoints"
            ],
            "platform_considerations": {
                "linux": "systemd service integration",
                "macos": "launchd plist generation",
                "windows": "Windows service support"
            }
        }

    def _analyze_config_management(self) -> Dict:
        """Analyze configuration management needs"""
        print("⚙️ Analyzing configuration management...")

        return {
            "current_config_handling": "Scattered configuration files",
            "required_capabilities": [
                "Centralized configuration validation",
                "Environment-specific configs",
                "Configuration templates",
                "Settings migration",
                "Config file generation"
            ],
            "config_sources": [
                "YAML/TOML configuration files",
                "Environment variables",
                "Command line arguments",
                "Default settings"
            ],
            "validation_requirements": [
                "Schema validation",
                "Dependency checking",
                "Performance impact assessment",
                "Security validation"
            ]
        }

    def _design_cli_migration_strategy(self) -> Dict:
        """Design comprehensive CLI migration strategy"""
        print("🚀 Designing CLI migration strategy...")

        return {
            "migration_principles": [
                "Preserve existing functionality",
                "Gradual transition with compatibility",
                "Clear deprecation warnings",
                "Comprehensive documentation"
            ],
            "technical_approach": {
                "framework": "Click or Typer for CLI framework",
                "structure": "Modular subcommand architecture",
                "configuration": "Unified config management",
                "error_handling": "Consistent error reporting"
            },
            "rollout_strategy": {
                "alpha": "Internal testing with core team",
                "beta": "Limited user testing with feedback",
                "production": "Full rollout with migration support"
            },
            "success_metrics": [
                "100% functionality preservation",
                "Reduced CLI complexity",
                "Improved user experience",
                "Faster command execution"
            ]
        }

    def _generate_cli_gap_report(self, inventory, target, gaps,
                               consolidation, daemon, config, migration) -> Dict:
        """Generate comprehensive CLI gap report"""
        print("📋 Generating CLI gap report...")

        return {
            "analysis_metadata": {
                "timestamp": "2025-09-20T22:11:00+02:00",
                "component": "Component 3 - CLI Utility",
                "analysis_type": "CLI Fragmentation and Gap Analysis"
            },
            "executive_summary": {
                "current_fragmentation": inventory["fragmentation_level"],
                "total_components": inventory["total_components"],
                "consolidation_target": "Single unified 'wqm' interface",
                "migration_complexity": "Medium - requires unified framework",
                "estimated_effort": "5-6 weeks for complete consolidation"
            },
            "current_inventory": inventory,
            "target_interface": target,
            "identified_gaps": [asdict(gap) for gap in gaps],
            "consolidation_plan": consolidation,
            "daemon_integration": daemon,
            "config_management": config,
            "migration_strategy": migration,
            "recommendations": {
                "immediate": "Start with wqm framework implementation",
                "short_term": "Implement essential service and admin commands",
                "long_term": "Complete migration and cleanup legacy components"
            }
        }

def main():
    """Main analysis execution"""
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"

    analyzer = CLIUtilityGapAnalyzer(project_root)
    report = analyzer.analyze_cli_gaps()

    # Save detailed report
    output_file = f"{project_root}/20250920-2211_cli_utility_gap_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ CLI utility gap analysis complete! Report saved to: {output_file}")

    # Print executive summary
    print(f"\n📊 CLI UTILITY EXECUTIVE SUMMARY:")
    print(f"Current Fragmentation: {report['executive_summary']['current_fragmentation']}")
    print(f"Total Components: {report['executive_summary']['total_components']}")
    print(f"Consolidation Target: {report['executive_summary']['consolidation_target']}")
    print(f"Migration Complexity: {report['executive_summary']['migration_complexity']}")
    print(f"Estimated Effort: {report['executive_summary']['estimated_effort']}")

    return report

if __name__ == "__main__":
    main()