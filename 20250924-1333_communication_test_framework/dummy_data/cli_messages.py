"""
CLI Command Generator

Generates dummy CLI command data for testing CLI-to-daemon communication.
Covers all wqm (workspace-qdrant-mcp) commands and their parameters.
"""

import random
import uuid
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class CliCommandSpec:
    """Specification for a CLI command and its options."""
    command: str
    subcommand: str
    options: Dict[str, Dict[str, Any]]
    description: str


class CliCommandGenerator:
    """Generates realistic CLI command data and parameters."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.seed = seed or int(time.time())
        random.seed(self.seed)

        self._init_command_specs()

    def _init_command_specs(self):
        """Initialize specifications for all CLI commands."""
        self.commands = {
            # Service Management Commands
            "wqm service install": CliCommandSpec(
                command="wqm",
                subcommand="service install",
                options={
                    "--user": {"type": "boolean", "description": "Install for current user only"},
                    "--system": {"type": "boolean", "description": "Install system-wide"},
                    "--config": {"type": "string", "description": "Custom configuration file path"}
                },
                description="Install the workspace daemon service"
            ),

            "wqm service start": CliCommandSpec(
                command="wqm",
                subcommand="service start",
                options={
                    "--background": {"type": "boolean", "description": "Run in background"},
                    "--port": {"type": "integer", "description": "gRPC service port"},
                    "--log-level": {"type": "string", "description": "Logging level"}
                },
                description="Start the workspace daemon service"
            ),

            "wqm service stop": CliCommandSpec(
                command="wqm",
                subcommand="service stop",
                options={
                    "--force": {"type": "boolean", "description": "Force stop service"},
                    "--timeout": {"type": "integer", "description": "Stop timeout in seconds"}
                },
                description="Stop the workspace daemon service"
            ),

            "wqm service status": CliCommandSpec(
                command="wqm",
                subcommand="service status",
                options={
                    "--verbose": {"type": "boolean", "description": "Show detailed status"},
                    "--json": {"type": "boolean", "description": "Output in JSON format"}
                },
                description="Check workspace daemon service status"
            ),

            "wqm service restart": CliCommandSpec(
                command="wqm",
                subcommand="service restart",
                options={
                    "--force": {"type": "boolean", "description": "Force restart"},
                    "--wait": {"type": "integer", "description": "Wait time between stop/start"}
                },
                description="Restart the workspace daemon service"
            ),

            # Admin Commands
            "wqm admin collections": CliCommandSpec(
                command="wqm",
                subcommand="admin collections",
                options={
                    "--project": {"type": "string", "description": "Filter by project name"},
                    "--detailed": {"type": "boolean", "description": "Show detailed collection info"},
                    "--json": {"type": "boolean", "description": "Output in JSON format"}
                },
                description="List and manage workspace collections"
            ),

            "wqm admin metrics": CliCommandSpec(
                command="wqm",
                subcommand="admin metrics",
                options={
                    "--service": {"type": "string", "description": "Filter by service name"},
                    "--time-range": {"type": "string", "description": "Time range for metrics"},
                    "--format": {"type": "string", "description": "Output format (json, csv, table)"}
                },
                description="View system metrics and performance data"
            ),

            "wqm admin cleanup": CliCommandSpec(
                command="wqm",
                subcommand="admin cleanup",
                options={
                    "--dry-run": {"type": "boolean", "description": "Show what would be cleaned without doing it"},
                    "--older-than": {"type": "string", "description": "Clean items older than specified duration"},
                    "--force": {"type": "boolean", "description": "Skip confirmation prompts"}
                },
                description="Clean up old data and temporary files"
            ),

            # Health Check Commands
            "wqm health check": CliCommandSpec(
                command="wqm",
                subcommand="health check",
                options={
                    "--service": {"type": "string", "description": "Check specific service"},
                    "--timeout": {"type": "integer", "description": "Health check timeout"},
                    "--detailed": {"type": "boolean", "description": "Show detailed health info"}
                },
                description="Perform comprehensive health checks"
            ),

            "wqm health monitor": CliCommandSpec(
                command="wqm",
                subcommand="health monitor",
                options={
                    "--interval": {"type": "integer", "description": "Monitoring interval in seconds"},
                    "--alert-threshold": {"type": "string", "description": "Alert threshold configuration"},
                    "--continuous": {"type": "boolean", "description": "Run continuous monitoring"}
                },
                description="Start health monitoring with alerting"
            ),

            # Document Processing Commands
            "wqm document process": CliCommandSpec(
                command="wqm",
                subcommand="document process",
                options={
                    "--file": {"type": "string", "description": "File to process"},
                    "--directory": {"type": "string", "description": "Directory to process recursively"},
                    "--collection": {"type": "string", "description": "Target collection name"},
                    "--project": {"type": "string", "description": "Project context"},
                    "--parallel": {"type": "integer", "description": "Number of parallel workers"}
                },
                description="Process documents for indexing"
            ),

            "wqm document batch": CliCommandSpec(
                command="wqm",
                subcommand="document batch",
                options={
                    "--input-file": {"type": "string", "description": "Input file list"},
                    "--output-format": {"type": "string", "description": "Output format specification"},
                    "--chunk-size": {"type": "integer", "description": "Batch chunk size"},
                    "--max-concurrency": {"type": "integer", "description": "Maximum concurrent operations"}
                },
                description="Batch process multiple documents"
            ),

            # Search Commands
            "wqm search query": CliCommandSpec(
                command="wqm",
                subcommand="search query",
                options={
                    "--query": {"type": "string", "description": "Search query text"},
                    "--collection": {"type": "string", "description": "Collection to search"},
                    "--limit": {"type": "integer", "description": "Maximum results to return"},
                    "--hybrid": {"type": "boolean", "description": "Use hybrid search"},
                    "--project": {"type": "string", "description": "Project scope"}
                },
                description="Execute search queries"
            ),

            "wqm search benchmark": CliCommandSpec(
                command="wqm",
                subcommand="search benchmark",
                options={
                    "--queries-file": {"type": "string", "description": "File containing test queries"},
                    "--iterations": {"type": "integer", "description": "Number of test iterations"},
                    "--concurrency": {"type": "integer", "description": "Concurrent query count"},
                    "--output-file": {"type": "string", "description": "Benchmark results output file"}
                },
                description="Run search performance benchmarks"
            ),

            # Collection Management Commands
            "wqm collection create": CliCommandSpec(
                command="wqm",
                subcommand="collection create",
                options={
                    "--name": {"type": "string", "description": "Collection name"},
                    "--vector-size": {"type": "integer", "description": "Vector dimension size"},
                    "--project": {"type": "string", "description": "Project association"},
                    "--metadata-schema": {"type": "string", "description": "Metadata schema definition"}
                },
                description="Create a new collection"
            ),

            "wqm collection delete": CliCommandSpec(
                command="wqm",
                subcommand="collection delete",
                options={
                    "--name": {"type": "string", "description": "Collection name to delete"},
                    "--force": {"type": "boolean", "description": "Skip confirmation"},
                    "--backup": {"type": "boolean", "description": "Create backup before deletion"}
                },
                description="Delete a collection"
            ),

            "wqm collection backup": CliCommandSpec(
                command="wqm",
                subcommand="collection backup",
                options={
                    "--name": {"type": "string", "description": "Collection to backup"},
                    "--output-path": {"type": "string", "description": "Backup output location"},
                    "--compress": {"type": "boolean", "description": "Compress backup files"},
                    "--include-vectors": {"type": "boolean", "description": "Include vector data in backup"}
                },
                description="Create collection backup"
            ),

            # Watch Management Commands
            "wqm watch start": CliCommandSpec(
                command="wqm",
                subcommand="watch start",
                options={
                    "--path": {"type": "string", "description": "Directory path to watch"},
                    "--recursive": {"type": "boolean", "description": "Watch subdirectories"},
                    "--patterns": {"type": "string", "description": "File patterns to watch"},
                    "--auto-process": {"type": "boolean", "description": "Automatically process changes"}
                },
                description="Start file system watching"
            ),

            "wqm watch stop": CliCommandSpec(
                command="wqm",
                subcommand="watch stop",
                options={
                    "--watch-id": {"type": "string", "description": "Specific watch ID to stop"},
                    "--all": {"type": "boolean", "description": "Stop all active watches"}
                },
                description="Stop file system watching"
            ),

            "wqm watch list": CliCommandSpec(
                command="wqm",
                subcommand="watch list",
                options={
                    "--active-only": {"type": "boolean", "description": "Show only active watches"},
                    "--detailed": {"type": "boolean", "description": "Show detailed watch information"}
                },
                description="List active file system watches"
            ),

            # Configuration Commands
            "wqm config show": CliCommandSpec(
                command="wqm",
                subcommand="config show",
                options={
                    "--section": {"type": "string", "description": "Show specific config section"},
                    "--format": {"type": "string", "description": "Output format (json, yaml, toml)"}
                },
                description="Show current configuration"
            ),

            "wqm config set": CliCommandSpec(
                command="wqm",
                subcommand="config set",
                options={
                    "--key": {"type": "string", "description": "Configuration key to set"},
                    "--value": {"type": "string", "description": "Configuration value"},
                    "--type": {"type": "string", "description": "Value type (string, int, bool, float)"}
                },
                description="Set configuration value"
            ),

            "wqm config validate": CliCommandSpec(
                command="wqm",
                subcommand="config validate",
                options={
                    "--config-file": {"type": "string", "description": "Configuration file to validate"},
                    "--strict": {"type": "boolean", "description": "Use strict validation"}
                },
                description="Validate configuration file"
            )
        }

    def generate_command_data(self, command_string: str, custom_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate CLI command data including options and arguments.

        Args:
            command_string: Full command string (e.g., "wqm service status")
            custom_options: Custom option values to override defaults

        Returns:
            CLI command data structure
        """
        if command_string not in self.commands:
            return self._generate_unknown_command_data(command_string, custom_options)

        command_spec = self.commands[command_string]

        # Generate option values
        options = {}
        if custom_options:
            options.update(custom_options)

        # Fill in missing options with realistic values
        for option_name, option_spec in command_spec.options.items():
            if option_name not in options:
                if random.random() > 0.6:  # 40% chance to include optional options
                    options[option_name] = self._generate_option_value(option_name, option_spec)

        return {
            "command": command_spec.command,
            "subcommand": command_spec.subcommand,
            "full_command": command_string,
            "options": options,
            "arguments": self._generate_command_arguments(command_string),
            "environment": self._generate_environment_context(),
            "execution_context": {
                "working_directory": f"/tmp/workspace_{random.randint(1, 100)}",
                "user": random.choice(["testuser", "admin", "developer"]),
                "shell": random.choice(["bash", "zsh", "fish"]),
                "timestamp": int(time.time())
            },
            "expected_output": self._generate_expected_output(command_string),
            "description": command_spec.description
        }

    def _generate_option_value(self, option_name: str, option_spec: Dict[str, Any]) -> Any:
        """Generate realistic option values based on option name and type."""
        option_type = option_spec.get("type", "string")

        if option_type == "boolean":
            return True  # If option is present, it's typically true

        elif option_type == "string":
            return self._generate_string_option(option_name)

        elif option_type == "integer":
            return self._generate_integer_option(option_name)

        else:
            return f"unknown_type_{option_type}"

    def _generate_string_option(self, option_name: str) -> str:
        """Generate realistic string option values."""
        string_generators = {
            "--config": lambda: f"/etc/wqm/config_{random.randint(1, 10)}.yaml",
            "--log-level": lambda: random.choice(["debug", "info", "warn", "error"]),
            "--project": lambda: random.choice(["workspace-qdrant-mcp", "test-project", "demo-app"]),
            "--service": lambda: random.choice(["DocumentProcessor", "SearchService", "MemoryService", "all"]),
            "--time-range": lambda: random.choice(["1h", "6h", "24h", "7d", "30d"]),
            "--format": lambda: random.choice(["json", "csv", "table", "yaml"]),
            "--older-than": lambda: random.choice(["1d", "7d", "30d", "90d"]),
            "--file": lambda: f"/tmp/test_document_{random.randint(1, 100)}.txt",
            "--directory": lambda: f"/tmp/documents_{random.randint(1, 10)}/",
            "--collection": lambda: f"test_collection_{random.randint(100, 999)}",
            "--input-file": lambda: f"/tmp/batch_input_{random.randint(1, 50)}.json",
            "--output-format": lambda: random.choice(["json", "csv", "parquet", "txt"]),
            "--query": lambda: random.choice([
                "machine learning algorithms",
                "python async patterns",
                "rust memory management",
                "gRPC communication",
                "vector database operations"
            ]),
            "--queries-file": lambda: f"/tmp/test_queries_{random.randint(1, 20)}.txt",
            "--output-file": lambda: f"/tmp/benchmark_results_{int(time.time())}.json",
            "--name": lambda: f"collection_{random.randint(1000, 9999)}",
            "--metadata-schema": lambda: f"/tmp/schema_{random.randint(1, 10)}.json",
            "--output-path": lambda: f"/tmp/backup_{int(time.time())}/",
            "--path": lambda: f"/tmp/watch_folder_{random.randint(1, 20)}/",
            "--patterns": lambda: random.choice(["*.txt,*.md", "*.py,*.rs", "*.json,*.yaml", "*"]),
            "--watch-id": lambda: str(uuid.uuid4()),
            "--section": lambda: random.choice(["server", "client", "logging", "storage"]),
            "--key": lambda: random.choice(["server.port", "logging.level", "storage.path", "client.timeout"]),
            "--value": lambda: random.choice(["8080", "info", "/tmp/storage", "30"]),
            "--type": lambda: random.choice(["string", "int", "bool", "float"]),
            "--config-file": lambda: f"/etc/wqm/test_config_{random.randint(1, 5)}.yaml",
            "--alert-threshold": lambda: random.choice(["cpu:80,memory:90", "error_rate:0.1", "response_time:1000"])
        }

        generator = string_generators.get(option_name)
        if generator:
            return generator()
        else:
            return f"test_{option_name.lstrip('-')}_{random.randint(1, 100)}"

    def _generate_integer_option(self, option_name: str) -> int:
        """Generate realistic integer option values."""
        integer_ranges = {
            "--port": (50000, 65535),
            "--timeout": (1, 300),
            "--wait": (1, 60),
            "--parallel": (1, 16),
            "--chunk-size": (10, 1000),
            "--max-concurrency": (1, 20),
            "--limit": (1, 1000),
            "--iterations": (10, 10000),
            "--concurrency": (1, 50),
            "--vector-size": [384, 768, 1536],
            "--interval": (1, 300),
            "--resolution-depth": (1, 10)
        }

        range_or_list = integer_ranges.get(option_name, (1, 100))
        if isinstance(range_or_list, list):
            return random.choice(range_or_list)
        else:
            return random.randint(*range_or_list)

    def _generate_command_arguments(self, command_string: str) -> List[str]:
        """Generate additional command arguments based on command type."""
        if "document process" in command_string:
            return [f"/tmp/document_{i}.txt" for i in range(random.randint(1, 5))]
        elif "search" in command_string:
            return [random.choice([
                "python functions",
                "rust structs",
                "API endpoints",
                "configuration options"
            ])]
        elif "collection" in command_string:
            return [f"collection_{random.randint(1, 100)}"]
        elif "watch" in command_string:
            return [f"/tmp/watch_path_{random.randint(1, 10)}/"]
        else:
            return []

    def _generate_environment_context(self) -> Dict[str, str]:
        """Generate environment context for CLI execution."""
        return {
            "WQM_CONFIG_PATH": f"/etc/wqm/config_{random.randint(1, 5)}.yaml",
            "WQM_LOG_LEVEL": random.choice(["DEBUG", "INFO", "WARN", "ERROR"]),
            "WQM_SERVICE_PORT": str(random.randint(50000, 60000)),
            "WQM_DATA_PATH": f"/tmp/wqm_data_{random.randint(1, 10)}/",
            "QDRANT_URL": f"http://localhost:{random.choice([6333, 6334, 6335])}",
            "QDRANT_API_KEY": f"test_key_{uuid.uuid4().hex[:16]}",
            "RUST_LOG": random.choice(["debug", "info", "warn", "error"]),
            "PATH": "/usr/local/bin:/usr/bin:/bin"
        }

    def _generate_expected_output(self, command_string: str) -> Dict[str, Any]:
        """Generate expected CLI output for different commands."""
        if "status" in command_string:
            return {
                "type": "status_report",
                "data": {
                    "service_status": random.choice(["running", "stopped", "starting", "error"]),
                    "pid": random.randint(1000, 99999) if random.random() > 0.2 else None,
                    "uptime": random.randint(0, 86400),
                    "memory_usage": f"{random.randint(50, 500)}MB",
                    "cpu_usage": f"{random.uniform(0.0, 100.0):.1f}%",
                    "active_connections": random.randint(0, 100)
                }
            }

        elif "collections" in command_string:
            return {
                "type": "collection_list",
                "data": [
                    {
                        "name": f"collection_{i}",
                        "documents": random.randint(0, 10000),
                        "status": random.choice(["active", "indexing", "error"]),
                        "size_mb": random.randint(1, 1000)
                    }
                    for i in range(random.randint(1, 10))
                ]
            }

        elif "health" in command_string:
            return {
                "type": "health_report",
                "data": {
                    "overall_status": random.choice(["healthy", "degraded", "unhealthy"]),
                    "services": {
                        service: {
                            "status": random.choice(["healthy", "degraded", "unhealthy"]),
                            "response_time": random.randint(1, 1000),
                            "last_check": int(time.time() - random.randint(0, 300))
                        }
                        for service in ["DocumentProcessor", "SearchService", "MemoryService", "SystemService"]
                    },
                    "system_metrics": {
                        "memory_usage": random.uniform(0.0, 100.0),
                        "cpu_usage": random.uniform(0.0, 100.0),
                        "disk_usage": random.uniform(0.0, 100.0)
                    }
                }
            }

        elif "search" in command_string:
            return {
                "type": "search_results",
                "data": {
                    "results": [
                        {
                            "document_id": str(uuid.uuid4()),
                            "score": random.uniform(0.0, 1.0),
                            "title": f"Result {i+1}",
                            "preview": f"Sample content preview for result {i+1}..."
                        }
                        for i in range(random.randint(0, 10))
                    ],
                    "total_found": random.randint(0, 1000),
                    "search_time_ms": random.randint(10, 1000)
                }
            }

        elif "process" in command_string:
            return {
                "type": "processing_result",
                "data": {
                    "processed_files": random.randint(1, 100),
                    "successful": random.randint(1, 95),
                    "failed": random.randint(0, 5),
                    "total_time_ms": random.randint(1000, 60000),
                    "documents_indexed": random.randint(1, 100)
                }
            }

        else:
            return {
                "type": "generic_output",
                "data": {
                    "message": f"Command executed successfully: {command_string}",
                    "exit_code": 0,
                    "execution_time_ms": random.randint(10, 5000)
                }
            }

    def _generate_unknown_command_data(self, command_string: str, custom_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data for unknown CLI commands."""
        return {
            "command": "wqm",
            "subcommand": "unknown",
            "full_command": command_string,
            "options": custom_options or {},
            "arguments": [],
            "environment": self._generate_environment_context(),
            "execution_context": {
                "working_directory": "/tmp/unknown_command_test/",
                "user": "testuser",
                "shell": "bash",
                "timestamp": int(time.time())
            },
            "expected_output": {
                "type": "error",
                "data": {
                    "message": f"Unknown command: {command_string}",
                    "exit_code": 1,
                    "error_type": "command_not_found"
                }
            },
            "description": f"Unknown command: {command_string}"
        }

    def generate_command_sequence(self, commands: List[str], dependencies: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Generate a sequence of CLI commands with optional dependencies.

        Args:
            commands: List of command strings to generate
            dependencies: Dictionary mapping command to its dependencies

        Returns:
            List of command data in dependency order
        """
        command_data = []
        executed = set()

        def can_execute(cmd: str) -> bool:
            if not dependencies or cmd not in dependencies:
                return True
            return all(dep in executed for dep in dependencies[cmd])

        remaining_commands = commands.copy()

        while remaining_commands:
            executed_this_round = []

            for cmd in remaining_commands[:]:
                if can_execute(cmd):
                    cmd_data = self.generate_command_data(cmd)
                    cmd_data["sequence_id"] = len(command_data)
                    cmd_data["dependencies"] = dependencies.get(cmd, []) if dependencies else []
                    command_data.append(cmd_data)
                    executed.add(cmd)
                    executed_this_round.append(cmd)
                    remaining_commands.remove(cmd)

            if not executed_this_round and remaining_commands:
                # Circular dependency or missing dependency, execute remaining anyway
                for cmd in remaining_commands:
                    cmd_data = self.generate_command_data(cmd)
                    cmd_data["sequence_id"] = len(command_data)
                    cmd_data["dependencies"] = dependencies.get(cmd, []) if dependencies else []
                    cmd_data["dependency_warning"] = "Unresolved dependencies detected"
                    command_data.append(cmd_data)
                break

        return command_data

    def generate_error_scenarios(self, command_string: str) -> List[Dict[str, Any]]:
        """Generate various error scenarios for CLI commands."""
        base_command = self.generate_command_data(command_string)
        error_scenarios = []

        # Invalid option scenarios
        invalid_options = base_command.copy()
        invalid_options["options"]["--invalid-option"] = "invalid_value"
        invalid_options["expected_output"] = {
            "type": "error",
            "data": {
                "message": "Unknown option: --invalid-option",
                "exit_code": 2,
                "error_type": "invalid_option"
            }
        }
        invalid_options["scenario"] = "invalid_option"
        error_scenarios.append(invalid_options)

        # Missing required argument
        missing_arg = base_command.copy()
        if "options" in missing_arg and missing_arg["options"]:
            # Remove a required option
            first_option = list(missing_arg["options"].keys())[0]
            del missing_arg["options"][first_option]
            missing_arg["expected_output"] = {
                "type": "error",
                "data": {
                    "message": f"Missing required option: {first_option}",
                    "exit_code": 2,
                    "error_type": "missing_argument"
                }
            }
            missing_arg["scenario"] = "missing_argument"
            error_scenarios.append(missing_arg)

        # Permission denied
        permission_denied = base_command.copy()
        permission_denied["execution_context"]["user"] = "unprivileged_user"
        permission_denied["expected_output"] = {
            "type": "error",
            "data": {
                "message": "Permission denied: insufficient privileges",
                "exit_code": 13,
                "error_type": "permission_denied"
            }
        }
        permission_denied["scenario"] = "permission_denied"
        error_scenarios.append(permission_denied)

        # Service unavailable
        service_unavailable = base_command.copy()
        service_unavailable["expected_output"] = {
            "type": "error",
            "data": {
                "message": "Service unavailable: daemon not running",
                "exit_code": 111,
                "error_type": "service_unavailable"
            }
        }
        service_unavailable["scenario"] = "service_unavailable"
        error_scenarios.append(service_unavailable)

        # Timeout
        timeout = base_command.copy()
        timeout["options"]["--timeout"] = 1  # Very short timeout
        timeout["expected_output"] = {
            "type": "error",
            "data": {
                "message": "Operation timed out after 1 seconds",
                "exit_code": 124,
                "error_type": "timeout"
            }
        }
        timeout["scenario"] = "timeout"
        error_scenarios.append(timeout)

        return error_scenarios

    def get_all_commands(self) -> List[str]:
        """Get list of all available CLI commands."""
        return list(self.commands.keys())

    def get_commands_by_category(self) -> Dict[str, List[str]]:
        """Get commands grouped by category."""
        categories = {}
        for cmd_string in self.commands.keys():
            parts = cmd_string.split(" ")
            if len(parts) >= 3:  # wqm <category> <action>
                category = parts[1]
                if category not in categories:
                    categories[category] = []
                categories[category].append(cmd_string)
        return categories

    def generate_load_test_commands(self, duration_seconds: int = 60, commands_per_second: int = 10) -> List[Dict[str, Any]]:
        """Generate commands for load testing scenarios."""
        total_commands = duration_seconds * commands_per_second
        load_test_commands = []

        available_commands = self.get_all_commands()

        for i in range(total_commands):
            command = random.choice(available_commands)
            cmd_data = self.generate_command_data(command)

            # Add load test specific metadata
            cmd_data["load_test"] = {
                "sequence_number": i,
                "target_time": time.time() + (i / commands_per_second),
                "expected_duration_ms": random.randint(100, 2000),
                "concurrent_group": i % commands_per_second
            }

            load_test_commands.append(cmd_data)

        return load_test_commands