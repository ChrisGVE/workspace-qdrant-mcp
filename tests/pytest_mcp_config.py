"""
Configuration for pytest-mcp Framework AI-Powered Evaluation.

This module provides configuration settings, evaluation criteria customization,
and integration options for the pytest-mcp framework.
"""

from dataclasses import dataclass, field
from typing import Any

from tests.utils.pytest_mcp_framework import EvaluationCriteria


@dataclass
class PytestMCPConfig:
    """Configuration for pytest-mcp framework."""

    # AI Evaluation Settings
    ai_evaluation_enabled: bool = True
    ai_confidence_threshold: float = 0.7
    ai_insights_enabled: bool = True
    ai_recommendations_enabled: bool = True

    # Evaluation Criteria Weights (must sum to 1.0)
    criteria_weights: dict[EvaluationCriteria, float] = field(default_factory=lambda: {
        EvaluationCriteria.FUNCTIONALITY: 0.25,
        EvaluationCriteria.USABILITY: 0.15,
        EvaluationCriteria.PERFORMANCE: 0.15,
        EvaluationCriteria.RELIABILITY: 0.20,
        EvaluationCriteria.COMPLETENESS: 0.10,
        EvaluationCriteria.ACCURACY: 0.10,
        EvaluationCriteria.CONSISTENCY: 0.05,
    })

    # Performance Thresholds (in milliseconds)
    performance_thresholds: dict[str, float] = field(default_factory=lambda: {
        "workspace_status": 100.0,
        "list_workspace_collections": 200.0,
        "search_workspace_tool": 500.0,
        "add_document_tool": 1000.0,
        "get_document_tool": 300.0,
        "delete_document_tool": 500.0,
        "update_document_tool": 800.0,
        "create_collection_tool": 1500.0,
        "delete_collection_tool": 1000.0,
        "collection_info_tool": 200.0,
        "health_check_tool": 100.0,
        "default": 1000.0  # Default threshold for unspecified tools
    })

    # Success Rate Thresholds
    reliability_thresholds: dict[str, float] = field(default_factory=lambda: {
        "critical_tools": 0.95,  # workspace_status, health_check
        "important_tools": 0.90,  # search, document operations
        "standard_tools": 0.80,  # collection management
        "default": 0.80
    })

    # Tool Criticality Levels
    tool_criticality: dict[str, str] = field(default_factory=lambda: {
        "workspace_status": "critical",
        "health_check_tool": "critical",
        "search_workspace_tool": "important",
        "add_document_tool": "important",
        "get_document_tool": "important",
        "list_workspace_collections": "standard",
        "create_collection_tool": "standard",
        "delete_collection_tool": "standard",
        "collection_info_tool": "standard",
        "delete_document_tool": "standard",
        "update_document_tool": "standard"
    })

    # Expected Response Fields for Tools
    expected_response_fields: dict[str, list[str]] = field(default_factory=lambda: {
        "workspace_status": ["connected", "current_project", "collections"],
        "search_workspace_tool": ["results", "total", "query"],
        "list_workspace_collections": [],  # May return list or dict
        "add_document_tool": ["success", "document_id"],
        "get_document_tool": ["content", "metadata"],
        "delete_document_tool": ["success", "document_id"],
        "update_document_tool": ["success", "document_id"],
        "create_collection_tool": ["success", "collection_name"],
        "delete_collection_tool": ["success", "collection_name"],
        "collection_info_tool": ["collection_name", "status", "points_count"],
        "health_check_tool": ["status", "qdrant_connected", "timestamp"]
    })

    # Evaluation Context Settings
    evaluation_contexts: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "development": {
            "performance_tolerance": 2.0,  # Allow 2x slower than threshold
            "error_tolerance": 0.2,  # Allow 20% error rate
            "ai_insight_level": "detailed"
        },
        "testing": {
            "performance_tolerance": 1.5,  # Allow 1.5x slower than threshold
            "error_tolerance": 0.1,  # Allow 10% error rate
            "ai_insight_level": "standard"
        },
        "staging": {
            "performance_tolerance": 1.2,  # Allow 1.2x slower than threshold
            "error_tolerance": 0.05,  # Allow 5% error rate
            "ai_insight_level": "critical_only"
        },
        "production": {
            "performance_tolerance": 1.0,  # Must meet exact thresholds
            "error_tolerance": 0.01,  # Allow 1% error rate
            "ai_insight_level": "critical_only"
        }
    })

    # Reporting Settings
    generate_detailed_reports: bool = True
    include_test_history: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["json", "html"])
    report_output_dir: str = "tests/reports/pytest_mcp"

    # Integration Settings
    fastmcp_integration: bool = True
    auto_discover_tools: bool = True
    parallel_evaluation: bool = False  # Set to True for parallel tool evaluation
    max_parallel_workers: int = 4

    def validate(self) -> list[str]:
        """Validate configuration settings and return any issues."""
        issues = []

        # Validate criteria weights sum to 1.0
        total_weight = sum(self.criteria_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Criteria weights sum to {total_weight}, expected 1.0")

        # Validate thresholds are positive
        for tool, threshold in self.performance_thresholds.items():
            if threshold <= 0:
                issues.append(f"Performance threshold for {tool} must be positive, got {threshold}")

        # Validate reliability thresholds are between 0 and 1
        for category, threshold in self.reliability_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                issues.append(f"Reliability threshold for {category} must be between 0 and 1, got {threshold}")

        # Validate AI confidence threshold
        if not 0.0 <= self.ai_confidence_threshold <= 1.0:
            issues.append(f"AI confidence threshold must be between 0 and 1, got {self.ai_confidence_threshold}")

        return issues

    def get_tool_performance_threshold(self, tool_name: str) -> float:
        """Get performance threshold for a specific tool."""
        return self.performance_thresholds.get(tool_name, self.performance_thresholds["default"])

    def get_tool_reliability_threshold(self, tool_name: str) -> float:
        """Get reliability threshold for a specific tool."""
        criticality = self.tool_criticality.get(tool_name, "standard")

        if criticality == "critical":
            return self.reliability_thresholds["critical_tools"]
        elif criticality == "important":
            return self.reliability_thresholds["important_tools"]
        else:
            return self.reliability_thresholds["standard_tools"]

    def get_expected_fields(self, tool_name: str) -> list[str]:
        """Get expected response fields for a specific tool."""
        return self.expected_response_fields.get(tool_name, [])

    def get_evaluation_context(self, context_name: str) -> dict[str, Any]:
        """Get evaluation context settings."""
        return self.evaluation_contexts.get(context_name, self.evaluation_contexts["development"])


# Default configuration instance
DEFAULT_PYTEST_MCP_CONFIG = PytestMCPConfig()


# Configuration presets for different scenarios
class PytestMCPPresets:
    """Predefined configuration presets for common scenarios."""

    @staticmethod
    def development_config() -> PytestMCPConfig:
        """Configuration optimized for development testing."""
        config = PytestMCPConfig()
        config.ai_insight_level = "detailed"
        config.performance_thresholds = {k: v * 2.0 for k, v in config.performance_thresholds.items()}  # Relaxed
        return config

    @staticmethod
    def ci_cd_config() -> PytestMCPConfig:
        """Configuration optimized for CI/CD pipeline testing."""
        config = PytestMCPConfig()
        config.ai_insights_enabled = False  # Faster execution
        config.generate_detailed_reports = False
        config.parallel_evaluation = True
        config.max_parallel_workers = 2  # Limited for CI resources
        return config

    @staticmethod
    def production_validation_config() -> PytestMCPConfig:
        """Configuration for production readiness validation."""
        config = PytestMCPConfig()

        # Stricter performance thresholds
        config.performance_thresholds = {k: v * 0.8 for k, v in config.performance_thresholds.items()}

        # Higher reliability requirements
        config.reliability_thresholds = {
            "critical_tools": 0.99,
            "important_tools": 0.95,
            "standard_tools": 0.90,
            "default": 0.90
        }

        # Stricter criteria weights (more focus on reliability)
        config.criteria_weights = {
            EvaluationCriteria.FUNCTIONALITY: 0.20,
            EvaluationCriteria.USABILITY: 0.10,
            EvaluationCriteria.PERFORMANCE: 0.20,
            EvaluationCriteria.RELIABILITY: 0.35,  # Increased
            EvaluationCriteria.COMPLETENESS: 0.05,
            EvaluationCriteria.ACCURACY: 0.05,
            EvaluationCriteria.CONSISTENCY: 0.05,
        }

        return config

    @staticmethod
    def performance_testing_config() -> PytestMCPConfig:
        """Configuration focused on performance testing."""
        config = PytestMCPConfig()

        # Performance-focused criteria weights
        config.criteria_weights = {
            EvaluationCriteria.FUNCTIONALITY: 0.15,
            EvaluationCriteria.USABILITY: 0.05,
            EvaluationCriteria.PERFORMANCE: 0.50,  # Major focus
            EvaluationCriteria.RELIABILITY: 0.20,
            EvaluationCriteria.COMPLETENESS: 0.05,
            EvaluationCriteria.ACCURACY: 0.05,
            EvaluationCriteria.CONSISTENCY: 0.00,
        }

        # Stricter performance thresholds
        config.performance_thresholds = {k: v * 0.5 for k, v in config.performance_thresholds.items()}

        return config

    @staticmethod
    def comprehensive_testing_config() -> PytestMCPConfig:
        """Configuration for comprehensive testing with detailed analysis."""
        config = PytestMCPConfig()
        config.ai_insights_enabled = True
        config.ai_recommendations_enabled = True
        config.generate_detailed_reports = True
        config.include_test_history = True
        config.export_formats = ["json", "html", "xml"]
        return config


# Utility functions for configuration management
def load_config_from_file(file_path: str) -> PytestMCPConfig:
    """Load configuration from JSON or YAML file."""
    import json
    from pathlib import Path

    config_file = Path(file_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    if config_file.suffix.lower() == '.json':
        with open(config_file) as f:
            config_data = json.load(f)
    elif config_file.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with open(config_file) as f:
                config_data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")

    # Convert string keys to EvaluationCriteria for criteria_weights
    if 'criteria_weights' in config_data:
        criteria_weights = {}
        for key, value in config_data['criteria_weights'].items():
            if isinstance(key, str):
                try:
                    criteria_key = EvaluationCriteria(key.lower())
                    criteria_weights[criteria_key] = value
                except ValueError:
                    print(f"Warning: Unknown evaluation criteria '{key}', skipping")
            else:
                criteria_weights[key] = value
        config_data['criteria_weights'] = criteria_weights

    return PytestMCPConfig(**config_data)


def save_config_to_file(config: PytestMCPConfig, file_path: str) -> None:
    """Save configuration to JSON or YAML file."""
    import json
    from pathlib import Path

    config_file = Path(file_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert EvaluationCriteria keys to strings for serialization
    config_dict = {
        'ai_evaluation_enabled': config.ai_evaluation_enabled,
        'ai_confidence_threshold': config.ai_confidence_threshold,
        'ai_insights_enabled': config.ai_insights_enabled,
        'ai_recommendations_enabled': config.ai_recommendations_enabled,
        'criteria_weights': {criteria.value: weight for criteria, weight in config.criteria_weights.items()},
        'performance_thresholds': config.performance_thresholds,
        'reliability_thresholds': config.reliability_thresholds,
        'tool_criticality': config.tool_criticality,
        'expected_response_fields': config.expected_response_fields,
        'evaluation_contexts': config.evaluation_contexts,
        'generate_detailed_reports': config.generate_detailed_reports,
        'include_test_history': config.include_test_history,
        'export_formats': config.export_formats,
        'report_output_dir': config.report_output_dir,
        'fastmcp_integration': config.fastmcp_integration,
        'auto_discover_tools': config.auto_discover_tools,
        'parallel_evaluation': config.parallel_evaluation,
        'max_parallel_workers': config.max_parallel_workers
    }

    if config_file.suffix.lower() == '.json':
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif config_file.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except ImportError:
            raise ImportError("PyYAML is required to save YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")


def validate_config(config: PytestMCPConfig) -> bool:
    """Validate configuration and print any issues."""
    issues = config.validate()

    if issues:
        print("Configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("Configuration validation passed")
    return True


# Environment-based configuration loading
def get_config_for_environment(environment: str = "development") -> PytestMCPConfig:
    """Get configuration appropriate for the specified environment."""
    env_configs = {
        "development": PytestMCPPresets.development_config(),
        "testing": DEFAULT_PYTEST_MCP_CONFIG,
        "staging": PytestMCPPresets.production_validation_config(),
        "production": PytestMCPPresets.production_validation_config(),
        "ci": PytestMCPPresets.ci_cd_config(),
        "performance": PytestMCPPresets.performance_testing_config(),
        "comprehensive": PytestMCPPresets.comprehensive_testing_config()
    }

    return env_configs.get(environment, DEFAULT_PYTEST_MCP_CONFIG)
