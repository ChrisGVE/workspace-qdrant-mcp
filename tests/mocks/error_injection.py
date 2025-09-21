"""
Error injection framework for comprehensive failure testing.

Provides configurable error injection capabilities for testing error handling,
resilience, and recovery scenarios across all external dependencies.
"""

import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class FailureMode(Enum):
    """Common failure modes for external dependencies."""

    # Network failures
    CONNECTION_TIMEOUT = "connection_timeout"
    CONNECTION_REFUSED = "connection_refused"
    NETWORK_UNREACHABLE = "network_unreachable"
    DNS_RESOLUTION_FAILED = "dns_resolution_failed"

    # Service failures
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    MAINTENANCE_MODE = "maintenance_mode"

    # Authentication/Authorization
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_DENIED = "authorization_denied"
    TOKEN_EXPIRED = "token_expired"
    INVALID_CREDENTIALS = "invalid_credentials"

    # Data/Protocol failures
    INVALID_REQUEST = "invalid_request"
    MALFORMED_RESPONSE = "malformed_response"
    PROTOCOL_ERROR = "protocol_error"
    DATA_CORRUPTION = "data_corruption"

    # Resource failures
    MEMORY_EXHAUSTED = "memory_exhausted"
    DISK_FULL = "disk_full"
    CPU_OVERLOAD = "cpu_overload"
    THREAD_POOL_EXHAUSTED = "thread_pool_exhausted"

    # Transient failures
    TEMPORARY_FAILURE = "temporary_failure"
    INTERMITTENT_ERROR = "intermittent_error"
    FLAKY_BEHAVIOR = "flaky_behavior"


class ErrorInjector(ABC):
    """Base class for error injection in external dependencies."""

    def __init__(self):
        self.failure_modes: Dict[str, Dict[str, Any]] = {}
        self.active_scenarios: Set[str] = set()
        self.error_count = 0
        self.total_operations = 0
        self.enabled = True

    def enable(self) -> None:
        """Enable error injection."""
        self.enabled = True

    def disable(self) -> None:
        """Disable error injection."""
        self.enabled = False

    def reset(self) -> None:
        """Reset error injection state."""
        self.error_count = 0
        self.total_operations = 0
        self.active_scenarios.clear()
        for mode in self.failure_modes.values():
            mode["probability"] = 0.0

    def configure_failure_mode(
        self,
        mode: str,
        probability: float,
        **kwargs
    ) -> None:
        """
        Configure a specific failure mode.

        Args:
            mode: Failure mode identifier
            probability: Probability of failure (0.0 to 1.0)
            **kwargs: Mode-specific configuration
        """
        if mode not in self.failure_modes:
            self.failure_modes[mode] = {}

        self.failure_modes[mode].update({
            "probability": max(0.0, min(1.0, probability)),
            **kwargs
        })

        if probability > 0:
            self.active_scenarios.add(mode)
        else:
            self.active_scenarios.discard(mode)

    def should_inject_error(self) -> bool:
        """Determine if an error should be injected."""
        if not self.enabled or not self.active_scenarios:
            self.total_operations += 1
            return False

        # Check each active failure mode
        for mode in self.active_scenarios:
            config = self.failure_modes.get(mode, {})
            probability = config.get("probability", 0.0)

            if random.random() < probability:
                self.error_count += 1
                self.total_operations += 1
                return True

        self.total_operations += 1
        return False

    def get_random_error(self) -> str:
        """Get a random error type from active scenarios."""
        if not self.active_scenarios:
            return "generic_error"

        # Weighted selection based on probabilities
        weighted_modes = []
        for mode in self.active_scenarios:
            config = self.failure_modes.get(mode, {})
            probability = config.get("probability", 0.0)
            if probability > 0:
                weighted_modes.extend([mode] * int(probability * 100))

        return random.choice(weighted_modes) if weighted_modes else "generic_error"

    def get_statistics(self) -> Dict[str, Any]:
        """Get error injection statistics."""
        return {
            "total_operations": self.total_operations,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.total_operations),
            "active_scenarios": list(self.active_scenarios),
            "enabled": self.enabled
        }


class FailureScenarios:
    """Pre-defined failure scenarios for common testing situations."""

    @staticmethod
    def connection_issues() -> Dict[str, Dict[str, Any]]:
        """Network connection related failures."""
        return {
            "connection_timeout": {"probability": 0.1, "delay": 30.0},
            "connection_refused": {"probability": 0.05, "retry_delay": 1.0},
            "network_unreachable": {"probability": 0.03, "error_message": "Network unreachable"},
        }

    @staticmethod
    def service_degradation() -> Dict[str, Dict[str, Any]]:
        """Service degradation and overload scenarios."""
        return {
            "service_unavailable": {"probability": 0.1, "status_code": 503},
            "rate_limited": {"probability": 0.15, "status_code": 429, "retry_after": 60},
            "maintenance_mode": {"probability": 0.02, "status_code": 503},
        }

    @staticmethod
    def authentication_problems() -> Dict[str, Dict[str, Any]]:
        """Authentication and authorization failures."""
        return {
            "authentication_failed": {"probability": 0.05, "status_code": 401},
            "authorization_denied": {"probability": 0.03, "status_code": 403},
            "token_expired": {"probability": 0.08, "refresh_required": True},
        }

    @staticmethod
    def data_corruption() -> Dict[str, Dict[str, Any]]:
        """Data integrity and protocol errors."""
        return {
            "malformed_response": {"probability": 0.02, "corruption_type": "json"},
            "protocol_error": {"probability": 0.01, "error_type": "version_mismatch"},
            "data_corruption": {"probability": 0.005, "corruption_bytes": 10},
        }

    @staticmethod
    def resource_exhaustion() -> Dict[str, Dict[str, Any]]:
        """Resource exhaustion scenarios."""
        return {
            "memory_exhausted": {"probability": 0.02, "available_memory": 0},
            "disk_full": {"probability": 0.01, "available_space": 0},
            "cpu_overload": {"probability": 0.05, "cpu_usage": 100},
        }

    @staticmethod
    def intermittent_failures() -> Dict[str, Dict[str, Any]]:
        """Flaky and intermittent behavior."""
        return {
            "temporary_failure": {"probability": 0.1, "duration": 5.0},
            "intermittent_error": {"probability": 0.08, "pattern": "every_3rd"},
            "flaky_behavior": {"probability": 0.15, "success_rate": 0.8},
        }

    @staticmethod
    def realistic_production() -> Dict[str, Dict[str, Any]]:
        """Realistic production-like error rates."""
        scenarios = {}
        scenarios.update(FailureScenarios.connection_issues())
        scenarios.update(FailureScenarios.service_degradation())
        scenarios.update(FailureScenarios.authentication_problems())

        # Reduce probabilities for realistic production rates
        for config in scenarios.values():
            config["probability"] *= 0.1  # 10x lower rates

        return scenarios

    @staticmethod
    def stress_testing() -> Dict[str, Dict[str, Any]]:
        """High failure rates for stress testing."""
        scenarios = {}
        scenarios.update(FailureScenarios.connection_issues())
        scenarios.update(FailureScenarios.service_degradation())
        scenarios.update(FailureScenarios.resource_exhaustion())

        # Increase probabilities for stress testing
        for config in scenarios.values():
            config["probability"] = min(0.5, config["probability"] * 5)

        return scenarios


class ErrorModeManager:
    """Manager for coordinating error injection across multiple components."""

    def __init__(self):
        self.injectors: Dict[str, ErrorInjector] = {}
        self.global_enabled = True
        self.scenario_active = False

    def register_injector(self, component: str, injector: ErrorInjector) -> None:
        """Register an error injector for a component."""
        self.injectors[component] = injector

    def enable_all(self) -> None:
        """Enable error injection for all components."""
        self.global_enabled = True
        for injector in self.injectors.values():
            injector.enable()

    def disable_all(self) -> None:
        """Disable error injection for all components."""
        self.global_enabled = False
        for injector in self.injectors.values():
            injector.disable()

    def reset_all(self) -> None:
        """Reset all error injectors."""
        for injector in self.injectors.values():
            injector.reset()
        self.scenario_active = False

    def apply_scenario(self, scenario_name: str, components: Optional[List[str]] = None) -> None:
        """
        Apply a predefined failure scenario to specified components.

        Args:
            scenario_name: Name of the scenario to apply
            components: List of components to apply to (None for all)
        """
        scenario_configs = {
            "connection_issues": FailureScenarios.connection_issues(),
            "service_degradation": FailureScenarios.service_degradation(),
            "authentication_problems": FailureScenarios.authentication_problems(),
            "data_corruption": FailureScenarios.data_corruption(),
            "resource_exhaustion": FailureScenarios.resource_exhaustion(),
            "intermittent_failures": FailureScenarios.intermittent_failures(),
            "realistic_production": FailureScenarios.realistic_production(),
            "stress_testing": FailureScenarios.stress_testing(),
        }

        if scenario_name not in scenario_configs:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_config = scenario_configs[scenario_name]
        target_components = components or list(self.injectors.keys())

        for component in target_components:
            if component in self.injectors:
                injector = self.injectors[component]
                for mode, config in scenario_config.items():
                    injector.configure_failure_mode(mode, **config)

        self.scenario_active = True

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics from all injectors."""
        stats = {
            "global_enabled": self.global_enabled,
            "scenario_active": self.scenario_active,
            "components": {}
        }

        total_operations = 0
        total_errors = 0

        for component, injector in self.injectors.items():
            component_stats = injector.get_statistics()
            stats["components"][component] = component_stats
            total_operations += component_stats["total_operations"]
            total_errors += component_stats["error_count"]

        stats["aggregate"] = {
            "total_operations": total_operations,
            "total_errors": total_errors,
            "global_error_rate": total_errors / max(1, total_operations)
        }

        return stats


def create_error_injector() -> ErrorInjector:
    """Create a basic error injector."""
    return ErrorInjector()


def create_error_manager() -> ErrorModeManager:
    """Create an error mode manager."""
    return ErrorModeManager()


# Convenience functions for common scenarios
def configure_realistic_errors(injector: ErrorInjector) -> None:
    """Configure realistic production-like error rates."""
    scenarios = FailureScenarios.realistic_production()
    for mode, config in scenarios.items():
        injector.configure_failure_mode(mode, **config)


def configure_stress_errors(injector: ErrorInjector) -> None:
    """Configure high error rates for stress testing."""
    scenarios = FailureScenarios.stress_testing()
    for mode, config in scenarios.items():
        injector.configure_failure_mode(mode, **config)


def configure_connection_errors(injector: ErrorInjector, probability: float = 0.1) -> None:
    """Configure network connection errors."""
    scenarios = FailureScenarios.connection_issues()
    for mode, config in scenarios.items():
        config["probability"] = probability
        injector.configure_failure_mode(mode, **config)