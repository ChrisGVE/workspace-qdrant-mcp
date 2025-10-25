"""
Graceful Degradation System for Four-Component Architecture.

This module provides comprehensive graceful degradation strategies for the workspace-qdrant-mcp
system, ensuring seamless operation under various failure scenarios and resource constraints.

Key Features:
    - Component failure detection and automatic degradation mode selection
    - Circuit breaker patterns with progressive failure handling
    - Fallback mechanisms for reduced functionality modes
    - Resource throttling under high load conditions
    - User-friendly status communication with actionable guidance
    - Integration with lifecycle management and health monitoring

Degradation Modes:
    - OFFLINE_CLI: Cached responses and local file operations only
    - READ_ONLY_MCP: Search and retrieval operations without indexing
    - CACHED_RESPONSES: Serve from cache, disable real-time processing
    - REDUCED_FEATURES: Disable non-essential features to conserve resources
    - EMERGENCY_MODE: Minimal functionality for critical operations only

Example:
    ```python
    from workspace_qdrant_mcp.core.graceful_degradation import DegradationManager

    # Initialize degradation manager
    degradation = DegradationManager(
        lifecycle_manager=lifecycle_manager,
        health_monitor=health_monitor
    )
    await degradation.initialize()

    # Check if feature is available
    if degradation.is_feature_available(FeatureType.SEMANTIC_SEARCH):
        # Proceed with full functionality
        pass
    else:
        # Use fallback mechanism
        fallback_result = degradation.get_fallback_response(request)
    ```
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from weakref import WeakSet

from loguru import logger

from .component_coordination import (
    ComponentCoordinator,
    ComponentType,
    ProcessingQueueType,
)
from .component_lifecycle import (
    ComponentLifecycleManager,
)
from .lsp_health_monitor import (
    LspHealthMonitor,
    NotificationLevel,
    UserNotification,
)


class DegradationMode(Enum):
    """System degradation modes with increasing severity."""

    NORMAL = 0                    # Full functionality available
    PERFORMANCE_REDUCED = 1       # Reduced performance, all features available
    FEATURES_LIMITED = 2          # Some non-essential features disabled
    READ_ONLY = 3                 # Read operations only, no modifications
    CACHED_ONLY = 4               # Serve cached responses only
    OFFLINE_CLI = 5               # CLI operations only, no network/daemon
    EMERGENCY = 6                 # Minimal critical functionality only
    UNAVAILABLE = 7               # System completely unavailable

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class FeatureType(Enum):
    """System feature types for degradation control."""

    # Core features
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    DOCUMENT_INGESTION = "document_ingestion"
    FILE_WATCHING = "file_watching"

    # Advanced features
    HYBRID_SEARCH = "hybrid_search"
    REAL_TIME_INDEXING = "real_time_indexing"
    LSP_INTEGRATION = "lsp_integration"
    CONTEXT_INJECTION = "context_injection"

    # Administrative features
    HEALTH_MONITORING = "health_monitoring"
    METRICS_COLLECTION = "metrics_collection"
    ADMIN_OPERATIONS = "admin_operations"
    BACKGROUND_PROCESSING = "background_processing"

    # User interface features
    MCP_SERVER = "mcp_server"
    CLI_OPERATIONS = "cli_operations"
    WEB_INTERFACE = "web_interface"


class CircuitBreakerState(Enum):
    """Circuit breaker states for component failure handling."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open" # Testing if recovery is possible


class ThrottleStrategy(Enum):
    """Resource throttling strategies under high load."""

    NONE = "none"                    # No throttling
    QUEUE_BASED = "queue_based"      # Limit queue sizes
    RATE_LIMITING = "rate_limiting"  # Limit request rates
    RESOURCE_BASED = "resource_based"# Throttle based on resource usage
    PRIORITY_BASED = "priority_based"# Prioritize critical operations


@dataclass
class FeatureConfig:
    """Configuration for individual feature degradation."""

    feature_type: FeatureType
    required_components: set[ComponentType]
    fallback_components: set[ComponentType] = field(default_factory=set)
    degradation_threshold: float = 0.8  # Health threshold for degradation
    recovery_threshold: float = 0.9     # Health threshold for recovery
    cache_duration_seconds: int = 300   # Cache duration for fallback responses
    max_cache_size: int = 1000         # Maximum cache entries
    priority: int = 5                  # Feature priority (1=critical, 10=optional)
    can_use_cache: bool = True         # Whether feature can use cached responses
    enable_partial_functionality: bool = True  # Allow partial feature operation


@dataclass
class CircuitBreaker:
    """Circuit breaker for component failure detection."""

    component_id: str
    failure_threshold: int = 5         # Failures before opening circuit
    recovery_timeout: int = 60         # Seconds before attempting recovery
    half_open_success_threshold: int = 3  # Successes needed to close circuit

    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_attempt_time: float = 0.0

    def should_allow_request(self) -> bool:
        """Check if requests should be allowed through circuit breaker."""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    def record_success(self):
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
        elif (self.state == CircuitBreakerState.CLOSED and
              self.failure_count >= self.failure_threshold):
            self.state = CircuitBreakerState.OPEN


@dataclass
class DegradationEvent:
    """Event record for degradation changes."""

    event_id: str
    degradation_mode: DegradationMode
    previous_mode: DegradationMode
    trigger_reason: str
    affected_features: list[FeatureType]
    affected_components: list[ComponentType]
    automatic_recovery: bool
    user_guidance: list[str]
    timestamp: datetime = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResourceThrottle:
    """Resource throttling configuration and state."""

    strategy: ThrottleStrategy
    cpu_threshold: float = 80.0        # CPU usage threshold for throttling
    memory_threshold: float = 85.0     # Memory usage threshold for throttling
    queue_size_threshold: int = 1000   # Queue size threshold for throttling
    rate_limit_per_second: int = 100   # Max requests per second

    # State tracking
    is_active: bool = False
    throttle_factor: float = 1.0       # 1.0 = no throttling, 0.1 = severe throttling
    last_update: float = 0.0


class DegradationManager:
    """
    Comprehensive graceful degradation manager for the four-component architecture.

    Provides automatic degradation mode selection, circuit breaker patterns,
    fallback mechanisms, and user-friendly status communication.
    """

    # Feature dependency mapping
    FEATURE_DEPENDENCIES = {
        FeatureType.SEMANTIC_SEARCH: {
            ComponentType.RUST_DAEMON,
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.KEYWORD_SEARCH: {
            ComponentType.RUST_DAEMON,
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.DOCUMENT_INGESTION: {
            ComponentType.RUST_DAEMON,
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.FILE_WATCHING: {
            ComponentType.RUST_DAEMON
        },
        FeatureType.HYBRID_SEARCH: {
            ComponentType.RUST_DAEMON,
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.REAL_TIME_INDEXING: {
            ComponentType.RUST_DAEMON,
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.LSP_INTEGRATION: {
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.CONTEXT_INJECTION: {
            ComponentType.CONTEXT_INJECTOR,
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.MCP_SERVER: {
            ComponentType.PYTHON_MCP_SERVER
        },
        FeatureType.CLI_OPERATIONS: {
            ComponentType.CLI_UTILITY
        },
        FeatureType.ADMIN_OPERATIONS: {
            ComponentType.CLI_UTILITY,
            ComponentType.PYTHON_MCP_SERVER
        }
    }

    # Degradation mode feature availability
    MODE_FEATURE_AVAILABILITY = {
        DegradationMode.NORMAL: set(FeatureType),
        DegradationMode.PERFORMANCE_REDUCED: set(FeatureType),
        DegradationMode.FEATURES_LIMITED: {
            FeatureType.SEMANTIC_SEARCH,
            FeatureType.KEYWORD_SEARCH,
            FeatureType.DOCUMENT_INGESTION,
            FeatureType.MCP_SERVER,
            FeatureType.CLI_OPERATIONS,
        },
        DegradationMode.READ_ONLY: {
            FeatureType.SEMANTIC_SEARCH,
            FeatureType.KEYWORD_SEARCH,
            FeatureType.MCP_SERVER,
            FeatureType.CLI_OPERATIONS,
        },
        DegradationMode.CACHED_ONLY: {
            FeatureType.SEMANTIC_SEARCH,
            FeatureType.KEYWORD_SEARCH,
            FeatureType.CLI_OPERATIONS,
        },
        DegradationMode.OFFLINE_CLI: {
            FeatureType.CLI_OPERATIONS,
        },
        DegradationMode.EMERGENCY: {
            FeatureType.CLI_OPERATIONS,
        },
        DegradationMode.UNAVAILABLE: set(),
    }

    def __init__(
        self,
        lifecycle_manager: ComponentLifecycleManager | None = None,
        health_monitor: LspHealthMonitor | None = None,
        coordinator: ComponentCoordinator | None = None,
        config: dict[str, Any] | None = None
    ):
        """
        Initialize degradation manager.

        Args:
            lifecycle_manager: Component lifecycle manager instance
            health_monitor: LSP health monitor instance
            coordinator: Component coordinator instance
            config: Configuration dictionary
        """
        self.lifecycle_manager = lifecycle_manager
        self.health_monitor = health_monitor
        self.coordinator = coordinator
        self.config = config or {}

        # Current state
        self.current_mode = DegradationMode.NORMAL
        self.previous_mode = DegradationMode.NORMAL

        # Feature management
        self.feature_configs: dict[FeatureType, FeatureConfig] = {}
        self.disabled_features: set[FeatureType] = set()
        self.cached_responses: dict[str, Any] = {}

        # Circuit breakers for each component
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Resource throttling
        self.resource_throttle = ResourceThrottle(ThrottleStrategy.NONE)

        # Event tracking
        self.degradation_events: list[DegradationEvent] = []

        # Notification handlers
        self.notification_handlers: WeakSet[Callable[[UserNotification], None]] = WeakSet()

        # Background tasks
        self.monitoring_tasks: list[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # Statistics
        self.start_time = time.time()
        self.degradation_count = 0
        self.recovery_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize feature configurations
        self._initialize_feature_configs()

        logger.info("Degradation manager initialized")

    def _initialize_feature_configs(self):
        """Initialize default feature configurations."""
        for feature_type in FeatureType:
            required_components = self.FEATURE_DEPENDENCIES.get(feature_type, set())

            # Determine priority based on feature type
            priority = 5  # Default priority
            if feature_type in {FeatureType.SEMANTIC_SEARCH, FeatureType.KEYWORD_SEARCH}:
                priority = 1  # Critical search features
            elif feature_type in {FeatureType.MCP_SERVER, FeatureType.CLI_OPERATIONS}:
                priority = 2  # Essential interfaces
            elif feature_type in {FeatureType.DOCUMENT_INGESTION, FeatureType.FILE_WATCHING}:
                priority = 3  # Important data processing
            elif feature_type in {FeatureType.LSP_INTEGRATION, FeatureType.CONTEXT_INJECTION}:
                priority = 7  # Nice-to-have integrations

            self.feature_configs[feature_type] = FeatureConfig(
                feature_type=feature_type,
                required_components=required_components,
                priority=priority
            )

    async def initialize(self) -> bool:
        """
        Initialize the degradation manager.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize circuit breakers for all component types
            for component_type in ComponentType:
                component_id = f"{component_type.value}-default"
                self.circuit_breakers[component_id] = CircuitBreaker(component_id)

            # Start monitoring tasks
            await self._start_monitoring_tasks()

            # Register with health monitor if available
            if self.health_monitor:
                self.health_monitor.register_notification_handler(
                    self._handle_health_notification
                )

            logger.info("Degradation manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize degradation manager: {e}")
            return False

    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        self.monitoring_tasks = [
            asyncio.create_task(self._degradation_monitoring_loop()),
            asyncio.create_task(self._resource_monitoring_loop()),
            asyncio.create_task(self._cache_cleanup_loop()),
        ]
        logger.info("Started degradation monitoring tasks")

    async def _degradation_monitoring_loop(self):
        """Main degradation monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                # Check component health and update degradation mode
                await self._evaluate_degradation_mode()

                # Check circuit breaker states
                await self._update_circuit_breakers()

                # Update feature availability
                await self._update_feature_availability()

                await asyncio.sleep(10.0)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in degradation monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _resource_monitoring_loop(self):
        """Monitor resource usage and adjust throttling."""
        while not self.shutdown_event.is_set():
            try:
                await self._monitor_resource_usage()
                await asyncio.sleep(30.0)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(10.0)

    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries."""
        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_expired_cache()
                await asyncio.sleep(300.0)  # Clean every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(60.0)

    async def _evaluate_degradation_mode(self):
        """Evaluate and update system degradation mode based on component health."""
        if not self.lifecycle_manager:
            return

        try:
            # Get component status
            component_status = await self.lifecycle_manager.get_component_status()

            if "components" not in component_status:
                return

            # Count healthy components
            healthy_components = set()
            degraded_components = set()
            failed_components = set()

            for comp_type_str, comp_data in component_status["components"].items():
                component_type = ComponentType(comp_type_str)
                state = comp_data.get("state", "unknown")

                if state in ["operational", "ready"]:
                    healthy_components.add(component_type)
                elif state in ["degraded"]:
                    degraded_components.add(component_type)
                else:
                    failed_components.add(component_type)

            # Determine appropriate degradation mode
            new_mode = self._calculate_degradation_mode(
                healthy_components, degraded_components, failed_components
            )

            # Update mode if changed
            if new_mode != self.current_mode:
                await self._change_degradation_mode(
                    new_mode,
                    f"Component health evaluation: {len(healthy_components)} healthy, "
                    f"{len(degraded_components)} degraded, {len(failed_components)} failed"
                )

        except Exception as e:
            logger.error(f"Failed to evaluate degradation mode: {e}")

    def _calculate_degradation_mode(
        self,
        healthy_components: set[ComponentType],
        degraded_components: set[ComponentType],
        failed_components: set[ComponentType]
    ) -> DegradationMode:
        """Calculate appropriate degradation mode based on component states."""
        total_components = len(ComponentType)
        healthy_count = len(healthy_components)
        degraded_count = len(degraded_components)
        len(failed_components)

        # Check for critical component failures
        if ComponentType.PYTHON_MCP_SERVER in failed_components:
            if ComponentType.CLI_UTILITY in healthy_components:
                return DegradationMode.OFFLINE_CLI
            else:
                return DegradationMode.UNAVAILABLE

        if ComponentType.RUST_DAEMON in failed_components:
            if ComponentType.PYTHON_MCP_SERVER in healthy_components:
                return DegradationMode.CACHED_ONLY
            elif ComponentType.CLI_UTILITY in healthy_components:
                return DegradationMode.OFFLINE_CLI
            else:
                return DegradationMode.UNAVAILABLE

        # Calculate health percentage
        health_percentage = (healthy_count + degraded_count * 0.5) / total_components

        if health_percentage >= 0.9:
            return DegradationMode.NORMAL
        elif health_percentage >= 0.8:
            return DegradationMode.PERFORMANCE_REDUCED
        elif health_percentage >= 0.6:
            return DegradationMode.FEATURES_LIMITED
        elif health_percentage >= 0.4:
            return DegradationMode.READ_ONLY
        elif health_percentage >= 0.2:
            return DegradationMode.CACHED_ONLY
        else:
            return DegradationMode.EMERGENCY

    async def _change_degradation_mode(self, new_mode: DegradationMode, reason: str):
        """Change degradation mode and notify stakeholders."""
        previous_mode = self.current_mode
        self.current_mode = new_mode

        # Track statistics
        if new_mode > previous_mode:  # Degradation (higher enum value = more severe)
            self.degradation_count += 1
        elif new_mode < previous_mode:  # Recovery
            self.recovery_count += 1

        # Update disabled features based on new mode
        self._update_disabled_features_for_mode(new_mode)

        # Create degradation event
        event = DegradationEvent(
            event_id=str(uuid.uuid4()),
            degradation_mode=new_mode,
            previous_mode=previous_mode,
            trigger_reason=reason,
            affected_features=list(self.disabled_features),
            affected_components=[],  # Will be populated by component analysis
            automatic_recovery=True,
            user_guidance=self._generate_user_guidance(new_mode, reason)
        )

        self.degradation_events.append(event)

        # Keep only last 100 events
        if len(self.degradation_events) > 100:
            self.degradation_events = self.degradation_events[-100:]

        # Log mode change
        logger.warning(
            f"Degradation mode changed: {previous_mode.name.lower()} -> {new_mode.name.lower()}",
            reason=reason,
            disabled_features=[f.name.lower() for f in self.disabled_features]
        )

        # Send notification
        await self._send_degradation_notification(event)

        # Store event in coordinator if available
        if self.coordinator:
            await self._store_degradation_event(event)

    def _update_disabled_features_for_mode(self, mode: DegradationMode):
        """Update disabled features based on degradation mode."""
        available_features = self.MODE_FEATURE_AVAILABILITY.get(mode, set())
        all_features = set(FeatureType)
        self.disabled_features = all_features - available_features

    def _generate_user_guidance(self, mode: DegradationMode, reason: str) -> list[str]:
        """Generate user guidance for degradation mode."""
        guidance = []

        if mode == DegradationMode.PERFORMANCE_REDUCED:
            guidance.extend([
                "System is experiencing reduced performance",
                "Some operations may take longer than usual",
                "Non-critical features remain available",
                "Monitor system resources and consider reducing workload"
            ])
        elif mode == DegradationMode.FEATURES_LIMITED:
            guidance.extend([
                "Some advanced features have been temporarily disabled",
                "Core search and document operations remain available",
                "Check component health status for specific issues",
                "Consider restarting affected services"
            ])
        elif mode == DegradationMode.READ_ONLY:
            guidance.extend([
                "System is in read-only mode",
                "Document ingestion and indexing are disabled",
                "Search operations remain available",
                "No new documents will be processed until issues are resolved"
            ])
        elif mode == DegradationMode.CACHED_ONLY:
            guidance.extend([
                "System is serving cached responses only",
                "Real-time search may not reflect recent changes",
                "Basic CLI operations may still be available",
                "Check Rust daemon and MCP server status"
            ])
        elif mode == DegradationMode.OFFLINE_CLI:
            guidance.extend([
                "Only CLI operations are available",
                "MCP server and web interfaces are unavailable",
                "Use CLI commands for basic file operations",
                "Check network connectivity and service status"
            ])
        elif mode == DegradationMode.EMERGENCY:
            guidance.extend([
                "System is in emergency mode with minimal functionality",
                "Only critical operations are available",
                "Immediate attention required to resolve issues",
                "Contact system administrator if available"
            ])
        elif mode == DegradationMode.UNAVAILABLE:
            guidance.extend([
                "System is currently unavailable",
                "All services appear to be down",
                "Check if processes are running and restart if needed",
                "Verify configuration and system resources"
            ])

        return guidance

    async def _update_circuit_breakers(self):
        """Update circuit breaker states based on component health."""
        if not self.lifecycle_manager:
            return

        try:
            component_status = await self.lifecycle_manager.get_component_status()

            if "components" not in component_status:
                return

            for comp_type_str, comp_data in component_status["components"].items():
                component_id = f"{comp_type_str}-default"

                if component_id in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[component_id]
                    state = comp_data.get("state", "unknown")

                    if state in ["operational", "ready"]:
                        circuit_breaker.record_success()
                    elif state in ["failed", "unhealthy"]:
                        circuit_breaker.record_failure()

        except Exception as e:
            logger.error(f"Failed to update circuit breakers: {e}")

    async def _update_feature_availability(self):
        """Update feature availability based on component states and circuit breakers."""
        # This is already handled by _update_disabled_features_for_mode
        # Additional logic could be added here for more granular control
        pass

    async def _monitor_resource_usage(self):
        """Monitor resource usage and update throttling."""
        try:
            # Get resource metrics from coordinator if available
            if self.coordinator:
                # This is a placeholder - actual implementation would get real metrics
                cpu_usage = 0.0  # Would get from system metrics
                memory_usage = 0.0  # Would get from system metrics

                # Update throttling based on resource usage
                if (cpu_usage > self.resource_throttle.cpu_threshold or
                    memory_usage > self.resource_throttle.memory_threshold):

                    if not self.resource_throttle.is_active:
                        self.resource_throttle.is_active = True
                        self.resource_throttle.strategy = ThrottleStrategy.RESOURCE_BASED
                        self.resource_throttle.throttle_factor = 0.5

                        logger.warning(
                            "Resource throttling activated",
                            cpu_usage=cpu_usage,
                            memory_usage=memory_usage
                        )
                else:
                    if self.resource_throttle.is_active:
                        self.resource_throttle.is_active = False
                        self.resource_throttle.throttle_factor = 1.0

                        logger.info("Resource throttling deactivated")

                self.resource_throttle.last_update = time.time()

        except Exception as e:
            logger.error(f"Failed to monitor resource usage: {e}")

    async def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        try:
            current_time = time.time()
            expired_keys = []

            for key, entry in self.cached_responses.items():
                if "timestamp" in entry and "ttl" in entry:
                    if current_time - entry["timestamp"] > entry["ttl"]:
                        expired_keys.append(key)

            for key in expired_keys:
                del self.cached_responses[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")

    def is_feature_available(self, feature_type: FeatureType) -> bool:
        """
        Check if a specific feature is currently available.

        Args:
            feature_type: Feature to check

        Returns:
            True if feature is available, False otherwise
        """
        return feature_type not in self.disabled_features

    def get_available_features(self) -> set[FeatureType]:
        """
        Get all currently available features.

        Returns:
            Set of available features
        """
        all_features = set(FeatureType)
        return all_features - self.disabled_features

    def get_unavailable_features(self) -> set[FeatureType]:
        """
        Get all currently unavailable features.

        Returns:
            Set of unavailable features
        """
        return self.disabled_features.copy()

    async def get_fallback_response(
        self,
        request_type: str,
        request_data: dict[str, Any],
        cache_key: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get a fallback response for a degraded feature.

        Args:
            request_type: Type of request
            request_data: Request data
            cache_key: Optional cache key for response

        Returns:
            Fallback response or None if not available
        """
        try:
            # Try to get from cache first
            if cache_key and cache_key in self.cached_responses:
                entry = self.cached_responses[cache_key]
                if time.time() - entry["timestamp"] < entry["ttl"]:
                    self.cache_hits += 1
                    return entry["response"]

            self.cache_misses += 1

            # Generate fallback response based on request type
            fallback_response = await self._generate_fallback_response(
                request_type, request_data
            )

            # Cache the response if cache_key provided
            if cache_key and fallback_response:
                self.cached_responses[cache_key] = {
                    "response": fallback_response,
                    "timestamp": time.time(),
                    "ttl": 300  # 5 minutes default TTL
                }

                # Limit cache size
                if len(self.cached_responses) > 1000:
                    # Remove oldest entries
                    sorted_keys = sorted(
                        self.cached_responses.keys(),
                        key=lambda k: self.cached_responses[k]["timestamp"]
                    )
                    for key in sorted_keys[:100]:  # Remove 100 oldest entries
                        del self.cached_responses[key]

            return fallback_response

        except Exception as e:
            logger.error(f"Failed to get fallback response: {e}")
            return None

    async def _generate_fallback_response(
        self,
        request_type: str,
        request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate a fallback response for a specific request type."""
        if request_type == "search":
            return {
                "results": [],
                "message": "Search service temporarily unavailable. Please try again later.",
                "fallback": True,
                "degradation_mode": self.current_mode.name.lower()
            }
        elif request_type == "document_ingestion":
            return {
                "success": False,
                "message": "Document ingestion temporarily disabled due to system degradation.",
                "fallback": True,
                "degradation_mode": self.current_mode.name.lower()
            }
        elif request_type == "health_check":
            return {
                "status": "degraded",
                "mode": self.current_mode.name.lower(),
                "available_features": [f.value for f in self.get_available_features()],
                "message": "System operating in degraded mode",
                "fallback": True
            }

        return None

    def should_throttle_request(self, request_priority: int = 5) -> bool:
        """
        Check if a request should be throttled.

        Args:
            request_priority: Priority of the request (1=high, 10=low)

        Returns:
            True if request should be throttled, False otherwise
        """
        if not self.resource_throttle.is_active:
            return False

        # Higher priority requests (lower numbers) are less likely to be throttled
        throttle_probability = (request_priority / 10.0) * (1.0 - self.resource_throttle.throttle_factor)

        return throttle_probability > 0.5

    def get_circuit_breaker_state(self, component_id: str) -> CircuitBreakerState | None:
        """
        Get circuit breaker state for a component.

        Args:
            component_id: Component identifier

        Returns:
            Circuit breaker state or None if not found
        """
        if component_id in self.circuit_breakers:
            return self.circuit_breakers[component_id].state
        return None

    def is_component_available(self, component_id: str) -> bool:
        """
        Check if a component is available (circuit breaker allows requests).

        Args:
            component_id: Component identifier

        Returns:
            True if component is available, False otherwise
        """
        if component_id in self.circuit_breakers:
            return self.circuit_breakers[component_id].should_allow_request()
        return True

    async def record_component_success(self, component_id: str):
        """Record successful operation for a component."""
        if component_id in self.circuit_breakers:
            self.circuit_breakers[component_id].record_success()

    async def record_component_failure(self, component_id: str):
        """Record failed operation for a component."""
        if component_id in self.circuit_breakers:
            self.circuit_breakers[component_id].record_failure()

    def register_notification_handler(
        self,
        handler: Callable[[UserNotification], None]
    ):
        """Register a handler for degradation notifications."""
        self.notification_handlers.add(handler)

    async def _handle_health_notification(self, notification: UserNotification):
        """Handle health notifications from health monitor."""
        # This integrates with the LSP health monitor notifications
        # and can trigger additional degradation responses
        pass

    async def _send_degradation_notification(self, event: DegradationEvent):
        """Send notification about degradation mode change."""
        # Determine notification level based on degradation severity
        if event.degradation_mode in {DegradationMode.UNAVAILABLE, DegradationMode.EMERGENCY}:
            level = NotificationLevel.CRITICAL
        elif event.degradation_mode in {DegradationMode.OFFLINE_CLI, DegradationMode.CACHED_ONLY}:
            level = NotificationLevel.ERROR
        elif event.degradation_mode in {DegradationMode.READ_ONLY, DegradationMode.FEATURES_LIMITED}:
            level = NotificationLevel.WARNING
        else:
            level = NotificationLevel.INFO

        title = f"System Degradation: {event.degradation_mode.name.replace('_', ' ').title()}"

        if event.degradation_mode > event.previous_mode:
            message = f"System degraded from {event.previous_mode.name.lower()} to {event.degradation_mode.name.lower()} mode. {event.trigger_reason}"
        else:
            message = f"System recovered from {event.previous_mode.name.lower()} to {event.degradation_mode.name.lower()} mode. {event.trigger_reason}"

        notification = UserNotification(
            timestamp=time.time(),
            level=level,
            title=title,
            message=message,
            server_name="workspace-qdrant-mcp",
            troubleshooting_steps=event.user_guidance,
            auto_recovery_attempted=event.automatic_recovery
        )

        # Send to all registered handlers
        for handler in list(self.notification_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(f"Error in degradation notification handler: {e}")

    async def _store_degradation_event(self, event: DegradationEvent):
        """Store degradation event in coordinator."""
        if not self.coordinator:
            return

        try:
            await self.coordinator.enqueue_processing_item(
                component_id="degradation-manager",
                queue_type=ProcessingQueueType.ADMIN_COMMANDS,
                payload={
                    "event_type": "degradation_event",
                    "event_data": asdict(event)
                },
                priority=2  # High priority for degradation events
            )
        except Exception as e:
            logger.error(f"Failed to store degradation event: {e}")

    def get_degradation_status(self) -> dict[str, Any]:
        """
        Get comprehensive degradation status.

        Returns:
            Degradation status information
        """
        uptime = time.time() - self.start_time

        return {
            "current_mode": self.current_mode.name.lower(),
            "previous_mode": self.previous_mode.name.lower(),
            "uptime_seconds": uptime,
            "degradation_count": self.degradation_count,
            "recovery_count": self.recovery_count,
            "available_features": [f.name.lower() for f in self.get_available_features()],
            "unavailable_features": [f.name.lower() for f in self.get_unavailable_features()],
            "circuit_breakers": {
                comp_id: cb.state.name.lower()
                for comp_id, cb in self.circuit_breakers.items()
            },
            "resource_throttle": {
                "active": self.resource_throttle.is_active,
                "strategy": self.resource_throttle.strategy.name.lower(),
                "throttle_factor": self.resource_throttle.throttle_factor
            },
            "cache_statistics": {
                "cache_size": len(self.cached_responses),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_ratio": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            },
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "mode": event.degradation_mode.name.lower(),
                    "previous_mode": event.previous_mode.name.lower(),
                    "reason": event.trigger_reason,
                    "timestamp": event.timestamp.isoformat(),
                    "affected_features": [f.name.lower() for f in event.affected_features]
                }
                for event in self.degradation_events[-10:]  # Last 10 events
            ]
        }

    async def force_degradation_mode(
        self,
        mode: DegradationMode,
        reason: str = "Manual override"
    ):
        """
        Force a specific degradation mode (for testing or manual intervention).

        Args:
            mode: Degradation mode to set
            reason: Reason for the change
        """
        await self._change_degradation_mode(mode, reason)
        logger.warning(f"Degradation mode manually set to: {mode.name.lower()}")

    async def shutdown(self):
        """Shutdown the degradation manager."""
        logger.info("Shutting down degradation manager")

        self.shutdown_event.set()

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.monitoring_tasks.clear()
        logger.info("Degradation manager shutdown complete")

    @asynccontextmanager
    async def degradation_context(self):
        """
        Async context manager for degradation management lifecycle.

        Usage:
            async with degradation_manager.degradation_context():
                # Degradation management is active
                await do_work()
            # Degradation manager is automatically stopped
        """
        await self.initialize()
        try:
            yield self
        finally:
            await self.shutdown()


# Global degradation manager instance
_degradation_manager: DegradationManager | None = None


async def get_degradation_manager(
    lifecycle_manager: ComponentLifecycleManager | None = None,
    health_monitor: LspHealthMonitor | None = None,
    coordinator: ComponentCoordinator | None = None,
    config: dict[str, Any] | None = None
) -> DegradationManager:
    """Get or create global degradation manager instance."""
    global _degradation_manager

    if _degradation_manager is None:
        _degradation_manager = DegradationManager(
            lifecycle_manager, health_monitor, coordinator, config
        )

        if not await _degradation_manager.initialize():
            raise RuntimeError("Failed to initialize degradation manager")

    return _degradation_manager


async def shutdown_degradation_manager():
    """Shutdown global degradation manager."""
    global _degradation_manager

    if _degradation_manager:
        await _degradation_manager.shutdown()
        _degradation_manager = None
