#!/usr/bin/env python3
"""
Component Lifecycle Management Demonstration Script.

This script demonstrates the component lifecycle management capabilities
including startup sequences, dependency management, health monitoring,
and graceful shutdown procedures.

Usage:
    python scripts/component_lifecycle_demo.py [command] [options]

Commands:
    startup     - Execute complete startup sequence
    shutdown    - Execute graceful shutdown sequence
    status      - Show component status and health
    restart     - Restart specific component
    monitor     - Monitor component health in real-time
    test        - Run lifecycle validation tests

Examples:
    python scripts/component_lifecycle_demo.py startup
    python scripts/component_lifecycle_demo.py status
    python scripts/component_lifecycle_demo.py restart rust_daemon
    python scripts/component_lifecycle_demo.py monitor --duration 60
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from common.core.component_lifecycle import (
    ComponentLifecycleManager,
    ComponentConfig,
    ComponentType,
    ComponentState,
    LifecyclePhase,
    StartupDependency,
    get_lifecycle_manager,
    shutdown_lifecycle_manager,
)
from loguru import logger


class ComponentLifecycleDemo:
    """Component Lifecycle Management Demonstration."""

    def __init__(
        self,
        db_path: str = "demo_workspace_state.db",
        project_name: str = "lifecycle_demo",
        project_path: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize the demo."""
        self.db_path = db_path
        self.project_name = project_name
        self.project_path = project_path or str(Path.cwd())
        self.verbose = verbose

        # Configure logging
        if verbose:
            logger.add(
                sys.stdout,
                level="DEBUG",
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            )
        else:
            logger.add(
                sys.stdout,
                level="INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
            )

        self.lifecycle_manager: Optional[ComponentLifecycleManager] = None

    async def startup_command(self, **kwargs) -> bool:
        """Execute complete startup sequence."""
        try:
            print("🚀 Starting Component Lifecycle Demo - Startup Sequence")
            print(f"Project: {self.project_name}")
            print(f"Database: {self.db_path}")
            print("-" * 60)

            # Create custom configurations for demo
            custom_configs = self._create_demo_configs()

            # Initialize lifecycle manager
            self.lifecycle_manager = ComponentLifecycleManager(
                db_path=self.db_path,
                project_name=self.project_name,
                project_path=self.project_path,
                component_configs=custom_configs
            )

            print("📋 Initializing Component Lifecycle Manager...")
            if not await self.lifecycle_manager.initialize():
                print("❌ Failed to initialize lifecycle manager")
                return False

            print("✅ Lifecycle manager initialized successfully")
            print()

            # Show startup order
            print("📊 Startup Order:")
            for i, dependency in enumerate(self.lifecycle_manager.STARTUP_ORDER, 1):
                components = [
                    comp_type.value for comp_type, config in custom_configs.items()
                    if config.startup_dependency == dependency
                ]
                if dependency.value == "sqlite_manager":
                    components = ["SQLite State Manager"]

                print(f"  {i}. {dependency.value}: {', '.join(components)}")
            print()

            # Execute startup sequence
            print("🔄 Executing startup sequence...")
            start_time = time.time()

            success = await self.lifecycle_manager.startup_sequence()

            duration = time.time() - start_time

            if success:
                print(f"✅ Startup sequence completed successfully in {duration:.2f}s")
                print(f"Current phase: {self.lifecycle_manager.current_phase.value}")
                print()

                # Show component status
                await self._show_component_status()
                return True
            else:
                print(f"❌ Startup sequence failed after {duration:.2f}s")
                return False

        except Exception as e:
            print(f"❌ Startup command failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def shutdown_command(self, **kwargs) -> bool:
        """Execute graceful shutdown sequence."""
        try:
            print("🛑 Starting Component Lifecycle Demo - Shutdown Sequence")
            print("-" * 60)

            if not self.lifecycle_manager:
                # Try to get existing lifecycle manager
                try:
                    self.lifecycle_manager = await get_lifecycle_manager(
                        db_path=self.db_path,
                        project_name=self.project_name,
                        project_path=self.project_path
                    )
                except Exception as e:
                    print(f"⚠️  No active lifecycle manager found: {e}")
                    return True

            # Show current status before shutdown
            print("📊 Current Component Status:")
            await self._show_component_status()
            print()

            # Execute shutdown sequence
            print("🔄 Executing shutdown sequence...")
            start_time = time.time()

            success = await self.lifecycle_manager.shutdown_sequence()

            duration = time.time() - start_time

            if success:
                print(f"✅ Shutdown sequence completed successfully in {duration:.2f}s")
                print(f"Final phase: {self.lifecycle_manager.current_phase.value}")
            else:
                print(f"❌ Shutdown sequence failed after {duration:.2f}s")

            # Global cleanup
            await shutdown_lifecycle_manager()
            print("🧹 Global lifecycle manager shutdown complete")

            return success

        except Exception as e:
            print(f"❌ Shutdown command failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def status_command(self, **kwargs) -> bool:
        """Show component status and health."""
        try:
            print("📊 Component Lifecycle Status")
            print("-" * 60)

            if not self.lifecycle_manager:
                try:
                    self.lifecycle_manager = await get_lifecycle_manager(
                        db_path=self.db_path,
                        project_name=self.project_name,
                        project_path=self.project_path
                    )
                except Exception as e:
                    print(f"⚠️  No active lifecycle manager found: {e}")
                    print("💡 Try running 'startup' command first")
                    return False

            # Get comprehensive status
            status = await self.lifecycle_manager.get_component_status()

            # Display lifecycle manager status
            lm_status = status["lifecycle_manager"]
            print(f"📋 Lifecycle Manager:")
            print(f"  Phase: {lm_status['current_phase']}")
            print(f"  Project: {lm_status['project_name']}")
            print(f"  Database: {self.db_path}")

            if lm_status.get("startup_time"):
                startup_time = datetime.fromisoformat(lm_status["startup_time"])
                print(f"  Started: {startup_time.strftime('%Y-%m-%d %H:%M:%S')}")

            print()

            # Display component status
            print("🔧 Components:")
            components = status["components"]

            for comp_name, comp_status in components.items():
                state = comp_status["state"]
                active = comp_status["instance_active"]

                # Choose appropriate emoji
                if state == ComponentState.OPERATIONAL.value:
                    emoji = "✅"
                elif state == ComponentState.STARTING.value:
                    emoji = "🔄"
                elif state == ComponentState.FAILED.value:
                    emoji = "❌"
                elif state == ComponentState.DEGRADED.value:
                    emoji = "⚠️"
                elif state == ComponentState.STOPPED.value:
                    emoji = "⏹️"
                else:
                    emoji = "❓"

                print(f"  {emoji} {comp_name}")
                print(f"    State: {state}")
                print(f"    Instance Active: {active}")

                if comp_status.get("coordinator_info"):
                    coord_info = comp_status["coordinator_info"]
                    print(f"    Health: {coord_info.get('health', 'unknown')}")
                    print(f"    Status: {coord_info.get('status', 'unknown')}")

                print()

            # Display recent events
            events = status.get("startup_events", [])
            if events:
                print("📝 Recent Events:")
                for event in events[-5:]:  # Show last 5 events
                    timestamp = datetime.fromisoformat(event["timestamp"])
                    success_emoji = "✅" if event["success"] else "❌"
                    print(f"  {success_emoji} [{timestamp.strftime('%H:%M:%S')}] {event['component_id']}: {event['message']}")

            return True

        except Exception as e:
            print(f"❌ Status command failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def restart_command(self, component_name: str, **kwargs) -> bool:
        """Restart specific component."""
        try:
            print(f"🔄 Restarting Component: {component_name}")
            print("-" * 60)

            if not self.lifecycle_manager:
                try:
                    self.lifecycle_manager = await get_lifecycle_manager(
                        db_path=self.db_path,
                        project_name=self.project_name,
                        project_path=self.project_path
                    )
                except Exception as e:
                    print(f"⚠️  No active lifecycle manager found: {e}")
                    return False

            # Map component name to type
            component_map = {
                "rust_daemon": ComponentType.RUST_DAEMON,
                "python_mcp_server": ComponentType.PYTHON_MCP_SERVER,
                "cli_utility": ComponentType.CLI_UTILITY,
                "context_injector": ComponentType.CONTEXT_INJECTOR,
            }

            component_type = component_map.get(component_name.lower())
            if not component_type:
                print(f"❌ Unknown component: {component_name}")
                print(f"Available components: {', '.join(component_map.keys())}")
                return False

            print(f"🔄 Restarting {component_type.value}...")
            start_time = time.time()

            success = await self.lifecycle_manager.restart_component(component_type)

            duration = time.time() - start_time

            if success:
                print(f"✅ Component restarted successfully in {duration:.2f}s")
            else:
                print(f"❌ Component restart failed after {duration:.2f}s")

            return success

        except Exception as e:
            print(f"❌ Restart command failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def monitor_command(self, duration: int = 30, interval: float = 2.0, **kwargs) -> bool:
        """Monitor component health in real-time."""
        try:
            print(f"👁️  Monitoring Component Health for {duration}s")
            print(f"Update interval: {interval}s")
            print("-" * 60)

            if not self.lifecycle_manager:
                try:
                    self.lifecycle_manager = await get_lifecycle_manager(
                        db_path=self.db_path,
                        project_name=self.project_name,
                        project_path=self.project_path
                    )
                except Exception as e:
                    print(f"⚠️  No active lifecycle manager found: {e}")
                    return False

            start_time = time.time()
            iteration = 0

            while (time.time() - start_time) < duration:
                iteration += 1
                elapsed = time.time() - start_time

                # Clear screen (simple version)
                if iteration > 1:
                    print("\n" + "="*60)

                print(f"🕐 Monitoring Update #{iteration} (Elapsed: {elapsed:.1f}s)")
                print(f"📊 Component Health Status:")

                # Get current status
                status = await self.lifecycle_manager.get_component_status()
                components = status["components"]

                for comp_name, comp_status in components.items():
                    state = comp_status["state"]
                    active = comp_status["instance_active"]

                    # Health indicator
                    if state == ComponentState.OPERATIONAL.value and active:
                        health_indicator = "🟢 HEALTHY"
                    elif state == ComponentState.DEGRADED.value:
                        health_indicator = "🟡 DEGRADED"
                    elif state == ComponentState.FAILED.value:
                        health_indicator = "🔴 FAILED"
                    elif state == ComponentState.STARTING.value:
                        health_indicator = "🔵 STARTING"
                    else:
                        health_indicator = "⚪ UNKNOWN"

                    print(f"  {health_indicator} {comp_name}: {state}")

                # Wait for next update
                await asyncio.sleep(interval)

            print(f"\n✅ Monitoring completed after {duration}s")
            return True

        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            return True
        except Exception as e:
            print(f"❌ Monitor command failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def test_command(self, **kwargs) -> bool:
        """Run lifecycle validation tests."""
        try:
            print("🧪 Running Component Lifecycle Validation Tests")
            print("-" * 60)

            tests_passed = 0
            tests_total = 0

            # Test 1: Configuration Validation
            tests_total += 1
            print("1. Testing configuration validation...")
            try:
                manager = ComponentLifecycleManager(
                    db_path="test.db",
                    project_name="test_project"
                )

                # Check configurations are valid
                assert len(manager.component_configs) == 4
                assert ComponentType.RUST_DAEMON in manager.component_configs
                assert ComponentType.PYTHON_MCP_SERVER in manager.component_configs

                print("   ✅ Configuration validation passed")
                tests_passed += 1
            except Exception as e:
                print(f"   ❌ Configuration validation failed: {e}")

            # Test 2: Startup Order Validation
            tests_total += 1
            print("2. Testing startup order validation...")
            try:
                manager = ComponentLifecycleManager()

                expected_order = [
                    StartupDependency.SQLITE_MANAGER,
                    StartupDependency.RUST_DAEMON,
                    StartupDependency.PYTHON_MCP_SERVER,
                    StartupDependency.CLI_CONTEXT_INJECTOR,
                ]

                assert manager.STARTUP_ORDER == expected_order
                assert manager.SHUTDOWN_ORDER == list(reversed(expected_order))

                print("   ✅ Startup order validation passed")
                tests_passed += 1
            except Exception as e:
                print(f"   ❌ Startup order validation failed: {e}")

            # Test 3: Event Logging
            tests_total += 1
            print("3. Testing event logging...")
            try:
                from unittest.mock import AsyncMock, patch

                with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
                    mock_coordinator = AsyncMock()
                    mock_coordinator.initialize = AsyncMock(return_value=True)
                    mock_coordinator.enqueue_processing_item = AsyncMock(return_value="queue-item-id")
                    mock_get_coordinator.return_value = mock_coordinator

                    manager = ComponentLifecycleManager(db_path="test.db")
                    await manager.initialize()

                    # Test logging an event
                    await manager._log_lifecycle_event(
                        component_id="test_component",
                        phase=LifecyclePhase.COMPONENT_STARTUP,
                        event_type="test",
                        message="Test event"
                    )

                    assert len(manager.startup_events) >= 1
                    print("   ✅ Event logging validation passed")
                    tests_passed += 1
            except Exception as e:
                print(f"   ❌ Event logging validation failed: {e}")

            # Test 4: Component State Management
            tests_total += 1
            print("4. Testing component state management...")
            try:
                manager = ComponentLifecycleManager()

                # Check initial states
                for comp_type in manager.component_configs.keys():
                    assert manager.component_states[comp_type] == ComponentState.NOT_STARTED

                print("   ✅ Component state management passed")
                tests_passed += 1
            except Exception as e:
                print(f"   ❌ Component state management failed: {e}")

            # Results
            print(f"\n📊 Test Results:")
            print(f"  Passed: {tests_passed}/{tests_total}")
            print(f"  Success Rate: {(tests_passed/tests_total)*100:.1f}%")

            if tests_passed == tests_total:
                print("✅ All tests passed!")
                return True
            else:
                print("❌ Some tests failed")
                return False

        except Exception as e:
            print(f"❌ Test command failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def _show_component_status(self):
        """Helper to show component status."""
        try:
            status = await self.lifecycle_manager.get_component_status()
            components = status["components"]

            for comp_name, comp_status in components.items():
                state = comp_status["state"]
                active = comp_status["instance_active"]

                if state == ComponentState.OPERATIONAL.value and active:
                    emoji = "✅"
                elif state == ComponentState.STARTING.value:
                    emoji = "🔄"
                elif state == ComponentState.FAILED.value:
                    emoji = "❌"
                else:
                    emoji = "⚠️"

                print(f"  {emoji} {comp_name}: {state} (Active: {active})")

        except Exception as e:
            print(f"⚠️  Error getting component status: {e}")

    def _create_demo_configs(self) -> Dict[ComponentType, ComponentConfig]:
        """Create demo-specific component configurations."""
        return {
            ComponentType.RUST_DAEMON: ComponentConfig(
                component_type=ComponentType.RUST_DAEMON,
                startup_dependency=StartupDependency.RUST_DAEMON,
                startup_timeout=20.0,  # Shorter for demo
                shutdown_timeout=10.0,
                readiness_checks=[
                    "grpc_server_responsive",
                    "sqlite_connection_active",
                    "process_health_ok"
                ],
                config_overrides={"demo_mode": True}
            ),
            ComponentType.PYTHON_MCP_SERVER: ComponentConfig(
                component_type=ComponentType.PYTHON_MCP_SERVER,
                startup_dependency=StartupDependency.PYTHON_MCP_SERVER,
                startup_timeout=15.0,  # Shorter for demo
                shutdown_timeout=8.0,
                readiness_checks=[
                    "mcp_server_listening",
                    "qdrant_connection_active",
                    "grpc_client_connected"
                ],
                config_overrides={"demo_mode": True}
            ),
            ComponentType.CLI_UTILITY: ComponentConfig(
                component_type=ComponentType.CLI_UTILITY,
                startup_dependency=StartupDependency.CLI_CONTEXT_INJECTOR,
                startup_timeout=10.0,
                shutdown_timeout=5.0,
                readiness_checks=[
                    "cli_commands_available",
                    "config_validation_passed"
                ],
                config_overrides={"demo_mode": True}
            ),
            ComponentType.CONTEXT_INJECTOR: ComponentConfig(
                component_type=ComponentType.CONTEXT_INJECTOR,
                startup_dependency=StartupDependency.CLI_CONTEXT_INJECTOR,
                startup_timeout=10.0,
                shutdown_timeout=5.0,
                readiness_checks=[
                    "context_hooks_registered",
                    "mcp_server_accessible"
                ],
                config_overrides={"demo_mode": True}
            ),
        }


async def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Component Lifecycle Management Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "command",
        choices=["startup", "shutdown", "status", "restart", "monitor", "test"],
        help="Command to execute"
    )

    parser.add_argument(
        "component",
        nargs="?",
        help="Component name for restart command"
    )

    parser.add_argument(
        "--db-path",
        default="demo_workspace_state.db",
        help="Database path for state management"
    )

    parser.add_argument(
        "--project-name",
        default="lifecycle_demo",
        help="Project name for component scoping"
    )

    parser.add_argument(
        "--project-path",
        help="Project path for workspace detection"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration for monitor command (seconds)"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Update interval for monitor command (seconds)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate restart command
    if args.command == "restart" and not args.component:
        print("❌ Component name required for restart command")
        print("Available components: rust_daemon, python_mcp_server, cli_utility, context_injector")
        sys.exit(1)

    # Create demo instance
    demo = ComponentLifecycleDemo(
        db_path=args.db_path,
        project_name=args.project_name,
        project_path=args.project_path,
        verbose=args.verbose
    )

    # Execute command
    try:
        if args.command == "startup":
            success = await demo.startup_command()
        elif args.command == "shutdown":
            success = await demo.shutdown_command()
        elif args.command == "status":
            success = await demo.status_command()
        elif args.command == "restart":
            success = await demo.restart_command(args.component)
        elif args.command == "monitor":
            success = await demo.monitor_command(
                duration=args.duration,
                interval=args.interval
            )
        elif args.command == "test":
            success = await demo.test_command()
        else:
            print(f"❌ Unknown command: {args.command}")
            success = False

        if not success:
            sys.exit(1)

        print("\n🎉 Demo completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())