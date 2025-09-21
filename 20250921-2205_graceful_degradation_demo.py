#!/usr/bin/env python3
"""
Graceful Degradation System Demo

This script demonstrates the graceful degradation strategies implemented
for the workspace-qdrant-mcp system, showing how the system maintains
functionality under various failure scenarios.
"""

import asyncio
import time
from typing import Dict, Any

# Import graceful degradation components
from src.python.common.core.graceful_degradation import (
    DegradationManager,
    DegradationMode,
    FeatureType,
    CircuitBreaker,
    CircuitBreakerState,
)
from src.python.common.core.component_coordination import ComponentType
from src.python.workspace_qdrant_mcp.tools.degradation_aware import (
    DegradationAwareMCPTools,
    CLIDegradationHandler,
)


class GracefulDegradationDemo:
    """Demo class for graceful degradation system."""

    def __init__(self):
        self.degradation_manager = None
        self.mcp_tools = None
        self.cli_handler = None

    async def initialize(self):
        """Initialize the demo system."""
        print("🚀 Initializing Graceful Degradation Demo System...")

        # Create degradation manager
        self.degradation_manager = DegradationManager()
        await self.degradation_manager.initialize()

        # Create mock original tools
        original_tools = {
            "search_workspace": self._mock_search_tool,
            "add_document": self._mock_add_document_tool,
            "list_collections": self._mock_list_collections_tool,
            "get_workspace_status": self._mock_status_tool,
        }

        # Create degradation-aware tools
        self.mcp_tools = DegradationAwareMCPTools(
            self.degradation_manager, original_tools
        )

        # Create CLI handler
        self.cli_handler = CLIDegradationHandler(self.degradation_manager)

        print("✅ Demo system initialized successfully!")
        print()

    async def demo_normal_operation(self):
        """Demonstrate normal operation mode."""
        print("📋 Demo 1: Normal Operation Mode")
        print("-" * 40)

        # Show current status
        await self._show_system_status()

        # Test search operation
        print("🔍 Testing search operation...")
        result = await self.mcp_tools.search_workspace("machine learning")
        self._print_operation_result("Search", result)

        # Test document addition
        print("\n📄 Testing document addition...")
        result = await self.mcp_tools.add_document(
            "This is a test document about AI.", "ai-docs"
        )
        self._print_operation_result("Add Document", result)

        print("\n" + "=" * 60 + "\n")

    async def demo_component_failure(self):
        """Demonstrate component failure scenario."""
        print("📋 Demo 2: Component Failure Scenario")
        print("-" * 40)

        print("💥 Simulating Rust daemon failure...")
        await self.degradation_manager.force_degradation_mode(
            DegradationMode.CACHED_ONLY,
            "Rust daemon connection lost"
        )

        await self._show_system_status()

        # Test search with caching
        print("🔍 Testing search with cache fallback...")
        result = await self.mcp_tools.search_workspace("machine learning")
        self._print_operation_result("Search (Degraded)", result)

        print("\n📄 Testing document addition (should fail)...")
        result = await self.mcp_tools.add_document(
            "This document won't be added.", "ai-docs"
        )
        self._print_operation_result("Add Document (Degraded)", result)

        print("\n" + "=" * 60 + "\n")

    async def demo_offline_mode(self):
        """Demonstrate offline mode scenario."""
        print("📋 Demo 3: Offline Mode Scenario")
        print("-" * 40)

        print("🌐 Simulating network connectivity loss...")
        await self.degradation_manager.force_degradation_mode(
            DegradationMode.OFFLINE_CLI,
            "Network connectivity lost"
        )

        await self._show_system_status()

        # Test CLI operations in offline mode
        print("💻 Testing CLI search in offline mode...")
        result = await self.cli_handler.handle_search_command("machine learning")
        self._print_cli_result("CLI Search", result)

        print("\n📊 Testing CLI status command...")
        result = await self.cli_handler.handle_status_command(verbose=True)
        self._print_cli_result("CLI Status", result)

        print("\n" + "=" * 60 + "\n")

    async def demo_circuit_breaker(self):
        """Demonstrate circuit breaker functionality."""
        print("📋 Demo 4: Circuit Breaker Pattern")
        print("-" * 40)

        # Reset to normal mode first
        await self.degradation_manager.force_degradation_mode(
            DegradationMode.NORMAL,
            "Reset for circuit breaker demo"
        )

        print("⚡ Simulating repeated service failures...")

        # Simulate multiple failures to trigger circuit breaker
        for i in range(6):
            await self.degradation_manager.record_component_failure("search-service")
            print(f"   Failure {i+1} recorded")

        # Show circuit breaker status
        cb_state = self.degradation_manager.get_circuit_breaker_state("search-service")
        print(f"\n🔌 Circuit breaker state: {cb_state}")

        # Test operation with circuit breaker open
        print("🔍 Testing search with circuit breaker open...")
        result = await self.mcp_tools.search_workspace("test query")
        self._print_operation_result("Search (Circuit Open)", result)

        print("\n" + "=" * 60 + "\n")

    async def demo_progressive_recovery(self):
        """Demonstrate progressive recovery scenario."""
        print("📋 Demo 5: Progressive Recovery")
        print("-" * 40)

        print("🔄 Simulating system recovery...")

        # Gradual recovery from emergency mode
        recovery_modes = [
            DegradationMode.EMERGENCY,
            DegradationMode.CACHED_ONLY,
            DegradationMode.READ_ONLY,
            DegradationMode.FEATURES_LIMITED,
            DegradationMode.NORMAL,
        ]

        for mode in recovery_modes:
            print(f"\n📈 Recovering to {mode.name.lower().replace('_', ' ')} mode...")
            await self.degradation_manager.force_degradation_mode(
                mode, f"Recovery step to {mode.name}"
            )

            await self._show_system_status()
            await asyncio.sleep(1)  # Brief pause for demo effect

        print("\n✅ Full recovery completed!")
        print("\n" + "=" * 60 + "\n")

    async def demo_feature_availability(self):
        """Demonstrate feature availability checking."""
        print("📋 Demo 6: Feature Availability Analysis")
        print("-" * 40)

        # Test different degradation modes and feature availability
        test_modes = [
            DegradationMode.NORMAL,
            DegradationMode.FEATURES_LIMITED,
            DegradationMode.READ_ONLY,
            DegradationMode.OFFLINE_CLI,
        ]

        for mode in test_modes:
            await self.degradation_manager.force_degradation_mode(
                mode, f"Testing {mode.name}"
            )

            print(f"\n🔧 Mode: {mode.name.lower().replace('_', ' ').title()}")

            # Check key features
            features_to_check = [
                FeatureType.SEMANTIC_SEARCH,
                FeatureType.DOCUMENT_INGESTION,
                FeatureType.MCP_SERVER,
                FeatureType.CLI_OPERATIONS,
            ]

            for feature in features_to_check:
                available = self.degradation_manager.is_feature_available(feature)
                status = "✅" if available else "❌"
                print(f"   {status} {feature.name.lower().replace('_', ' ').title()}")

        print("\n" + "=" * 60 + "\n")

    async def demo_statistics_and_monitoring(self):
        """Demonstrate statistics and monitoring capabilities."""
        print("📋 Demo 7: Statistics and Monitoring")
        print("-" * 40)

        # Show degradation statistics
        degradation_status = self.degradation_manager.get_degradation_status()
        print("📊 Degradation Manager Statistics:")
        print(f"   Current Mode: {degradation_status['current_mode']}")
        print(f"   Uptime: {degradation_status['uptime_seconds']:.1f} seconds")
        print(f"   Degradation Events: {degradation_status['degradation_count']}")
        print(f"   Recovery Events: {degradation_status['recovery_count']}")

        # Show tool statistics
        tool_stats = self.mcp_tools.get_tool_statistics()
        print(f"\n🔧 MCP Tools Statistics:")
        print(f"   Total Executions: {tool_stats['total_executions']}")
        print(f"   Degraded Executions: {tool_stats['degraded_executions']}")
        print(f"   Cache Hits: {tool_stats['cache_hits']}")
        print(f"   Degradation Rate: {tool_stats['degradation_rate']:.1%}")

        # Show recent events
        recent_events = degradation_status.get("recent_events", [])
        if recent_events:
            print(f"\n📝 Recent Events:")
            for event in recent_events[-3:]:  # Last 3 events
                print(f"   • {event['reason']} -> {event['mode']}")

        print("\n" + "=" * 60 + "\n")

    async def _show_system_status(self):
        """Show current system status."""
        status = self.degradation_manager.get_degradation_status()
        mode = status["current_mode"]

        print(f"🎯 System Status: {mode.upper()}")

        available_features = status["available_features"]
        unavailable_features = status["unavailable_features"]

        if available_features:
            print(f"✅ Available: {', '.join(available_features[:3])}{'...' if len(available_features) > 3 else ''}")

        if unavailable_features:
            print(f"❌ Unavailable: {', '.join(unavailable_features[:3])}{'...' if len(unavailable_features) > 3 else ''}")

        print()

    def _print_operation_result(self, operation: str, result):
        """Print MCP operation result."""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"   {status}: {operation}")

        if result.success:
            print(f"   Data: {str(result.data)[:50]}{'...' if len(str(result.data)) > 50 else ''}")
            if result.from_cache:
                print("   📦 Source: Cache")
        else:
            print(f"   Error: {result.error_message}")

        if result.user_guidance:
            print(f"   💡 Guidance: {result.user_guidance}")

    def _print_cli_result(self, operation: str, result):
        """Print CLI operation result."""
        print(f"   🖥️  {operation}:")
        if isinstance(result, dict):
            for key, value in list(result.items())[:3]:  # Show first 3 items
                if isinstance(value, list):
                    print(f"      {key}: {len(value)} items")
                else:
                    print(f"      {key}: {value}")

    # Mock tool implementations
    async def _mock_search_tool(self, **kwargs):
        """Mock search tool implementation."""
        query = kwargs.get("query", "")
        return {
            "results": [
                {"content": f"Result for '{query}'", "score": 0.95},
                {"content": f"Another result about '{query}'", "score": 0.87},
            ],
            "total": 2,
            "mode": "mock_search"
        }

    async def _mock_add_document_tool(self, **kwargs):
        """Mock document addition tool."""
        content = kwargs.get("content", "")
        collection = kwargs.get("collection", "")
        return {
            "success": True,
            "document_id": f"doc_{int(time.time())}",
            "collection": collection,
            "chunks_added": len(content.split()) // 50 + 1
        }

    async def _mock_list_collections_tool(self, **kwargs):
        """Mock list collections tool."""
        return ["ai-docs", "research-papers", "code-samples"]

    async def _mock_status_tool(self, **kwargs):
        """Mock status tool."""
        return {
            "status": "healthy",
            "components": {
                "rust_daemon": "operational",
                "python_mcp_server": "operational",
                "cli_utility": "ready"
            },
            "collections": 3,
            "documents": 1247
        }

    async def shutdown(self):
        """Shutdown demo system."""
        if self.degradation_manager:
            await self.degradation_manager.shutdown()
        print("👋 Demo system shutdown complete!")


async def main():
    """Run the graceful degradation demo."""
    print("🎭 Workspace Qdrant MCP - Graceful Degradation System Demo")
    print("=" * 60)
    print()

    demo = GracefulDegradationDemo()

    try:
        await demo.initialize()

        # Run all demo scenarios
        await demo.demo_normal_operation()
        await demo.demo_component_failure()
        await demo.demo_offline_mode()
        await demo.demo_circuit_breaker()
        await demo.demo_progressive_recovery()
        await demo.demo_feature_availability()
        await demo.demo_statistics_and_monitoring()

        print("🎉 All degradation scenarios demonstrated successfully!")
        print("\nKey Benefits of Graceful Degradation:")
        print("• ✅ System remains functional during component failures")
        print("• 🔄 Automatic fallback mechanisms preserve user experience")
        print("• 💡 Clear user guidance for degraded states")
        print("• 📊 Comprehensive monitoring and statistics")
        print("• 🛡️  Circuit breaker patterns prevent cascade failures")
        print("• 🚀 Seamless recovery as components become available")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await demo.shutdown()


if __name__ == "__main__":
    asyncio.run(main())