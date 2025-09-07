"""
LSP Health Monitor Integration Demo

This script demonstrates the complete LSP Health Monitor system with:
- Multi-server monitoring with different LSP types
- Health check scheduling and recovery attempts
- User notification handling with troubleshooting guidance
- Integration with AsyncioLspClient and PriorityQueueManager
- Graceful degradation and fallback mode demonstration

Run this demo to see the health monitor in action.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List

from workspace_qdrant_mcp.core.lsp_health_monitor import (
    LspHealthMonitor,
    HealthCheckConfig,
    HealthStatus,
    NotificationLevel,
    UserNotification,
)
from workspace_qdrant_mcp.core.lsp_client import AsyncioLspClient, ConnectionState
from workspace_qdrant_mcp.core.priority_queue_manager import PriorityQueueManager
from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager


# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoNotificationHandler:
    """Demo notification handler that prints user-friendly messages"""
    
    def __init__(self):
        self.notifications: List[UserNotification] = []
    
    def handle_notification(self, notification: UserNotification) -> None:
        """Handle LSP health notifications"""
        self.notifications.append(notification)
        
        # Print colored output based on notification level
        colors = {
            NotificationLevel.INFO: "\033[92m",      # Green
            NotificationLevel.WARNING: "\033[93m",   # Yellow  
            NotificationLevel.ERROR: "\033[91m",     # Red
            NotificationLevel.CRITICAL: "\033[95m",  # Magenta
        }
        reset = "\033[0m"
        
        color = colors.get(notification.level, "")
        
        print(f"\n{color}🔔 LSP Health Alert{reset}")
        print(f"   Server: {notification.server_name}")
        print(f"   Level:  {notification.level.value.upper()}")
        print(f"   Title:  {notification.title}")
        print(f"   Message: {notification.message}")
        
        if notification.troubleshooting_steps:
            print(f"   📋 Troubleshooting Steps:")
            for i, step in enumerate(notification.troubleshooting_steps, 1):
                print(f"      {i}. {step}")
        
        if notification.auto_recovery_attempted:
            print(f"   🔧 Auto-recovery was attempted")
        
        print()


class MockLspServer:
    """Mock LSP server that can simulate different health states"""
    
    def __init__(self, name: str, initial_health: HealthStatus = HealthStatus.HEALTHY):
        self.name = name
        self.health = initial_health
        self.response_delay = 0.1  # Default response time
        self.failure_count = 0
        self.connected = True
        
    async def workspace_symbol(self, query: str) -> List:
        """Simulate workspace symbol request"""
        await asyncio.sleep(self.response_delay)
        
        if not self.connected:
            raise ConnectionError("Server not connected")
        
        if self.health == HealthStatus.FAILED:
            self.failure_count += 1
            raise Exception(f"Server failed (failure #{self.failure_count})")
        elif self.health == HealthStatus.UNHEALTHY:
            if self.failure_count % 3 == 0:  # Intermittent failures
                self.failure_count += 1
                raise Exception("Intermittent server error")
        
        return []  # Return empty result for demo
    
    def set_health(self, health: HealthStatus, response_delay: float = 0.1):
        """Change server health state"""
        self.health = health
        self.response_delay = response_delay
        logger.info(f"📊 Mock server '{self.name}' health set to {health.value}")
    
    def disconnect(self):
        """Simulate server disconnection"""
        self.connected = False
        logger.info(f"🔌 Mock server '{self.name}' disconnected")
    
    def reconnect(self):
        """Simulate server reconnection"""
        self.connected = True
        self.health = HealthStatus.HEALTHY
        self.response_delay = 0.1
        self.failure_count = 0
        logger.info(f"🔌 Mock server '{self.name}' reconnected")


async def create_mock_lsp_client(server_name: str, mock_server: MockLspServer) -> AsyncioLspClient:
    """Create a mock LSP client for demonstration"""
    client = AsyncioLspClient(server_name=server_name)
    
    # Patch the client methods to use our mock server
    client.workspace_symbol = mock_server.workspace_symbol
    client.is_connected = lambda: mock_server.connected
    
    # Mock server capabilities
    from unittest.mock import Mock
    capabilities = Mock()
    capabilities.supports_hover.return_value = True
    capabilities.supports_definition.return_value = True
    capabilities.supports_references.return_value = True
    capabilities.supports_document_symbol.return_value = True
    capabilities.supports_workspace_symbol.return_value = True
    client.server_capabilities = capabilities
    
    return client


async def demonstrate_health_monitoring():
    """Demonstrate comprehensive LSP health monitoring"""
    print("🚀 Starting LSP Health Monitor Demonstration")
    print("=" * 60)
    
    # Create health monitor configuration for fast demo
    config = HealthCheckConfig(
        check_interval=3.0,           # Check every 3 seconds
        fast_check_interval=1.0,      # Fast check every 1 second when issues
        health_check_timeout=2.0,     # 2-second timeout
        consecutive_failures_threshold=2,  # 2 failures before unhealthy
        recovery_success_threshold=2,      # 2 successes to mark recovered
        max_recovery_attempts=3,           # Max 3 recovery attempts
        base_backoff_delay=0.5,           # Start with 0.5s backoff
        max_backoff_delay=5.0,            # Max 5s backoff
        enable_auto_recovery=True,
        enable_fallback_mode=True,
        enable_user_notifications=True,
    )
    
    # Create notification handler
    notification_handler = DemoNotificationHandler()
    
    # Create health monitor
    health_monitor = LspHealthMonitor(config=config)
    health_monitor.register_notification_handler(notification_handler.handle_notification)
    
    # Create mock servers with different behaviors
    python_server = MockLspServer("python-lsp", HealthStatus.HEALTHY)
    rust_server = MockLspServer("rust-analyzer", HealthStatus.HEALTHY)  
    typescript_server = MockLspServer("typescript-lsp", HealthStatus.DEGRADED)
    
    # Create LSP clients
    python_client = await create_mock_lsp_client("python-lsp", python_server)
    rust_client = await create_mock_lsp_client("rust-analyzer", rust_server)
    typescript_client = await create_mock_lsp_client("typescript-lsp", typescript_server)
    
    # Register servers with health monitor
    health_monitor.register_server("python-lsp", python_client)
    health_monitor.register_server("rust-analyzer", rust_client) 
    health_monitor.register_server("typescript-lsp", typescript_client)
    
    print("📝 Registered 3 LSP servers for monitoring:")
    print("   • python-lsp (healthy)")
    print("   • rust-analyzer (healthy)")  
    print("   • typescript-lsp (degraded)")
    print()
    
    try:
        async with health_monitor.monitoring_context():
            print("🔍 Starting health monitoring...")
            
            # Phase 1: Normal operation (5 seconds)
            print("\n📈 Phase 1: Normal Operation (5s)")
            await asyncio.sleep(5)
            
            # Show initial statistics
            stats = health_monitor.get_health_statistics()
            print(f"   Healthy servers: {stats['healthy_servers']}/{stats['total_servers']}")
            print(f"   Available features: {len(stats['available_features'])}")
            
            # Phase 2: Introduce performance degradation (5 seconds)
            print("\n⚠️  Phase 2: Performance Degradation (5s)")
            typescript_server.set_health(HealthStatus.DEGRADED, response_delay=1.5)  # Slow responses
            rust_server.set_health(HealthStatus.DEGRADED, response_delay=2.0)        # Very slow responses
            await asyncio.sleep(5)
            
            # Phase 3: Server failure and recovery (10 seconds)
            print("\n💥 Phase 3: Server Failures and Recovery (10s)")
            python_server.set_health(HealthStatus.FAILED)  # Complete failure
            rust_server.disconnect()                       # Disconnection
            await asyncio.sleep(8)
            
            # Attempt manual recovery
            print("🔧 Attempting manual recovery...")
            python_server.set_health(HealthStatus.HEALTHY)  # Fix python server
            rust_server.reconnect()                         # Reconnect rust server
            await asyncio.sleep(2)
            
            # Phase 4: Fallback mode demonstration (5 seconds)
            print("\n🛡️ Phase 4: Fallback Mode Testing (5s)")
            # Simulate persistent failure
            typescript_server.set_health(HealthStatus.FAILED)
            
            # Let health monitor detect and enter fallback mode
            await asyncio.sleep(5)
            
            # Show final statistics
            print("\n📊 Final Health Statistics:")
            stats = health_monitor.get_health_statistics()
            
            print(f"   Total servers: {stats['total_servers']}")
            print(f"   Healthy servers: {stats['healthy_servers']}")
            print(f"   Servers in fallback: {stats['servers_in_fallback']}")
            print(f"   Successful recoveries: {stats['successful_recoveries']}")
            print(f"   Failed recoveries: {stats['failed_recoveries']}")
            print(f"   Notifications sent: {stats['total_notifications_sent']}")
            
            if stats['degraded_features']:
                print(f"   Degraded features: {', '.join(stats['degraded_features'])}")
            
            # Show per-server details
            print("\n📋 Per-Server Details:")
            for server_name, server_stats in stats['servers'].items():
                status_icons = {
                    'healthy': '✅',
                    'degraded': '⚠️',
                    'unhealthy': '❌', 
                    'disconnected': '🔌',
                    'failed': '💥',
                    'unknown': '❓'
                }
                
                icon = status_icons.get(server_stats['status'], '❓')
                print(f"   {icon} {server_name}:")
                print(f"      Status: {server_stats['status']}")
                print(f"      Uptime: {server_stats['uptime_percentage']:.1f}%")
                print(f"      Avg Response: {server_stats['average_response_time_ms']:.1f}ms")
                print(f"      Total Checks: {server_stats['total_checks']}")
                print(f"      Failures: {server_stats['total_failures']}")
                
                if server_stats['in_fallback_mode']:
                    print(f"      🛡️ In fallback mode")
                
                if server_stats['supported_features']:
                    features = ', '.join(server_stats['supported_features'])
                    print(f"      Features: {features}")
            
            # Show feature availability
            available_features = health_monitor.get_available_features()
            print(f"\n🎯 Available LSP Features: {', '.join(available_features) if available_features else 'None (fallback mode)'}")
            
            # Show notification summary
            print(f"\n📬 Notification Summary ({len(notification_handler.notifications)} total):")
            level_counts = {}
            for notification in notification_handler.notifications:
                level = notification.level.value
                level_counts[level] = level_counts.get(level, 0) + 1
            
            for level, count in level_counts.items():
                level_icons = {
                    'info': 'ℹ️',
                    'warning': '⚠️',
                    'error': '❌',
                    'critical': '🚨'
                }
                icon = level_icons.get(level, '📄')
                print(f"   {icon} {level.title()}: {count}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    finally:
        print("\n🔚 Health monitoring demonstration complete")
        print("=" * 60)


async def demonstrate_integration_scenario():
    """Demonstrate real-world integration scenario"""
    print("\n🏗️ Integration Scenario: Development Workspace Setup")
    print("-" * 50)
    
    # Simulate setting up a development workspace with multiple LSP servers
    config = HealthCheckConfig(
        check_interval=2.0,
        enable_auto_recovery=True,
        enable_fallback_mode=True,
    )
    
    health_monitor = LspHealthMonitor(config=config)
    
    # Mock workspace directory
    workspace_dir = Path("/tmp/demo_workspace")
    
    # Simulate different language projects
    languages = {
        "python": {"files": ["main.py", "utils.py"], "lsp": "pylsp"},
        "rust": {"files": ["main.rs", "lib.rs"], "lsp": "rust-analyzer"},
        "typescript": {"files": ["index.ts", "app.ts"], "lsp": "typescript-language-server"},
        "go": {"files": ["main.go", "handler.go"], "lsp": "gopls"},
    }
    
    print("🗂️ Workspace Languages:")
    for lang, info in languages.items():
        print(f"   • {lang}: {len(info['files'])} files (LSP: {info['lsp']})")
    
    # Show how health monitor would integrate with file processing
    print(f"\n🔄 Integration Points:")
    print("   1. File ingestion prioritization based on LSP health")
    print("   2. Feature availability checks before LSP requests") 
    print("   3. Graceful degradation when servers are unavailable")
    print("   4. User notifications for development workflow interruptions")
    print("   5. Automatic recovery to restore full IDE functionality")
    
    # Demonstrate feature availability checks
    print(f"\n🎯 Feature Availability Examples:")
    features = ["hover", "definition", "references", "completion", "diagnostics"]
    
    for feature in features:
        # In real integration, this would check actual LSP server health
        available = True  # health_monitor.is_feature_available(feature)
        status = "✅ Available" if available else "❌ Unavailable (fallback mode)"
        print(f"   • {feature.title()}: {status}")
    
    print("\n✨ This demonstrates how the health monitor provides reliable")
    print("   LSP functionality with automatic recovery and graceful degradation")


if __name__ == "__main__":
    print("🔍 LSP Health Monitor Demonstration")
    print("This demo shows comprehensive LSP health monitoring with:")
    print("• Multi-server health tracking")
    print("• Automatic recovery mechanisms") 
    print("• User notifications with troubleshooting")
    print("• Graceful degradation and fallback modes")
    print("• Integration with development workflows")
    print()
    
    try:
        # Run main demonstration
        asyncio.run(demonstrate_health_monitoring())
        
        # Run integration scenario
        asyncio.run(demonstrate_integration_scenario())
        
    except KeyboardInterrupt:
        print("\n👋 Demo terminated by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()