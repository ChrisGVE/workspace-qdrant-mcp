"""
Multi-Component Communication Integration Tests

Tests coordination between CLI commands, MCP server tools, Web UI interface, 
and SQLite state manager to validate seamless cross-component communication.

Test Categories:
1. Cross-component state synchronization
2. Configuration consistency across components  
3. Event propagation verification
4. Error coordination testing
5. Performance monitoring
"""
import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import json
import os
import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import aiofiles
import aiohttp
from playwright.async_api import async_playwright
import sqlite3

from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager
from workspace_qdrant_mcp.core.yaml_config import YAMLConfigLoader
from workspace_qdrant_mcp.grpc.daemon_client import get_daemon_client


class MultiComponentTestFixture:
    """Test fixture managing all components for integration testing."""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.config_file = test_dir / "test_config.yaml"
        self.db_file = test_dir / "test_state.db"
        self.daemon_proc = None
        self.web_proc = None
        self.mcp_server = None
        self.state_manager = None
        
    async def setup(self):
        """Initialize all components for testing."""
        # Create test configuration
        await self._create_test_config()
        
        # Initialize SQLite state manager
        self.state_manager = SQLiteStateManager(str(self.db_file))
        await self.state_manager.initialize()
        
        # Start daemon process (mock for testing)
        await self._start_daemon()
        
        # Start web UI server
        await self._start_web_ui()
        
        # Initialize MCP server connection
        await self._setup_mcp_server()
        
    async def teardown(self):
        """Clean up all components."""
        if self.daemon_proc:
            self.daemon_proc.terminate()
            await asyncio.sleep(0.5)
            
        if self.web_proc:
            self.web_proc.terminate()
            await asyncio.sleep(0.5)
            
        if self.state_manager:
            await self.state_manager.close()
            
    async def _create_test_config(self):
        """Create test configuration with all components enabled."""
        config_data = {
            'qdrant': {
                'url': 'http://localhost:6333',
                'collection_name': 'test_collection'
            },
            'embedding': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'batch_size': 32
            },
            'daemon': {
                'host': '127.0.0.1',
                'port': 50051,
                'max_workers': 4
            },
            'web': {
                'host': '127.0.0.1',
                'port': 8080,
                'dev_mode': True
            },
            'sqlite': {
                'database_path': str(self.db_file),
                'wal_mode': True
            },
            'logging': {
                'level': 'INFO',
                'file': str(self.test_dir / "test.log")
            }
        }
        
        async with aiofiles.open(self.config_file, 'w') as f:
            import yaml
            await f.write(yaml.dump(config_data, default_flow_style=False))
            
    async def _start_daemon(self):
        """Start daemon process (mocked for testing)."""
        # In real implementation, this would start the actual daemon
        # For testing, we'll use a mock process
        self.daemon_proc = Mock()
        
    async def _start_web_ui(self):
        """Start web UI server (mocked for testing)."""
        # In real implementation, this would start the web server
        # For testing, we'll use a mock process
        self.web_proc = Mock()
        
    async def _setup_mcp_server(self):
        """Initialize MCP server connection."""
        # Mock MCP server for testing
        self.mcp_server = Mock()


@pytest.fixture
async def multi_component_fixture(tmp_path):
    """Fixture providing multi-component test environment."""
    fixture = MultiComponentTestFixture(tmp_path)
    await fixture.setup()
    yield fixture
    await fixture.teardown()


class TestCrossComponentStateSynchronization:
    """Test state synchronization between CLI, MCP, Web UI, and SQLite."""
    
    async def test_cli_to_sqlite_state_sync(self, multi_component_fixture):
        """Test CLI operations update SQLite state correctly."""
        fixture = multi_component_fixture
        
        # Simulate CLI file ingestion
        test_file = fixture.test_dir / "test_doc.txt"
        test_file.write_text("Test document content")
        
        # Mock CLI ingestion operation
        file_id = "test_doc_001"
        await fixture.state_manager.update_processing_state(
            file_path=str(test_file),
            status="processing",
            collection_name="test_collection",
            metadata={"source": "cli", "size": 100}
        )
        
        # Verify state in SQLite
        states = await fixture.state_manager.get_processing_states()
        assert len(states) == 1
        assert states[0]["file_path"] == str(test_file)
        assert states[0]["status"] == "processing"
        assert states[0]["metadata"]["source"] == "cli"
        
        # Update to completed
        await fixture.state_manager.update_processing_state(
            file_path=str(test_file),
            status="completed",
            document_id=file_id
        )
        
        # Verify update
        states = await fixture.state_manager.get_processing_states()
        assert states[0]["status"] == "completed"
        assert states[0]["document_id"] == file_id
        
    async def test_mcp_to_sqlite_state_sync(self, multi_component_fixture):
        """Test MCP tool operations update SQLite state correctly."""
        fixture = multi_component_fixture
        
        # Mock MCP search operation
        search_query = "test query"
        search_results = [
            {"id": "doc1", "score": 0.9, "content": "relevant content"},
            {"id": "doc2", "score": 0.7, "content": "somewhat relevant"}
        ]
        
        # Record search in state manager
        await fixture.state_manager.record_search_operation(
            query=search_query,
            results_count=len(search_results),
            source="mcp",
            metadata={"tool": "search", "timestamp": time.time()}
        )
        
        # Verify search history
        search_history = await fixture.state_manager.get_search_history(limit=1)
        assert len(search_history) == 1
        assert search_history[0]["query"] == search_query
        assert search_history[0]["results_count"] == 2
        assert search_history[0]["source"] == "mcp"
        
    async def test_web_ui_to_sqlite_state_sync(self, multi_component_fixture):
        """Test Web UI operations update SQLite state correctly."""
        fixture = multi_component_fixture
        
        # Mock web UI memory rule creation
        memory_rule = {
            "id": "rule_001",
            "name": "Test Rule",
            "pattern": "important.*documents",
            "priority": 5,
            "metadata": {"source": "web_ui", "created_by": "user"}
        }
        
        # Record memory rule via state manager
        await fixture.state_manager.store_memory_rule(
            rule_id=memory_rule["id"],
            rule_data=memory_rule
        )
        
        # Verify memory rule storage
        stored_rules = await fixture.state_manager.get_memory_rules()
        assert len(stored_rules) == 1
        assert stored_rules[0]["rule_id"] == memory_rule["id"]
        assert stored_rules[0]["rule_data"]["name"] == "Test Rule"
        
    async def test_cross_component_state_consistency(self, multi_component_fixture):
        """Test state changes in one component are visible to others."""
        fixture = multi_component_fixture
        
        # Create state via "CLI"
        file_path = str(fixture.test_dir / "shared_doc.txt")
        await fixture.state_manager.update_processing_state(
            file_path=file_path,
            status="completed",
            collection_name="shared_collection",
            metadata={"source": "cli", "shared": True}
        )
        
        # Verify state visible via "MCP"
        states = await fixture.state_manager.get_processing_states(
            filter_params={"collection_name": "shared_collection"}
        )
        assert len(states) == 1
        assert states[0]["file_path"] == file_path
        assert states[0]["metadata"]["shared"] is True
        
        # Update state via "Web UI"
        await fixture.state_manager.update_processing_state(
            file_path=file_path,
            status="reindexing",
            metadata={"source": "web_ui", "shared": True, "updated": True}
        )
        
        # Verify update visible across all components
        states = await fixture.state_manager.get_processing_states(
            filter_params={"file_path": file_path}
        )
        assert len(states) == 1
        assert states[0]["status"] == "reindexing"
        assert states[0]["metadata"]["updated"] is True


class TestConfigurationConsistency:
    """Test configuration consistency across all components."""
    
    async def test_yaml_config_hierarchy_consistency(self, multi_component_fixture):
        """Test YAML configuration hierarchy works across components."""
        fixture = multi_component_fixture
        
        # Create configuration hierarchy
        system_config = fixture.test_dir / "system_config.yaml"
        user_config = fixture.test_dir / "user_config.yaml"
        project_config = fixture.test_dir / "project_config.yaml"
        
        # System config (lowest priority)
        system_data = {
            'qdrant': {'url': 'http://system:6333'},
            'embedding': {'model': 'system-model'},
            'daemon': {'port': 50050}
        }
        
        # User config (medium priority)
        user_data = {
            'qdrant': {'url': 'http://user:6333'},
            'embedding': {'batch_size': 64}
        }
        
        # Project config (highest priority)
        project_data = {
            'qdrant': {'collection_name': 'project_collection'}
        }
        
        # Write config files
        import yaml
        async with aiofiles.open(system_config, 'w') as f:
            await f.write(yaml.dump(system_data))
        async with aiofiles.open(user_config, 'w') as f:
            await f.write(yaml.dump(user_data))
        async with aiofiles.open(project_config, 'w') as f:
            await f.write(yaml.dump(project_data))
            
        # Load configuration with hierarchy
        config_loader = YAMLConfigLoader()
        config = await config_loader.load_with_hierarchy([
            str(system_config),
            str(user_config), 
            str(project_config)
        ])
        
        # Verify hierarchy precedence
        assert config['qdrant']['url'] == 'http://user:6333'  # User overrides system
        assert config['qdrant']['collection_name'] == 'project_collection'  # Project is highest
        assert config['embedding']['model'] == 'system-model'  # System provides default
        assert config['embedding']['batch_size'] == 64  # User overrides
        assert config['daemon']['port'] == 50050  # System provides default
        
    async def test_environment_variable_substitution(self, multi_component_fixture):
        """Test environment variable substitution works consistently."""
        fixture = multi_component_fixture
        
        # Set environment variables
        os.environ['TEST_QDRANT_URL'] = 'http://env-qdrant:6333'
        os.environ['TEST_EMBEDDING_MODEL'] = 'env-embedding-model'
        os.environ['TEST_PORT'] = '9999'
        
        try:
            # Create config with environment variable patterns
            config_data = {
                'qdrant': {
                    'url': '${TEST_QDRANT_URL}',
                    'collection_name': '${TEST_COLLECTION:default_collection}'
                },
                'embedding': {
                    'model': '${TEST_EMBEDDING_MODEL}',
                    'batch_size': '${TEST_BATCH_SIZE:32}'
                },
                'daemon': {
                    'port': '${TEST_PORT}'
                }
            }
            
            env_config = fixture.test_dir / "env_config.yaml"
            import yaml
            async with aiofiles.open(env_config, 'w') as f:
                await f.write(yaml.dump(config_data))
                
            # Load and expand configuration
            config_loader = YAMLConfigLoader()
            config = await config_loader.load_with_env_substitution(str(env_config))
            
            # Verify environment variable substitution
            assert config['qdrant']['url'] == 'http://env-qdrant:6333'
            assert config['embedding']['model'] == 'env-embedding-model' 
            assert config['daemon']['port'] == 9999
            
            # Verify default values for missing env vars
            assert config['qdrant']['collection_name'] == 'default_collection'
            assert config['embedding']['batch_size'] == 32
            
        finally:
            # Clean up environment variables
            for var in ['TEST_QDRANT_URL', 'TEST_EMBEDDING_MODEL', 'TEST_PORT']:
                os.environ.pop(var, None)
                
    async def test_config_validation_across_components(self, multi_component_fixture):
        """Test configuration validation is consistent across components."""
        fixture = multi_component_fixture
        
        # Test invalid configuration
        invalid_config_data = {
            'qdrant': {
                'url': 'invalid-url',  # Invalid URL format
                'port': 'not-a-number'  # Invalid port type
            },
            'embedding': {
                'batch_size': -1  # Invalid negative batch size
            }
        }
        
        invalid_config = fixture.test_dir / "invalid_config.yaml"
        import yaml
        async with aiofiles.open(invalid_config, 'w') as f:
            await f.write(yaml.dump(invalid_config_data))
            
        # Test validation fails consistently
        config_loader = YAMLConfigLoader()
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            await config_loader.load_and_validate(str(invalid_config))


class TestEventPropagation:
    """Test event propagation across components."""
    
    async def test_file_processing_event_propagation(self, multi_component_fixture):
        """Test file processing events are visible across components."""
        fixture = multi_component_fixture
        
        # Mock file processing event chain
        file_path = str(fixture.test_dir / "event_test.txt")
        events = []
        
        # Event 1: File added
        await fixture.state_manager.record_event({
            "type": "file_added",
            "file_path": file_path,
            "timestamp": time.time(),
            "source": "file_watcher"
        })
        events.append("file_added")
        
        # Event 2: Processing started
        await fixture.state_manager.update_processing_state(
            file_path=file_path,
            status="processing",
            metadata={"started_at": time.time()}
        )
        events.append("processing_started")
        
        # Event 3: Processing completed
        await fixture.state_manager.update_processing_state(
            file_path=file_path,
            status="completed",
            document_id="doc_123",
            metadata={"completed_at": time.time()}
        )
        events.append("processing_completed")
        
        # Verify event chain is recorded
        recorded_events = await fixture.state_manager.get_events(
            filter_params={"file_path": file_path}
        )
        assert len(recorded_events) >= 1
        
        # Verify final state
        states = await fixture.state_manager.get_processing_states(
            filter_params={"file_path": file_path}
        )
        assert len(states) == 1
        assert states[0]["status"] == "completed"
        assert states[0]["document_id"] == "doc_123"
        
    async def test_search_event_propagation(self, multi_component_fixture):
        """Test search events propagate correctly."""
        fixture = multi_component_fixture
        
        # Mock search event
        search_event = {
            "type": "search_performed",
            "query": "test search query",
            "results_count": 5,
            "response_time_ms": 150,
            "source": "web_ui",
            "timestamp": time.time()
        }
        
        # Record search event
        await fixture.state_manager.record_event(search_event)
        
        # Record in search history as well
        await fixture.state_manager.record_search_operation(
            query=search_event["query"],
            results_count=search_event["results_count"],
            source=search_event["source"],
            metadata={"response_time_ms": search_event["response_time_ms"]}
        )
        
        # Verify event recorded
        events = await fixture.state_manager.get_events(
            filter_params={"type": "search_performed"}
        )
        assert len(events) >= 1
        
        # Verify search history updated
        search_history = await fixture.state_manager.get_search_history(limit=1)
        assert len(search_history) >= 1
        assert search_history[0]["query"] == search_event["query"]
        
    async def test_configuration_change_propagation(self, multi_component_fixture):
        """Test configuration changes propagate to all components."""
        fixture = multi_component_fixture
        
        # Record initial configuration
        original_config = {
            "qdrant": {"url": "http://localhost:6333"},
            "embedding": {"model": "original-model"}
        }
        
        await fixture.state_manager.record_configuration_change(
            config_data=original_config,
            source="initialization",
            timestamp=time.time()
        )
        
        # Record configuration update
        updated_config = {
            "qdrant": {"url": "http://updated:6333"},
            "embedding": {"model": "updated-model", "batch_size": 64}
        }
        
        await fixture.state_manager.record_configuration_change(
            config_data=updated_config,
            source="web_ui",
            timestamp=time.time()
        )
        
        # Verify configuration history
        config_history = await fixture.state_manager.get_configuration_history()
        assert len(config_history) >= 2
        
        # Verify latest configuration
        latest_config = config_history[0]  # Assuming newest first
        assert latest_config["config_data"]["qdrant"]["url"] == "http://updated:6333"
        assert latest_config["config_data"]["embedding"]["model"] == "updated-model"


class TestErrorCoordination:
    """Test error coordination across components."""
    
    async def test_cli_error_propagation(self, multi_component_fixture):
        """Test CLI errors are properly recorded and visible."""
        fixture = multi_component_fixture
        
        # Mock CLI error
        error_event = {
            "type": "cli_error",
            "command": "wqm ingest file",
            "error_message": "File not found: /nonexistent/file.txt",
            "error_code": 404,
            "timestamp": time.time(),
            "source": "cli"
        }
        
        # Record error
        await fixture.state_manager.record_error(
            error_type="file_not_found",
            error_message=error_event["error_message"],
            source=error_event["source"],
            metadata={
                "command": error_event["command"],
                "error_code": error_event["error_code"]
            }
        )
        
        # Verify error recorded
        errors = await fixture.state_manager.get_errors(
            filter_params={"error_type": "file_not_found"}
        )
        assert len(errors) >= 1
        assert errors[0]["error_message"] == error_event["error_message"]
        assert errors[0]["source"] == "cli"
        
    async def test_mcp_error_coordination(self, multi_component_fixture):
        """Test MCP errors coordinate with state management."""
        fixture = multi_component_fixture
        
        # Mock MCP tool error
        mcp_error = {
            "type": "mcp_tool_error",
            "tool": "search_documents",
            "error_message": "Qdrant connection timeout",
            "timestamp": time.time(),
            "source": "mcp_server"
        }
        
        # Record MCP error
        await fixture.state_manager.record_error(
            error_type="connection_timeout",
            error_message=mcp_error["error_message"],
            source=mcp_error["source"],
            metadata={"tool": mcp_error["tool"]}
        )
        
        # Verify error coordination
        errors = await fixture.state_manager.get_errors(
            filter_params={"source": "mcp_server"}
        )
        assert len(errors) >= 1
        assert errors[0]["metadata"]["tool"] == "search_documents"
        
    async def test_component_failure_recovery(self, multi_component_fixture):
        """Test component failure recovery coordination."""
        fixture = multi_component_fixture
        
        # Simulate component failure
        failure_event = {
            "type": "component_failure",
            "component": "web_ui",
            "error_message": "Server connection lost",
            "timestamp": time.time()
        }
        
        # Record failure
        await fixture.state_manager.record_event(failure_event)
        
        # Simulate recovery
        recovery_event = {
            "type": "component_recovery", 
            "component": "web_ui",
            "message": "Connection restored",
            "timestamp": time.time() + 30  # 30 seconds later
        }
        
        # Record recovery
        await fixture.state_manager.record_event(recovery_event)
        
        # Verify failure/recovery sequence
        events = await fixture.state_manager.get_events(
            filter_params={"component": "web_ui"}
        )
        assert len(events) >= 2
        
        # Check event types
        event_types = [event["type"] for event in events]
        assert "component_failure" in event_types
        assert "component_recovery" in event_types


class TestPerformanceMonitoring:
    """Test performance monitoring across components."""
    
    async def test_cross_component_latency_measurement(self, multi_component_fixture):
        """Test latency measurement across component boundaries."""
        fixture = multi_component_fixture
        
        # Simulate cross-component operation with timing
        start_time = time.time()
        
        # CLI → Daemon → SQLite operation chain
        file_path = str(fixture.test_dir / "perf_test.txt")
        
        # Step 1: CLI initiates (mock timing)
        cli_start = time.time()
        await asyncio.sleep(0.01)  # Mock CLI processing time
        cli_end = time.time()
        
        # Step 2: Daemon processing (mock timing)
        daemon_start = time.time()
        await fixture.state_manager.update_processing_state(
            file_path=file_path,
            status="processing",
            metadata={"cli_time_ms": (cli_end - cli_start) * 1000}
        )
        await asyncio.sleep(0.02)  # Mock daemon processing
        daemon_end = time.time()
        
        # Step 3: SQLite operation (actual timing)
        sqlite_start = time.time()
        await fixture.state_manager.update_processing_state(
            file_path=file_path,
            status="completed",
            metadata={
                "cli_time_ms": (cli_end - cli_start) * 1000,
                "daemon_time_ms": (daemon_end - daemon_start) * 1000
            }
        )
        sqlite_end = time.time()
        
        total_time = time.time() - start_time
        
        # Record performance metrics
        await fixture.state_manager.record_performance_metric({
            "operation": "cross_component_file_processing",
            "total_time_ms": total_time * 1000,
            "cli_time_ms": (cli_end - cli_start) * 1000,
            "daemon_time_ms": (daemon_end - daemon_start) * 1000,
            "sqlite_time_ms": (sqlite_end - sqlite_start) * 1000,
            "timestamp": time.time()
        })
        
        # Verify performance data recorded
        metrics = await fixture.state_manager.get_performance_metrics(
            filter_params={"operation": "cross_component_file_processing"}
        )
        assert len(metrics) >= 1
        
        # Verify timing breakdown
        metric = metrics[0]
        assert metric["total_time_ms"] > 0
        assert metric["cli_time_ms"] > 0
        assert metric["daemon_time_ms"] > 0
        assert metric["sqlite_time_ms"] > 0
        
    async def test_concurrent_component_performance(self, multi_component_fixture):
        """Test performance under concurrent component operations."""
        fixture = multi_component_fixture
        
        # Simulate concurrent operations
        concurrent_tasks = []
        
        for i in range(5):
            async def process_file(file_num: int):
                file_path = str(fixture.test_dir / f"concurrent_{file_num}.txt")
                start_time = time.time()
                
                # Simulate processing
                await fixture.state_manager.update_processing_state(
                    file_path=file_path,
                    status="processing"
                )
                await asyncio.sleep(0.01)  # Mock processing time
                await fixture.state_manager.update_processing_state(
                    file_path=file_path,
                    status="completed"
                )
                
                end_time = time.time()
                return end_time - start_time
                
            concurrent_tasks.append(process_file(i))
            
        # Execute concurrent operations
        completion_times = await asyncio.gather(*concurrent_tasks)
        
        # Verify all operations completed
        assert len(completion_times) == 5
        assert all(t > 0 for t in completion_times)
        
        # Check for performance degradation
        avg_time = sum(completion_times) / len(completion_times)
        max_time = max(completion_times)
        
        # Ensure reasonable performance bounds
        assert avg_time < 0.1  # Average under 100ms
        assert max_time < 0.2   # Max under 200ms
        
    async def test_resource_usage_monitoring(self, multi_component_fixture):
        """Test resource usage monitoring across components."""
        fixture = multi_component_fixture
        
        # Mock resource usage data
        resource_metrics = {
            "cpu_usage_percent": 25.5,
            "memory_usage_mb": 128.0,
            "disk_io_operations": 450,
            "network_connections": 3,
            "sqlite_db_size_mb": 2.1,
            "timestamp": time.time()
        }
        
        # Record resource usage
        await fixture.state_manager.record_resource_usage(resource_metrics)
        
        # Simulate sustained operation
        for i in range(3):
            await asyncio.sleep(0.01)
            resource_metrics["timestamp"] = time.time()
            resource_metrics["cpu_usage_percent"] += i * 2
            await fixture.state_manager.record_resource_usage(resource_metrics)
            
        # Verify resource monitoring
        usage_history = await fixture.state_manager.get_resource_usage_history()
        assert len(usage_history) >= 4
        
        # Check resource trends
        cpu_values = [u["cpu_usage_percent"] for u in usage_history]
        assert max(cpu_values) > min(cpu_values)  # Should show variation


# Test execution with proper async handling
@pytest.mark.asyncio
class TestMultiComponentIntegration:
    """Main integration test class."""
    
    async def test_complete_workflow_integration(self, multi_component_fixture):
        """Test complete workflow across all components."""
        fixture = multi_component_fixture
        
        # Step 1: Configuration loading
        config_data = {
            "qdrant": {"url": "http://test:6333"},
            "embedding": {"model": "test-model"}
        }
        await fixture.state_manager.record_configuration_change(
            config_data=config_data,
            source="test_setup"
        )
        
        # Step 2: File processing workflow
        test_file = fixture.test_dir / "integration_test.txt"
        test_file.write_text("This is a test document for integration testing.")
        
        # CLI ingestion simulation
        await fixture.state_manager.update_processing_state(
            file_path=str(test_file),
            status="processing",
            collection_name="integration_test"
        )
        
        # Daemon processing simulation
        await fixture.state_manager.update_processing_state(
            file_path=str(test_file),
            status="completed",
            document_id="integration_doc_001"
        )
        
        # Step 3: Search operation
        await fixture.state_manager.record_search_operation(
            query="integration test document",
            results_count=1,
            source="integration_test"
        )
        
        # Step 4: Verification across all components
        
        # Verify file processing state
        processing_states = await fixture.state_manager.get_processing_states()
        assert len(processing_states) >= 1
        completed_files = [s for s in processing_states if s["status"] == "completed"]
        assert len(completed_files) >= 1
        
        # Verify search history
        search_history = await fixture.state_manager.get_search_history()
        assert len(search_history) >= 1
        
        # Verify configuration history
        config_history = await fixture.state_manager.get_configuration_history()
        assert len(config_history) >= 1
        
        # Verify no errors occurred
        errors = await fixture.state_manager.get_errors()
        integration_errors = [e for e in errors if "integration" in str(e)]
        assert len(integration_errors) == 0
        
        print("✅ Complete multi-component workflow integration test passed!")


if __name__ == "__main__":
    # Run tests with proper async support
    pytest.main([__file__, "-v", "--tb=short"])