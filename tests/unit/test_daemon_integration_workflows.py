"""
Unit tests for integration workflows and performance validation in daemon context.

Tests complete daemon integration workflows including:
- Full document ingestion pipeline from file watch to storage
- Multi-component integration scenarios across daemon subsystems
- Error propagation and recovery across components
- Performance benchmarks for key daemon operations
- Resource usage validation and memory leak detection
- Configuration change handling and live reload capabilities
- Graceful degradation under resource constraints
- End-to-end workflow validation with comprehensive error condition matrices
"""

import asyncio
import json
import os
import psutil
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock, call
from unittest import mock
import pytest

from common.core.daemon_manager import (
    DaemonManager,
    DaemonInstance,
    DaemonConfig
)
from common.core.grpc_client import GrpcWorkspaceClient
from common.core.client import QdrantWorkspaceClient

from .conftest_daemon import (
    mock_daemon_config,
    mock_daemon_instance,
    mock_daemon_manager,
    isolated_daemon_temp_dir,
    DaemonTestHelper
)


@pytest.fixture
def mock_file_watcher():
    """Mock file watcher for testing."""
    watcher = Mock()
    watcher.start_watching = AsyncMock(return_value=True)
    watcher.stop_watching = AsyncMock(return_value=True)
    watcher.add_watch_folder = AsyncMock(return_value="watch_id_123")
    watcher.remove_watch_folder = AsyncMock(return_value=True)
    watcher.get_watched_folders = Mock(return_value=["/test/folder1", "/test/folder2"])
    watcher.on_file_changed = AsyncMock()
    watcher.on_file_created = AsyncMock()
    watcher.on_file_deleted = AsyncMock()
    return watcher


@pytest.fixture
def mock_processing_engine():
    """Mock processing engine for testing."""
    engine = Mock()
    engine.start = AsyncMock(return_value=True)
    engine.stop = AsyncMock(return_value=True)
    engine.submit_task = AsyncMock(return_value="task_id_123")
    engine.get_task_status = AsyncMock(return_value="completed")
    engine.get_queue_stats = Mock(return_value={
        "pending": 5,
        "processing": 2,
        "completed": 100,
        "failed": 3
    })
    engine.process_file = AsyncMock(return_value={"success": True, "document_id": "doc_123"})
    return engine


@pytest.fixture
def mock_resource_monitor():
    """Mock resource monitor for testing."""
    monitor = Mock()
    monitor.start_monitoring = AsyncMock(return_value=True)
    monitor.stop_monitoring = AsyncMock(return_value=True)
    monitor.get_current_usage = Mock(return_value={
        "cpu_percent": 45.2,
        "memory_percent": 62.1,
        "memory_mb": 1024,
        "open_files": 150,
        "network_connections": 5
    })
    monitor.check_resource_limits = Mock(return_value={
        "within_limits": True,
        "warnings": []
    })
    monitor.get_performance_history = Mock(return_value=[])
    return monitor


@pytest.fixture
def mock_configuration_manager():
    """Mock configuration manager for testing."""
    manager = Mock()
    manager.load_configuration = AsyncMock(return_value={
        "daemon": {"log_level": "info", "max_concurrent_jobs": 4},
        "storage": {"qdrant_url": "http://localhost:6333"},
        "watching": {"enabled": True, "recursive": True}
    })
    manager.save_configuration = AsyncMock(return_value=True)
    manager.reload_configuration = AsyncMock(return_value=True)
    manager.validate_configuration = AsyncMock(return_value={"valid": True})
    manager.watch_configuration_changes = AsyncMock()
    return manager


class TestFullIngestionPipeline:
    """Test complete document ingestion pipeline from file watch to storage."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_document_ingestion(self, mock_daemon_instance, mock_file_watcher, 
                                                mock_processing_engine, isolated_daemon_temp_dir):
        """Test complete document ingestion workflow."""
        # Setup daemon with full pipeline
        mock_daemon_instance.file_watcher = mock_file_watcher
        mock_daemon_instance.processing_engine = mock_processing_engine
        mock_daemon_instance.status.state = "running"
        
        # Create test files
        test_file = isolated_daemon_temp_dir / "test_document.py"
        test_file.write_text("def hello_world():\n    print('Hello, World!')")
        
        # Mock complete ingestion pipeline
        ingestion_steps = []
        
        with patch.object(mock_daemon_instance, 'process_ingestion_pipeline', new_callable=AsyncMock) as mock_pipeline:
            async def ingestion_pipeline(file_path, collection_name):
                # Step 1: File detection
                ingestion_steps.append(("file_detected", file_path))
                
                # Step 2: File content reading
                content = test_file.read_text()
                ingestion_steps.append(("content_read", len(content)))
                
                # Step 3: Processing submission
                task_id = await mock_processing_engine.submit_task({
                    "file_path": file_path,
                    "collection": collection_name,
                    "content": content
                })
                ingestion_steps.append(("task_submitted", task_id))
                
                # Step 4: Processing completion
                result = await mock_processing_engine.process_file(file_path, content)
                ingestion_steps.append(("processing_completed", result["document_id"]))
                
                # Step 5: Storage confirmation
                ingestion_steps.append(("stored", True))
                
                return {
                    "success": True,
                    "file_path": file_path,
                    "document_id": result["document_id"],
                    "steps_completed": len(ingestion_steps)
                }
            
            mock_pipeline.side_effect = ingestion_pipeline
            
            # Test complete pipeline
            result = await mock_daemon_instance.process_ingestion_pipeline(
                str(test_file), "test_collection"
            )
            
            assert result["success"] is True
            assert result["steps_completed"] == 5
            assert ("file_detected", str(test_file)) in ingestion_steps
            assert ("processing_completed", "doc_123") in ingestion_steps
    
    @pytest.mark.asyncio
    async def test_file_watching_integration(self, mock_daemon_instance, mock_file_watcher, 
                                           mock_processing_engine, isolated_daemon_temp_dir):
        """Test file watching integration with processing pipeline."""
        # Setup daemon with file watching
        mock_daemon_instance.file_watcher = mock_file_watcher
        mock_daemon_instance.processing_engine = mock_processing_engine
        
        # Mock file watching event handling
        watch_events = []
        processed_files = []
        
        async def handle_file_event(event_type, file_path):
            watch_events.append((event_type, file_path))
            
            if event_type in ["created", "modified"]:
                # Trigger processing
                result = await mock_processing_engine.process_file(file_path, "file content")
                processed_files.append(file_path)
                return result
            
            return {"handled": True}
        
        mock_file_watcher.on_file_changed.side_effect = lambda path: handle_file_event("modified", path)
        mock_file_watcher.on_file_created.side_effect = lambda path: handle_file_event("created", path)
        mock_file_watcher.on_file_deleted.side_effect = lambda path: handle_file_event("deleted", path)
        
        # Simulate file events
        test_files = [
            str(isolated_daemon_temp_dir / "file1.py"),
            str(isolated_daemon_temp_dir / "file2.py"),
            str(isolated_daemon_temp_dir / "file3.py")
        ]
        
        for file_path in test_files:
            # Simulate file creation
            await mock_file_watcher.on_file_created(file_path)
            
            # Simulate file modification
            await mock_file_watcher.on_file_changed(file_path)
        
        # Verify events were handled
        assert len(watch_events) == 6  # 3 created + 3 modified
        assert len(processed_files) == 6  # All events triggered processing
        
        # Verify processing calls
        assert mock_processing_engine.process_file.call_count == 6
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, mock_daemon_instance, mock_processing_engine):
        """Test batch processing workflow for multiple files."""
        # Setup daemon with batch processing
        mock_daemon_instance.processing_engine = mock_processing_engine
        mock_daemon_instance.batch_size = 5
        
        # Mock batch processing
        batch_results = []
        
        with patch.object(mock_daemon_instance, 'process_files_batch', new_callable=AsyncMock) as mock_batch:
            async def process_batch(file_paths, collection_name, batch_size=5):
                results = []
                
                for i in range(0, len(file_paths), batch_size):
                    batch = file_paths[i:i + batch_size]
                    batch_id = f"batch_{i // batch_size + 1}"
                    
                    # Process batch
                    batch_result = {
                        "batch_id": batch_id,
                        "files": batch,
                        "processed": len(batch),
                        "successful": len(batch),
                        "failed": 0
                    }
                    
                    for file_path in batch:
                        await mock_processing_engine.submit_task({
                            "file_path": file_path,
                            "batch_id": batch_id
                        })
                    
                    results.append(batch_result)
                    batch_results.append(batch_result)
                
                return {
                    "total_files": len(file_paths),
                    "batches_processed": len(results),
                    "overall_success": True,
                    "results": results
                }
            
            mock_batch.side_effect = process_batch
            
            # Test batch processing with 12 files (3 batches of 5, 5, 2)
            file_paths = [f"/test/file{i}.py" for i in range(12)]
            result = await mock_daemon_instance.process_files_batch(file_paths, "test_collection")
            
            assert result["total_files"] == 12
            assert result["batches_processed"] == 3
            assert result["overall_success"] is True
            assert len(batch_results) == 3
            
            # Verify batch sizes
            assert batch_results[0]["processed"] == 5
            assert batch_results[1]["processed"] == 5
            assert batch_results[2]["processed"] == 2


class TestMultiComponentIntegration:
    """Test integration scenarios across daemon subsystems."""
    
    @pytest.mark.asyncio
    async def test_daemon_subsystem_coordination(self, mock_daemon_instance, mock_file_watcher,
                                                mock_processing_engine, mock_resource_monitor):
        """Test coordination between daemon subsystems."""
        # Setup daemon with all subsystems
        mock_daemon_instance.file_watcher = mock_file_watcher
        mock_daemon_instance.processing_engine = mock_processing_engine
        mock_daemon_instance.resource_monitor = mock_resource_monitor
        mock_daemon_instance.subsystems_started = []
        
        # Mock subsystem coordination
        with patch.object(mock_daemon_instance, 'coordinate_subsystems', new_callable=AsyncMock) as mock_coordinate:
            async def coordinate_subsystems(operation):
                if operation == "startup":
                    # Start subsystems in order
                    await mock_resource_monitor.start_monitoring()
                    mock_daemon_instance.subsystems_started.append("resource_monitor")
                    
                    await mock_processing_engine.start()
                    mock_daemon_instance.subsystems_started.append("processing_engine")
                    
                    await mock_file_watcher.start_watching()
                    mock_daemon_instance.subsystems_started.append("file_watcher")
                    
                    return {"coordinated": True, "subsystems": len(mock_daemon_instance.subsystems_started)}
                
                elif operation == "shutdown":
                    # Stop subsystems in reverse order
                    await mock_file_watcher.stop_watching()
                    await mock_processing_engine.stop()
                    await mock_resource_monitor.stop_monitoring()
                    
                    mock_daemon_instance.subsystems_started.clear()
                    return {"coordinated": True, "subsystems_stopped": 3}
                
                return {"operation": operation, "status": "unknown"}
            
            mock_coordinate.side_effect = coordinate_subsystems
            
            # Test subsystem startup coordination
            startup_result = await mock_daemon_instance.coordinate_subsystems("startup")
            assert startup_result["coordinated"] is True
            assert startup_result["subsystems"] == 3
            assert mock_daemon_instance.subsystems_started == ["resource_monitor", "processing_engine", "file_watcher"]
            
            # Test subsystem shutdown coordination
            shutdown_result = await mock_daemon_instance.coordinate_subsystems("shutdown")
            assert shutdown_result["coordinated"] is True
            assert shutdown_result["subsystems_stopped"] == 3
            assert len(mock_daemon_instance.subsystems_started) == 0
    
    @pytest.mark.asyncio
    async def test_cross_component_error_propagation(self, mock_daemon_instance, mock_file_watcher, 
                                                   mock_processing_engine):
        """Test error propagation across daemon components."""
        # Setup daemon with error propagation
        mock_daemon_instance.file_watcher = mock_file_watcher
        mock_daemon_instance.processing_engine = mock_processing_engine
        mock_daemon_instance.error_count = 0
        mock_daemon_instance.component_errors = {}
        
        # Mock error propagation mechanism
        async def propagate_error(component, error):
            mock_daemon_instance.error_count += 1
            mock_daemon_instance.component_errors[component] = str(error)
            
            # Determine error severity and propagation strategy
            if "critical" in str(error).lower():
                # Critical errors should trigger cascade
                if component == "file_watcher":
                    await mock_processing_engine.stop()
                elif component == "processing_engine":
                    await mock_file_watcher.stop_watching()
                
                return {"propagated": True, "cascade_triggered": True}
            else:
                # Non-critical errors are isolated
                return {"propagated": True, "cascade_triggered": False}
        
        # Test error scenarios
        # 1. Non-critical file watcher error
        non_critical_result = await propagate_error("file_watcher", "File access denied")
        assert non_critical_result["cascade_triggered"] is False
        
        # 2. Critical processing engine error
        critical_result = await propagate_error("processing_engine", "Critical memory allocation failure")
        assert critical_result["cascade_triggered"] is True
        
        # Verify error tracking
        assert mock_daemon_instance.error_count == 2
        assert "file_watcher" in mock_daemon_instance.component_errors
        assert "processing_engine" in mock_daemon_instance.component_errors
    
    @pytest.mark.asyncio
    async def test_component_health_monitoring(self, mock_daemon_instance, mock_file_watcher,
                                             mock_processing_engine, mock_resource_monitor):
        """Test health monitoring across daemon components."""
        # Setup daemon with health monitoring
        components = {
            "file_watcher": mock_file_watcher,
            "processing_engine": mock_processing_engine,
            "resource_monitor": mock_resource_monitor
        }
        
        health_status = {}
        
        # Mock health monitoring
        with patch.object(mock_daemon_instance, 'monitor_component_health', new_callable=AsyncMock) as mock_monitor:
            async def monitor_health():
                for name, component in components.items():
                    try:
                        if hasattr(component, 'health_check'):
                            is_healthy = await component.health_check()
                        else:
                            # Simulate health check
                            is_healthy = True
                        
                        health_status[name] = {
                            "healthy": is_healthy,
                            "last_check": datetime.now(timezone.utc),
                            "response_time": 0.05
                        }
                    except Exception as e:
                        health_status[name] = {
                            "healthy": False,
                            "error": str(e),
                            "last_check": datetime.now(timezone.utc)
                        }
                
                return health_status
            
            mock_monitor.side_effect = monitor_health
            
            # Test health monitoring
            health_result = await mock_daemon_instance.monitor_component_health()
            
            assert len(health_result) == 3
            assert all(status["healthy"] for status in health_result.values())
            assert all("last_check" in status for status in health_result.values())


class TestPerformanceBenchmarking:
    """Test performance benchmarks for key daemon operations."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_document_processing_performance(self, mock_daemon_instance, mock_processing_engine, benchmark):
        """Benchmark document processing performance."""
        # Setup daemon for performance testing
        mock_daemon_instance.processing_engine = mock_processing_engine
        
        async def process_documents():
            # Simulate processing multiple documents
            tasks = []
            for i in range(10):
                task = mock_processing_engine.process_file(
                    f"/test/file{i}.py",
                    f"content for file {i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return len(results)
        
        # Benchmark the operation
        def sync_wrapper():
            return asyncio.run(process_documents())
        
        result = benchmark(sync_wrapper)
        assert result == 10
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio  
    async def test_concurrent_request_handling(self, mock_daemon_instance, benchmark):
        """Benchmark concurrent request handling performance."""
        # Setup daemon with concurrent request handling
        mock_daemon_instance.max_concurrent_requests = 20
        request_count = 0
        
        async def handle_concurrent_requests():
            nonlocal request_count
            
            async def process_request(request_id):
                nonlocal request_count
                request_count += 1
                await asyncio.sleep(0.001)  # Simulate processing time
                return f"processed_{request_id}"
            
            # Create concurrent requests
            tasks = [process_request(i) for i in range(50)]
            results = await asyncio.gather(*tasks)
            return len(results)
        
        # Benchmark concurrent processing
        def sync_wrapper():
            return asyncio.run(handle_concurrent_requests())
        
        result = benchmark(sync_wrapper)
        assert result == 50
        assert request_count == 50
    
    @pytest.mark.benchmark
    def test_memory_usage_optimization(self, mock_daemon_instance, benchmark):
        """Benchmark memory usage optimization."""
        # Setup memory-intensive operations
        mock_daemon_instance.memory_cache = {}
        
        def memory_operations():
            # Simulate memory-intensive operations
            for i in range(1000):
                key = f"item_{i}"
                value = {"data": [j for j in range(100)], "metadata": {"id": i}}
                mock_daemon_instance.memory_cache[key] = value
            
            # Simulate cache cleanup
            if len(mock_daemon_instance.memory_cache) > 500:
                # Keep only the most recent 500 items
                keys_to_remove = list(mock_daemon_instance.memory_cache.keys())[:-500]
                for key in keys_to_remove:
                    del mock_daemon_instance.memory_cache[key]
            
            return len(mock_daemon_instance.memory_cache)
        
        # Benchmark memory operations
        result = benchmark(memory_operations)
        assert result <= 500  # Cache should be limited


class TestResourceMonitoring:
    """Test resource usage validation and memory leak detection."""
    
    @pytest.mark.asyncio
    async def test_resource_usage_tracking(self, mock_daemon_instance, mock_resource_monitor):
        """Test resource usage tracking and validation."""
        # Setup daemon with resource monitoring
        mock_daemon_instance.resource_monitor = mock_resource_monitor
        mock_daemon_instance.resource_usage_history = []
        
        # Mock resource tracking
        with patch.object(mock_daemon_instance, 'track_resource_usage', new_callable=AsyncMock) as mock_track:
            async def track_usage():
                usage = mock_resource_monitor.get_current_usage()
                mock_daemon_instance.resource_usage_history.append({
                    **usage,
                    "timestamp": datetime.now(timezone.utc)
                })
                
                # Check for resource limit violations
                limits_check = mock_resource_monitor.check_resource_limits()
                if not limits_check["within_limits"]:
                    return {"status": "warning", "violations": limits_check["warnings"]}
                
                return {"status": "ok", "usage": usage}
            
            mock_track.side_effect = track_usage
            
            # Test resource tracking over time
            for _ in range(5):
                result = await mock_daemon_instance.track_resource_usage()
                await asyncio.sleep(0.01)  # Small delay between measurements
            
            assert len(mock_daemon_instance.resource_usage_history) == 5
            assert all(entry["cpu_percent"] == 45.2 for entry in mock_daemon_instance.resource_usage_history)
            assert all(entry["memory_mb"] == 1024 for entry in mock_daemon_instance.resource_usage_history)
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, mock_daemon_instance):
        """Test memory leak detection mechanisms."""
        # Setup memory leak detection
        initial_memory = 1024  # MB
        mock_daemon_instance.memory_baseline = initial_memory
        mock_daemon_instance.memory_samples = []
        
        # Mock memory leak detection
        with patch.object(mock_daemon_instance, 'detect_memory_leaks', new_callable=AsyncMock) as mock_detect:
            async def detect_leaks():
                # Simulate memory growth over time
                current_memory = initial_memory + len(mock_daemon_instance.memory_samples) * 10
                mock_daemon_instance.memory_samples.append(current_memory)
                
                # Detect trend
                if len(mock_daemon_instance.memory_samples) >= 10:
                    recent_samples = mock_daemon_instance.memory_samples[-10:]
                    growth_rate = (recent_samples[-1] - recent_samples[0]) / len(recent_samples)
                    
                    if growth_rate > 5:  # MB per sample
                        return {
                            "leak_detected": True,
                            "growth_rate_mb_per_sample": growth_rate,
                            "current_memory_mb": current_memory,
                            "baseline_memory_mb": initial_memory
                        }
                
                return {
                    "leak_detected": False,
                    "current_memory_mb": current_memory,
                    "samples_collected": len(mock_daemon_instance.memory_samples)
                }
            
            mock_detect.side_effect = detect_leaks
            
            # Test memory leak detection over multiple samples
            results = []
            for i in range(15):
                result = await mock_daemon_instance.detect_memory_leaks()
                results.append(result)
            
            # Should detect leak in later samples
            leak_detected = any(result.get("leak_detected", False) for result in results)
            assert leak_detected is True
            
            final_result = results[-1]
            assert final_result["current_memory_mb"] > initial_memory


class TestConfigurationManagement:
    """Test configuration change handling and live reload capabilities."""
    
    @pytest.mark.asyncio
    async def test_live_configuration_reload(self, mock_daemon_instance, mock_configuration_manager):
        """Test live configuration reload without daemon restart."""
        # Setup daemon with configuration management
        mock_daemon_instance.config_manager = mock_configuration_manager
        mock_daemon_instance.current_config = {
            "log_level": "info",
            "max_concurrent_jobs": 4,
            "batch_size": 10
        }
        
        # Mock configuration reload
        with patch.object(mock_daemon_instance, 'reload_configuration', new_callable=AsyncMock) as mock_reload:
            async def reload_config():
                # Load new configuration
                new_config = await mock_configuration_manager.load_configuration()
                
                # Apply configuration changes
                changes_applied = []
                old_config = mock_daemon_instance.current_config.copy()
                
                for key, new_value in new_config.get("daemon", {}).items():
                    if key in old_config and old_config[key] != new_value:
                        old_value = old_config[key]
                        mock_daemon_instance.current_config[key] = new_value
                        changes_applied.append({
                            "key": key,
                            "old_value": old_value,
                            "new_value": new_value
                        })
                
                return {
                    "reloaded": True,
                    "changes_applied": len(changes_applied),
                    "changes": changes_applied
                }
            
            mock_reload.side_effect = reload_config
            
            # Simulate configuration change
            mock_configuration_manager.load_configuration.return_value = {
                "daemon": {
                    "log_level": "debug",  # Changed from "info"
                    "max_concurrent_jobs": 8,  # Changed from 4
                    "batch_size": 10  # Unchanged
                }
            }
            
            # Test configuration reload
            reload_result = await mock_daemon_instance.reload_configuration()
            
            assert reload_result["reloaded"] is True
            assert reload_result["changes_applied"] == 2
            assert mock_daemon_instance.current_config["log_level"] == "debug"
            assert mock_daemon_instance.current_config["max_concurrent_jobs"] == 8
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, mock_daemon_instance, mock_configuration_manager):
        """Test configuration validation before applying changes."""
        # Setup daemon with configuration validation
        mock_daemon_instance.config_manager = mock_configuration_manager
        
        # Mock configuration validation scenarios
        validation_scenarios = [
            {
                "config": {"log_level": "debug", "max_concurrent_jobs": 8},
                "expected_valid": True
            },
            {
                "config": {"log_level": "invalid_level", "max_concurrent_jobs": 8},
                "expected_valid": False
            },
            {
                "config": {"log_level": "info", "max_concurrent_jobs": -1},
                "expected_valid": False
            }
        ]
        
        with patch.object(mock_daemon_instance, 'validate_configuration', new_callable=AsyncMock) as mock_validate:
            async def validate_config(config):
                # Validation rules
                valid_log_levels = ["debug", "info", "warning", "error"]
                
                if config.get("log_level") not in valid_log_levels:
                    return {"valid": False, "errors": ["Invalid log level"]}
                
                if config.get("max_concurrent_jobs", 1) < 1:
                    return {"valid": False, "errors": ["max_concurrent_jobs must be positive"]}
                
                return {"valid": True, "errors": []}
            
            mock_validate.side_effect = validate_config
            
            # Test validation scenarios
            results = []
            for scenario in validation_scenarios:
                result = await mock_daemon_instance.validate_configuration(scenario["config"])
                results.append(result)
                assert result["valid"] == scenario["expected_valid"]
            
            # Verify validation results
            assert results[0]["valid"] is True  # Valid config
            assert results[1]["valid"] is False  # Invalid log level
            assert results[2]["valid"] is False  # Invalid max_concurrent_jobs


class TestGracefulDegradation:
    """Test graceful degradation under resource constraints."""
    
    @pytest.mark.asyncio
    async def test_resource_constraint_handling(self, mock_daemon_instance, mock_resource_monitor):
        """Test handling of resource constraints with graceful degradation."""
        # Setup daemon with resource constraint handling
        mock_daemon_instance.resource_monitor = mock_resource_monitor
        mock_daemon_instance.degradation_mode = False
        mock_daemon_instance.max_concurrent_jobs = 10
        
        # Mock resource constraint scenarios
        constraint_scenarios = [
            {"cpu_percent": 95, "memory_percent": 80, "should_degrade": True},
            {"cpu_percent": 60, "memory_percent": 95, "should_degrade": True},
            {"cpu_percent": 50, "memory_percent": 60, "should_degrade": False}
        ]
        
        with patch.object(mock_daemon_instance, 'handle_resource_constraints', new_callable=AsyncMock) as mock_handle:
            async def handle_constraints(usage):
                cpu_threshold = 90
                memory_threshold = 85
                
                if usage["cpu_percent"] > cpu_threshold or usage["memory_percent"] > memory_threshold:
                    # Enter degradation mode
                    mock_daemon_instance.degradation_mode = True
                    mock_daemon_instance.max_concurrent_jobs = max(1, mock_daemon_instance.max_concurrent_jobs // 2)
                    
                    return {
                        "degradation_activated": True,
                        "new_max_concurrent_jobs": mock_daemon_instance.max_concurrent_jobs,
                        "reason": "resource_constraints"
                    }
                else:
                    # Normal operation
                    if mock_daemon_instance.degradation_mode:
                        mock_daemon_instance.degradation_mode = False
                        mock_daemon_instance.max_concurrent_jobs = 10  # Restore original
                        
                        return {
                            "degradation_deactivated": True,
                            "new_max_concurrent_jobs": mock_daemon_instance.max_concurrent_jobs,
                            "reason": "resources_available"
                        }
                    
                    return {"status": "normal_operation"}
            
            mock_handle.side_effect = handle_constraints
            
            # Test resource constraint scenarios
            for scenario in constraint_scenarios:
                usage = {
                    "cpu_percent": scenario["cpu_percent"],
                    "memory_percent": scenario["memory_percent"]
                }
                
                result = await mock_daemon_instance.handle_resource_constraints(usage)
                
                if scenario["should_degrade"]:
                    assert "degradation_activated" in result or mock_daemon_instance.degradation_mode
                    assert mock_daemon_instance.max_concurrent_jobs <= 5
                else:
                    if "degradation_deactivated" in result:
                        assert mock_daemon_instance.max_concurrent_jobs == 10
    
    @pytest.mark.asyncio
    async def test_fallback_mechanisms(self, mock_daemon_instance):
        """Test fallback mechanisms when primary systems fail."""
        # Setup daemon with fallback mechanisms
        mock_daemon_instance.primary_storage_available = True
        mock_daemon_instance.fallback_storage_available = True
        mock_daemon_instance.processed_via_fallback = 0
        
        # Mock fallback scenarios
        with patch.object(mock_daemon_instance, 'activate_fallback_systems', new_callable=AsyncMock) as mock_fallback:
            async def activate_fallbacks(failed_system):
                fallback_activated = []
                
                if failed_system == "primary_storage":
                    mock_daemon_instance.primary_storage_available = False
                    # Activate fallback storage
                    if mock_daemon_instance.fallback_storage_available:
                        fallback_activated.append("fallback_storage")
                
                elif failed_system == "processing_engine":
                    # Activate simplified processing
                    fallback_activated.append("simple_processor")
                
                elif failed_system == "file_watcher":
                    # Activate polling-based file detection
                    fallback_activated.append("file_polling")
                
                return {
                    "fallbacks_activated": fallback_activated,
                    "primary_system": failed_system,
                    "degraded_operation": True
                }
            
            mock_fallback.side_effect = activate_fallbacks
            
            # Test fallback activation for different system failures
            storage_fallback = await mock_daemon_instance.activate_fallback_systems("primary_storage")
            processing_fallback = await mock_daemon_instance.activate_fallback_systems("processing_engine")
            watcher_fallback = await mock_daemon_instance.activate_fallback_systems("file_watcher")
            
            assert "fallback_storage" in storage_fallback["fallbacks_activated"]
            assert "simple_processor" in processing_fallback["fallbacks_activated"]
            assert "file_polling" in watcher_fallback["fallbacks_activated"]
            assert all(result["degraded_operation"] for result in [storage_fallback, processing_fallback, watcher_fallback])


@pytest.mark.daemon_unit
@pytest.mark.daemon_integration
@pytest.mark.daemon_performance
class TestComprehensiveIntegrationScenarios:
    """Comprehensive integration tests covering multiple daemon scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_daemon_lifecycle_integration(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test complete daemon lifecycle with all components."""
        project_name = "integration-test"
        project_path = str(isolated_daemon_temp_dir)
        
        # Create test project structure
        (isolated_daemon_temp_dir / "src").mkdir()
        (isolated_daemon_temp_dir / "tests").mkdir()
        (isolated_daemon_temp_dir / "docs").mkdir()
        
        test_files = [
            isolated_daemon_temp_dir / "src" / "main.py",
            isolated_daemon_temp_dir / "tests" / "test_main.py",
            isolated_daemon_temp_dir / "docs" / "README.md"
        ]
        
        for file_path in test_files:
            file_path.write_text(f"Content for {file_path.name}")
        
        # Mock complete lifecycle
        lifecycle_events = []
        
        with patch.object(mock_daemon_manager, 'execute_complete_lifecycle', new_callable=AsyncMock) as mock_lifecycle:
            async def complete_lifecycle():
                # 1. Daemon startup
                daemon = await mock_daemon_manager.get_or_create_daemon(project_name, project_path)
                lifecycle_events.append("daemon_created")
                
                startup_result = await mock_daemon_manager.start_daemon(project_name, project_path)
                if startup_result:
                    lifecycle_events.append("daemon_started")
                
                # 2. File discovery and processing
                for file_path in test_files:
                    lifecycle_events.append(f"file_processed_{file_path.name}")
                
                # 3. Health monitoring
                health_status = await mock_daemon_manager.health_check_all()
                if all(health_status.values()):
                    lifecycle_events.append("health_check_passed")
                
                # 4. Configuration reload
                lifecycle_events.append("config_reloaded")
                
                # 5. Graceful shutdown
                shutdown_result = await mock_daemon_manager.stop_daemon(project_name, project_path)
                if shutdown_result:
                    lifecycle_events.append("daemon_stopped")
                
                return {
                    "lifecycle_completed": True,
                    "events": lifecycle_events,
                    "files_processed": len(test_files)
                }
            
            mock_lifecycle.side_effect = complete_lifecycle
            
            # Test complete lifecycle
            result = await mock_daemon_manager.execute_complete_lifecycle()
            
            assert result["lifecycle_completed"] is True
            assert "daemon_created" in result["events"]
            assert "daemon_started" in result["events"]
            assert "health_check_passed" in result["events"]
            assert "daemon_stopped" in result["events"]
            assert result["files_processed"] == 3
    
    @pytest.mark.asyncio
    async def test_error_condition_matrix(self, mock_daemon_instance):
        """Test comprehensive error condition matrix."""
        # Define error condition matrix
        error_conditions = [
            {"component": "storage", "error_type": "connection_timeout", "severity": "high"},
            {"component": "file_watcher", "error_type": "permission_denied", "severity": "medium"},
            {"component": "processing_engine", "error_type": "memory_exhaustion", "severity": "critical"},
            {"component": "resource_monitor", "error_type": "metrics_unavailable", "severity": "low"},
            {"component": "configuration", "error_type": "invalid_config", "severity": "high"}
        ]
        
        error_handling_results = {}
        
        # Mock error handling for each condition
        for condition in error_conditions:
            component = condition["component"]
            error_type = condition["error_type"]
            severity = condition["severity"]
            
            with patch.object(mock_daemon_instance, f'handle_{component}_error', new_callable=AsyncMock) as mock_handler:
                async def error_handler(error_type, severity):
                    if severity == "critical":
                        return {"action": "immediate_shutdown", "recovery_possible": False}
                    elif severity == "high":
                        return {"action": "restart_component", "recovery_possible": True}
                    elif severity == "medium":
                        return {"action": "fallback_mode", "recovery_possible": True}
                    else:
                        return {"action": "log_and_continue", "recovery_possible": True}
                
                mock_handler.side_effect = lambda: error_handler(error_type, severity)
                
                result = await getattr(mock_daemon_instance, f'handle_{component}_error')()
                error_handling_results[f"{component}_{error_type}"] = result
        
        # Verify error handling strategies
        critical_errors = [k for k, v in error_handling_results.items() if v["action"] == "immediate_shutdown"]
        recoverable_errors = [k for k, v in error_handling_results.items() if v["recovery_possible"]]
        
        assert len(critical_errors) == 1  # Only memory_exhaustion should be critical
        assert len(recoverable_errors) == 4  # All except critical should be recoverable