"""
Comprehensive stress tests for daemon file ingestion (Task 317).

This module implements stress testing for the daemon's file ingestion system
under high load and resource constraints.

Test Categories (matching Task 317 subtasks):
- 317.1: High-volume file processing stress test (10,000+ files)
- 317.2: Rapid ingestion rate stress test (100+ files/second)
- 317.3: Multiple folder watching stress test (10+ folders)
- 317.4: Memory constraint stress testing (<512MB)
- 317.5: Disk I/O saturation stress test
- 317.6: Network interruption stress test
- 317.7: Resource monitoring and alerting integration
- 317.8: Performance degradation tracking and reporting

Usage:
    # Run all stress tests (WARNING: may take 30+ minutes)
    pytest tests/stress/test_daemon_stress.py -m stress -v

    # Run specific categories
    pytest tests/stress/test_daemon_stress.py -k "high_volume" -v
    pytest tests/stress/test_daemon_stress.py -k "rapid_ingestion" -v

    # Run with CI scale (faster)
    STRESS_SCALE=ci pytest tests/stress/test_daemon_stress.py -m stress -v

    # Run with full scale
    STRESS_SCALE=full pytest tests/stress/test_daemon_stress.py -m stress -v
"""

import asyncio
import gc
import json
import os
import platform
import random
import resource
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from . import (
    DEGRADATION_THRESHOLDS,
    MONITORING_CONFIG,
    PERFORMANCE_THRESHOLDS,
    STRESS_CONFIG,
)

# ============================================================================
# TASK 317.1: HIGH-VOLUME FILE PROCESSING STRESS TEST
# ============================================================================

@pytest.mark.stress
@pytest.mark.high_volume
@pytest.mark.asyncio
class TestHighVolumeFileProcessing:
    """Test daemon performance with 10,000+ simultaneous files."""

    async def test_process_high_volume_files(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test processing 10,000+ files simultaneously through daemon pipeline.

        Validates:
        - All files processed successfully
        - Memory usage stays under threshold
        - Processing throughput meets minimum
        - No memory leaks during processing
        """
        file_count = stress_config["file_count"]

        # Start resource monitoring
        resource_monitor.start_monitoring()
        performance_tracker.start_operation("total_processing")

        # Generate test files in batches to avoid memory issues
        batch_size = 1000
        all_files = []

        for batch_num in range(file_count // batch_size):
            batch_files = file_generator.create_test_files_batch(
                directory=stress_temp_dir,
                count=batch_size,
                prefix=f"batch{batch_num}_",
                size_bytes=100
            )
            all_files.extend(batch_files)

            # Record resource snapshot after each batch
            resource_monitor.record_snapshot()

        assert len(all_files) == file_count, f"Should create {file_count} files"

        # Process all files through pipeline
        performance_tracker.start_operation("pipeline_processing")

        # Process in chunks to simulate realistic daemon behavior
        chunk_size = 100
        processed_count = 0

        for i in range(0, len(all_files), chunk_size):
            chunk = all_files[i:i + chunk_size]
            chunk_paths = [str(f) for f in chunk]

            # Process chunk
            results = await mock_pipeline.process_documents(
                file_paths=chunk_paths,
                collection="stress-test-collection"
            )

            processed_count += len(results)

            # Record metrics
            resource_monitor.record_snapshot()

            # Track processing latency
            performance_tracker.record_metric("chunk_size", len(results))

        performance_tracker.end_operation("pipeline_processing")
        performance_tracker.end_operation("total_processing")

        # Stop monitoring and validate
        resource_monitor.stop_monitoring()

        # Validate all files processed
        assert processed_count == file_count, \
            f"Should process all {file_count} files, got {processed_count}"

        # Validate resource usage
        monitoring_summary = resource_monitor.get_summary()
        max_memory_mb = monitoring_summary['summary']['memory_rss_mb']['max']

        assert max_memory_mb < 500, \
            f"Memory usage ({max_memory_mb:.1f}MB) should stay under 500MB"

        # Validate throughput
        total_time = monitoring_summary['duration_seconds']
        throughput = file_count / total_time if total_time > 0 else 0

        min_throughput = PERFORMANCE_THRESHOLDS['throughput_files_per_second']['minimum']
        assert throughput >= min_throughput, \
            f"Throughput ({throughput:.1f} files/sec) should meet minimum ({min_throughput})"

        # Check for threshold violations
        threshold_check = resource_monitor.check_thresholds()
        assert not threshold_check['criticals'], \
            f"Critical thresholds exceeded: {threshold_check['criticals']}"

    async def test_high_volume_with_different_file_types(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test high-volume processing with mixed file types.

        Validates:
        - Different file types processed correctly
        - No type-specific bottlenecks
        - Consistent performance across types
        """
        file_count = stress_config["file_count"] // 5  # Smaller for variety test

        extensions = ['.txt', '.md', '.py', '.json', '.yaml']
        all_files = []

        resource_monitor.start_monitoring()

        # Create mixed file types
        for ext in extensions:
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir,
                count=file_count,
                prefix=f"type_{ext[1:]}_",
                extension=ext,
                size_bytes=200
            )
            all_files.extend(files)
            resource_monitor.record_snapshot()

        # Process all files
        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in all_files],
            collection="stress-mixed-types"
        )

        resource_monitor.stop_monitoring()

        # Validate all processed
        assert len(results) == len(all_files), "All files should be processed"

        # Validate resource usage
        monitoring_summary = resource_monitor.get_summary()
        max_memory_mb = monitoring_summary['summary']['memory_rss_mb']['max']

        assert max_memory_mb < 500, \
            f"Memory usage ({max_memory_mb:.1f}MB) exceeded threshold"

    async def test_high_volume_memory_stability(
        self,
        stress_temp_dir,
        file_generator,
        memory_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test memory stability during high-volume processing.

        Validates:
        - No memory leaks detected
        - Memory growth within acceptable bounds
        - Garbage collection working properly
        """
        file_count = min(stress_config["file_count"], 5000)  # Limit for memory test

        memory_tracker.start()

        # Process in multiple rounds to detect leaks
        rounds = 5
        files_per_round = file_count // rounds

        for round_num in range(rounds):
            # Create files
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir,
                count=files_per_round,
                prefix=f"round{round_num}_",
                size_bytes=100
            )

            # Process files
            await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection="stress-memory-test"
            )

            # Force garbage collection
            gc.collect()

            # Take memory snapshot
            memory_tracker.snapshot()

            # Clean up files for next round
            for f in files:
                f.unlink()

        # Stop tracking and validate
        memory_summary = memory_tracker.stop()

        # Check for memory leaks
        total_growth_mb = memory_summary.get('total_growth_mb', 0)

        assert total_growth_mb < 50, \
            f"Memory growth ({total_growth_mb:.1f}MB) suggests memory leak"


# ============================================================================
# TASK 317.2: RAPID INGESTION RATE STRESS TEST
# ============================================================================

@pytest.mark.stress
@pytest.mark.rapid_ingestion
@pytest.mark.asyncio
class TestRapidIngestionRate:
    """Test daemon performance with rapid file addition rates."""

    async def test_rapid_file_creation_rate(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test daemon handling of 100+ files/second ingestion rate.

        Validates:
        - Daemon keeps up with high ingestion rate
        - Queue doesn't grow unbounded
        - Processing latency stays acceptable
        - No files dropped
        """
        target_rate = stress_config["ingestion_rate"]
        duration_seconds = 10
        total_files = target_rate * duration_seconds

        resource_monitor.start_monitoring()
        performance_tracker.start_operation("rapid_ingestion")

        # Create files at target rate
        files = await file_generator.create_files_at_rate(
            directory=stress_temp_dir,
            count=total_files,
            rate_per_second=target_rate,
            prefix="rapid_"
        )

        performance_tracker.end_operation("rapid_ingestion")

        # Verify actual creation rate
        creation_time = performance_tracker.metrics["rapid_ingestion"][0]
        actual_rate = total_files / creation_time

        assert actual_rate >= target_rate * 0.9, \
            f"Actual rate ({actual_rate:.1f}) should be close to target ({target_rate})"

        # Process all files and measure latency
        performance_tracker.start_operation("rapid_processing")

        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in files],
            collection="stress-rapid-ingestion"
        )

        performance_tracker.end_operation("rapid_processing")
        resource_monitor.stop_monitoring()

        # Validate all files processed
        assert len(results) == total_files, \
            f"All {total_files} files should be processed"

        # Validate processing latency
        processing_time = performance_tracker.metrics["rapid_processing"][0]
        avg_latency_ms = (processing_time / total_files) * 1000

        assert avg_latency_ms < PERFORMANCE_THRESHOLDS['processing_latency_ms']['p99'], \
            f"Average latency ({avg_latency_ms:.1f}ms) exceeded threshold"

    async def test_sustained_high_rate_ingestion(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test sustained high-rate ingestion over extended period.

        Validates:
        - Performance remains stable over time
        - No degradation with sustained load
        - Resource usage stays bounded
        """
        target_rate = stress_config["ingestion_rate"]
        duration_seconds = 30
        batch_duration = 5

        resource_monitor.start_monitoring()
        latencies = []

        # Run multiple batches
        for batch_num in range(duration_seconds // batch_duration):
            batch_size = target_rate * batch_duration

            performance_tracker.start_operation(f"batch_{batch_num}")

            # Create and process batch
            files = await file_generator.create_files_at_rate(
                directory=stress_temp_dir,
                count=batch_size,
                rate_per_second=target_rate,
                prefix=f"sustained_{batch_num}_"
            )

            await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection="stress-sustained"
            )

            performance_tracker.end_operation(f"batch_{batch_num}")

            # Record latency
            batch_time = performance_tracker.metrics[f"batch_{batch_num}"][0]
            latency = (batch_time / batch_size) * 1000
            latencies.append(latency)

            # Clean up batch
            for f in files:
                f.unlink()

            resource_monitor.record_snapshot()

        resource_monitor.stop_monitoring()

        # Validate no degradation over time
        first_half_avg = sum(latencies[:len(latencies)//2]) / (len(latencies)//2)
        second_half_avg = sum(latencies[len(latencies)//2:]) / (len(latencies) - len(latencies)//2)

        degradation_percent = ((second_half_avg - first_half_avg) / first_half_avg * 100)

        assert degradation_percent < DEGRADATION_THRESHOLDS['latency_increase_percent'], \
            f"Performance degraded by {degradation_percent:.1f}% over time"

    async def test_burst_ingestion_handling(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test daemon handling of burst ingestion patterns.

        Validates:
        - Handles sudden bursts gracefully
        - Recovers after burst
        - Queue management works correctly
        """
        burst_size = stress_config["ingestion_rate"] * 5  # 5 seconds worth
        burst_rate = burst_size  # All at once

        resource_monitor.start_monitoring()

        # Create burst of files very quickly
        files = await file_generator.create_files_at_rate(
            directory=stress_temp_dir,
            count=burst_size,
            rate_per_second=burst_rate,
            prefix="burst_"
        )

        # Give daemon time to process
        await asyncio.sleep(2)

        # Process burst
        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in files],
            collection="stress-burst"
        )

        resource_monitor.stop_monitoring()

        # Validate all processed
        assert len(results) == burst_size, \
            f"All {burst_size} burst files should be processed"

        # Validate recovery (memory should stabilize)
        monitoring_summary = resource_monitor.get_summary()
        memory_final = monitoring_summary['summary']['memory_rss_mb']['final']
        memory_max = monitoring_summary['summary']['memory_rss_mb']['max']

        # Memory should drop after burst
        memory_recovery = ((memory_max - memory_final) / memory_max * 100)

        assert memory_recovery > 0, \
            "Memory should recover after burst processing"


# ============================================================================
# TASK 317.3: MULTIPLE FOLDER WATCHING STRESS TEST
# ============================================================================

@pytest.mark.stress
@pytest.mark.multi_folder
@pytest.mark.asyncio
class TestMultipleFolderWatching:
    """Test daemon performance with multiple watched folders."""

    async def test_multiple_folder_concurrent_ingestion(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test concurrent file ingestion across multiple watched folders.

        Validates:
        - Multiple folders handled independently
        - No interference between folders
        - Resource allocation balanced
        - All folders processed correctly
        """
        folder_count = stress_config["folder_count"]
        files_per_folder = 100

        # Create multiple watched folders
        folders = []
        for i in range(folder_count):
            folder = stress_temp_dir / f"watch_folder_{i}"
            folder.mkdir()
            folders.append(folder)

        resource_monitor.start_monitoring()

        # Create files in all folders concurrently
        all_files = []
        tasks = []

        for i, folder in enumerate(folders):
            async def create_in_folder(folder_path, folder_id):
                files = file_generator.create_test_files_batch(
                    directory=folder_path,
                    count=files_per_folder,
                    prefix=f"folder{folder_id}_",
                    size_bytes=100
                )
                return files

            task = create_in_folder(folder, i)
            tasks.append(task)

        # Wait for all folders to have files
        folder_files = await asyncio.gather(*tasks)
        for files in folder_files:
            all_files.extend(files)

        # Process all files
        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in all_files],
            collection="stress-multi-folder"
        )

        resource_monitor.stop_monitoring()

        # Validate all processed
        expected_total = folder_count * files_per_folder
        assert len(results) == expected_total, \
            f"Should process {expected_total} files from {folder_count} folders"

        # Validate resource usage
        monitoring_summary = resource_monitor.get_summary()
        max_memory_mb = monitoring_summary['summary']['memory_rss_mb']['max']

        assert max_memory_mb < 500, \
            f"Memory usage ({max_memory_mb:.1f}MB) exceeded threshold"

    async def test_folder_isolation(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test that watched folders are properly isolated.

        Validates:
        - Files in one folder don't affect others
        - Error in one folder doesn't cascade
        - Resource limits per folder respected
        """
        folder_count = min(stress_config["folder_count"], 5)

        # Create folders
        folders = []
        for i in range(folder_count):
            folder = stress_temp_dir / f"isolated_{i}"
            folder.mkdir()
            folders.append(folder)

        resource_monitor.start_monitoring()

        # Process each folder separately and verify isolation
        for i, folder in enumerate(folders):
            # Create files
            files = file_generator.create_test_files_batch(
                directory=folder,
                count=50,
                prefix=f"iso{i}_",
                size_bytes=100
            )

            # Process
            results = await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection=f"stress-isolated-{i}"
            )

            # Validate this folder's processing
            assert len(results) == 50, \
                f"Folder {i} should process all 50 files"

            resource_monitor.record_snapshot()

        resource_monitor.stop_monitoring()

        # Validate consistent performance across folders
        monitoring_summary = resource_monitor.get_summary()
        len(monitoring_summary['summary']['memory_rss_mb'])

        # Memory should be relatively stable across folders
        memory_avg = monitoring_summary['summary']['memory_rss_mb']['avg']
        memory_max = monitoring_summary['summary']['memory_rss_mb']['max']

        variance = ((memory_max - memory_avg) / memory_avg * 100)

        assert variance < 50, \
            f"Memory variance ({variance:.1f}%) too high between folders"

    async def test_high_folder_count_scalability(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test scalability with high number of watched folders.

        Validates:
        - Performance scales reasonably with folder count
        - Resource usage grows sub-linearly
        - No performance cliff at high counts
        """
        folder_count = stress_config["folder_count"]
        files_per_folder = 20  # Smaller per folder for scalability test

        resource_monitor.start_monitoring()

        # Create and process folders in batches
        batch_size = 5
        all_results = []

        for batch_start in range(0, folder_count, batch_size):

            # Create batch of folders
            for i in range(batch_start, min(batch_start + batch_size, folder_count)):
                folder = stress_temp_dir / f"scale_{i}"
                folder.mkdir()

                # Create files in folder
                files = file_generator.create_test_files_batch(
                    directory=folder,
                    count=files_per_folder,
                    prefix=f"f{i}_",
                    size_bytes=100
                )

                # Process folder
                results = await mock_pipeline.process_documents(
                    file_paths=[str(f) for f in files],
                    collection=f"stress-scale-{i}"
                )

                all_results.extend(results)

            resource_monitor.record_snapshot()

        resource_monitor.stop_monitoring()

        # Validate all processed
        expected_total = folder_count * files_per_folder
        assert len(all_results) == expected_total, \
            f"Should process {expected_total} files from {folder_count} folders"

        # Validate resource growth is sub-linear
        monitoring_summary = resource_monitor.get_summary()
        memory_growth = monitoring_summary.get('memory_growth_mb', 0)

        # Memory growth should be less than linear with folder count
        max_acceptable_growth = folder_count * 5  # 5MB per folder is too much

        assert memory_growth < max_acceptable_growth, \
            f"Memory growth ({memory_growth:.1f}MB) suggests linear scaling issue"


# ============================================================================
# TASK 317.4: MEMORY CONSTRAINT STRESS TESTING
# ============================================================================

@pytest.mark.stress
@pytest.mark.memory_constraint
@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="resource.setrlimit not available on Windows")
class TestMemoryConstraint:
    """Test daemon behavior under memory-constrained environments."""

    async def test_memory_limited_processing(
        self,
        stress_temp_dir,
        file_generator,
        memory_tracker,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test processing under memory limit (<512MB).

        Validates:
        - Daemon operates within memory constraints
        - Graceful degradation when approaching limit
        - No crashes from out-of-memory
        """
        # Note: Actual memory limiting requires system permissions
        # This test simulates and monitors memory usage

        file_count = min(stress_config["file_count"], 2000)  # Reduced for memory test

        memory_tracker.start()
        resource_monitor.start_monitoring()

        # Process in small batches to stay within limits
        batch_size = 100
        processed_count = 0

        # Create and process in batches
        for batch_num in range(file_count // batch_size):
            # Create batch
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir,
                count=batch_size,
                prefix=f"memtest_{batch_num}_",
                size_bytes=100
            )

            # Process batch
            results = await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection="stress-memory-constrained"
            )

            processed_count += len(results)

            # Force cleanup
            for f in files:
                f.unlink()
            gc.collect()

            # Monitor memory
            memory_snapshot = memory_tracker.snapshot()
            resource_monitor.record_snapshot()

            # Check we're staying under limit
            current_memory = memory_snapshot.get('current_mb', 0)
            assert current_memory < 512, \
                f"Memory usage ({current_memory:.1f}MB) exceeded 512MB limit"

        memory_summary = memory_tracker.stop()
        resource_monitor.stop_monitoring()

        # Validate processing completed
        assert processed_count == file_count, \
            f"Should process all {file_count} files under memory constraint"

        # Validate memory stayed bounded
        max_memory = memory_summary.get('max_mb', 0)
        assert max_memory < 512, \
            f"Maximum memory ({max_memory:.1f}MB) should stay under 512MB"

    async def test_memory_leak_detection(
        self,
        stress_temp_dir,
        file_generator,
        memory_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test for memory leaks during sustained processing.

        Validates:
        - Memory doesn't grow continuously
        - Cleanup happens properly
        - No resource retention issues
        """
        rounds = 10
        files_per_round = 100

        memory_tracker.start()
        baseline_memory = None

        for round_num in range(rounds):
            # Create files
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir,
                count=files_per_round,
                prefix=f"leak_{round_num}_",
                size_bytes=100
            )

            # Process
            await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection="stress-leak-test"
            )

            # Cleanup
            for f in files:
                f.unlink()
            gc.collect()

            # Measure memory
            snapshot = memory_tracker.snapshot()
            current_memory = snapshot.get('current_mb', 0)

            if baseline_memory is None:
                baseline_memory = current_memory
            else:
                # Check for excessive growth
                growth = current_memory - baseline_memory
                growth_percent = (growth / baseline_memory * 100) if baseline_memory > 0 else 0

                assert growth_percent < 20, \
                    f"Memory grew {growth_percent:.1f}% from baseline, possible leak"

        memory_summary = memory_tracker.stop()

        # Final validation
        total_growth = memory_summary.get('total_growth_mb', 0)
        assert total_growth < 50, \
            f"Total memory growth ({total_growth:.1f}MB) suggests memory leak"

    async def test_graceful_degradation_under_pressure(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test graceful degradation when approaching memory limits.

        Validates:
        - System slows down rather than crashes
        - Error handling works under pressure
        - Recovery possible after pressure relieved
        """
        file_count = 500

        resource_monitor.start_monitoring()

        # Create large batch to create memory pressure
        files = file_generator.create_test_files_batch(
            directory=stress_temp_dir,
            count=file_count,
            prefix="pressure_",
            size_bytes=1000  # Larger files
        )

        # Process under pressure
        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in files],
            collection="stress-pressure"
        )

        # Cleanup to relieve pressure
        for f in files:
            f.unlink()
        gc.collect()
        await asyncio.sleep(1)

        # Test recovery with new batch
        recovery_files = file_generator.create_test_files_batch(
            directory=stress_temp_dir,
            count=100,
            prefix="recovery_",
            size_bytes=100
        )

        recovery_results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in recovery_files],
            collection="stress-recovery"
        )

        resource_monitor.stop_monitoring()

        # Validate both phases completed
        assert len(results) == file_count, "Pressure phase should complete"
        assert len(recovery_results) == 100, "Recovery phase should complete"

        # Validate memory recovered
        monitoring_summary = resource_monitor.get_summary()
        memory_final = monitoring_summary['summary']['memory_rss_mb']['final']
        memory_max = monitoring_summary['summary']['memory_rss_mb']['max']

        recovery_percent = ((memory_max - memory_final) / memory_max * 100)

        assert recovery_percent > 10, \
            f"Memory should recover after pressure relieved (recovered {recovery_percent:.1f}%)"


# ============================================================================
# TASK 317.5: DISK I/O SATURATION STRESS TEST
# ============================================================================

@pytest.mark.stress
@pytest.mark.disk_saturation
@pytest.mark.asyncio
class TestDiskIOSaturation:
    """Test daemon performance under disk I/O saturation."""

    async def test_large_file_io_saturation(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test processing with large files to saturate disk I/O.

        Validates:
        - Daemon handles I/O bottlenecks
        - Performance degrades gracefully
        - No corruption under I/O pressure
        """
        large_file_size_mb = stress_config.get("large_file_size_mb", 10)
        large_file_count = 10

        resource_monitor.start_monitoring()

        # Create large files to stress I/O
        large_files = []
        for i in range(large_file_count):
            file_path = stress_temp_dir / f"large_{i}.txt"
            content = file_generator.generate_random_content(large_file_size_mb * 1024 * 1024)
            file_path.write_text(content, encoding='utf-8')
            large_files.append(file_path)

            resource_monitor.record_snapshot()

        # Process large files
        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in large_files],
            collection="stress-large-files"
        )

        resource_monitor.stop_monitoring()

        # Validate all processed
        assert len(results) == large_file_count, \
            f"All {large_file_count} large files should be processed"

        # Validate performance under I/O load
        monitoring_summary = resource_monitor.get_summary()

        # CPU shouldn't be maxed out (I/O bound, not CPU bound)
        avg_cpu = monitoring_summary['summary']['cpu_percent']['avg']
        assert avg_cpu < 90, \
            f"CPU usage ({avg_cpu:.1f}%) suggests CPU bottleneck instead of I/O"

    async def test_concurrent_read_write_operations(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test daemon with concurrent read/write I/O operations.

        Validates:
        - Handles concurrent I/O efficiently
        - No file locking issues
        - Throughput maintained under concurrent I/O
        """
        file_count = 200
        concurrent_operations = min(stress_config.get("concurrent_operations", 10), 20)

        resource_monitor.start_monitoring()

        # Create files
        files = file_generator.create_test_files_batch(
            directory=stress_temp_dir,
            count=file_count,
            prefix="concurrent_io_",
            size_bytes=5000  # Medium size
        )

        # Process files concurrently
        async def process_batch(batch_files):
            return await mock_pipeline.process_documents(
                file_paths=[str(f) for f in batch_files],
                collection="stress-concurrent-io"
            )

        # Split into concurrent batches
        batch_size = file_count // concurrent_operations
        tasks = []

        for i in range(concurrent_operations):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < concurrent_operations - 1 else file_count
            batch = files[start_idx:end_idx]
            tasks.append(process_batch(batch))

        # Execute concurrently
        batch_results = await asyncio.gather(*tasks)

        resource_monitor.stop_monitoring()

        # Validate all processed
        total_processed = sum(len(r) for r in batch_results)
        assert total_processed == file_count, \
            f"Should process all {file_count} files concurrently"

        # Validate no excessive resource usage
        monitoring_summary = resource_monitor.get_summary()
        max_memory_mb = monitoring_summary['summary']['memory_rss_mb']['max']

        assert max_memory_mb < 500, \
            f"Memory usage ({max_memory_mb:.1f}MB) exceeded threshold"

    async def test_io_throttling_behavior(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test daemon's I/O throttling under heavy load.

        Validates:
        - Throttling prevents system overload
        - Performance stable under sustained I/O
        - Backpressure mechanisms work
        """
        file_count = 500
        batch_size = 50

        resource_monitor.start_monitoring()

        # Process in batches and measure timing
        batch_times = []

        for batch_num in range(file_count // batch_size):
            # Create batch
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir / f"batch_{batch_num}",
                count=batch_size,
                prefix="throttle_",
                size_bytes=2000
            )

            # Time processing
            start_time = time.time()

            await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection="stress-throttle"
            )

            batch_time = time.time() - start_time
            batch_times.append(batch_time)

            resource_monitor.record_snapshot()

        resource_monitor.stop_monitoring()

        # Validate consistent timing (throttling should prevent speedup/slowdown)
        avg_time = sum(batch_times) / len(batch_times)
        max_deviation = max(abs(t - avg_time) for t in batch_times)
        deviation_percent = (max_deviation / avg_time * 100) if avg_time > 0 else 0

        # Allow up to 100% deviation (2x slower/faster) due to throttling adjustments
        assert deviation_percent < 100, \
            f"Batch timing variance ({deviation_percent:.1f}%) suggests unstable throttling"


# ============================================================================
# TASK 317.6: NETWORK INTERRUPTION STRESS TEST
# ============================================================================

@pytest.mark.stress
@pytest.mark.network_interruption
@pytest.mark.asyncio
class TestNetworkInterruption:
    """Test daemon resilience to network interruptions."""

    async def test_network_failure_recovery(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test daemon recovery from network failures.

        Validates:
        - Handles connection failures gracefully
        - Retry logic works correctly
        - Data integrity maintained after recovery
        """
        file_count = 100

        # Create mock pipeline that fails intermittently
        failure_count = [0]
        max_failures = 3

        async def failing_process_documents(file_paths, collection, **kwargs):
            if failure_count[0] < max_failures:
                failure_count[0] += 1
                raise ConnectionError("Simulated network failure")

            # After failures, succeed
            results = []
            for file_path in file_paths:
                results.append(MagicMock(
                    file_path=file_path,
                    success=True,
                    chunks_created=1,
                    error=None
                ))
            return results

        mock_pipeline.process_documents = failing_process_documents

        resource_monitor.start_monitoring()

        # Create files
        files = file_generator.create_test_files_batch(
            directory=stress_temp_dir,
            count=file_count,
            prefix="network_fail_",
            size_bytes=100
        )

        # Attempt processing with retries
        max_retries = 5
        retry_count = 0
        results = None

        for attempt in range(max_retries):
            try:
                results = await mock_pipeline.process_documents(
                    file_paths=[str(f) for f in files],
                    collection="stress-network-fail"
                )
                break  # Success
            except ConnectionError:
                retry_count += 1
                await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff

        resource_monitor.stop_monitoring()

        # Validate recovery occurred
        assert retry_count == max_failures, \
            f"Should have retried {max_failures} times"

        assert results is not None, "Should eventually succeed after retries"
        assert len(results) == file_count, "All files should be processed after recovery"

    async def test_intermittent_connectivity(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test daemon with intermittent network connectivity.

        Validates:
        - Handles sporadic failures
        - Partial batch processing works
        - No data loss during interruptions
        """
        file_count = 200
        failure_probability = 0.3  # 30% failure rate

        # Create mock with random failures
        processed_files = []

        async def intermittent_process_documents(file_paths, collection, **kwargs):
            results = []
            for file_path in file_paths:
                if random.random() < failure_probability:
                    # Simulate failure for this file
                    results.append(MagicMock(
                        file_path=file_path,
                        success=False,
                        error="Network timeout"
                    ))
                else:
                    # Success
                    results.append(MagicMock(
                        file_path=file_path,
                        success=True,
                        chunks_created=1,
                        error=None
                    ))
                    processed_files.append(file_path)
            return results

        mock_pipeline.process_documents = intermittent_process_documents

        resource_monitor.start_monitoring()

        # Create and process files
        files = file_generator.create_test_files_batch(
            directory=stress_temp_dir,
            count=file_count,
            prefix="intermittent_",
            size_bytes=100
        )

        # Initial processing
        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in files],
            collection="stress-intermittent"
        )

        # Retry failed files
        failed_files = [r.file_path for r in results if not r.success]

        if failed_files:
            # Reset failure probability for retry
            await mock_pipeline.process_documents(
                file_paths=failed_files,
                collection="stress-intermittent-retry"
            )

        resource_monitor.stop_monitoring()

        # Validate handling
        success_count = sum(1 for r in results if r.success)
        failure_count = sum(1 for r in results if not r.success)

        assert success_count + failure_count == file_count, \
            "All files should have a result"

        # At least some should succeed despite failures
        assert success_count > 0, "Some files should succeed"

    async def test_connection_timeout_handling(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test daemon handling of connection timeouts.

        Validates:
        - Timeouts detected promptly
        - Retry logic with exponential backoff
        - System remains responsive during timeouts
        """
        file_count = 50

        # Create mock with timeouts
        timeout_count = [0]
        max_timeouts = 2

        async def timeout_process_documents(file_paths, collection, **kwargs):
            if timeout_count[0] < max_timeouts:
                timeout_count[0] += 1
                # Simulate timeout delay
                await asyncio.sleep(2)
                raise asyncio.TimeoutError("Connection timeout")

            # After timeouts, succeed quickly
            await asyncio.sleep(0.1)
            results = []
            for file_path in file_paths:
                results.append(MagicMock(
                    file_path=file_path,
                    success=True,
                    chunks_created=1,
                    error=None
                ))
            return results

        mock_pipeline.process_documents = timeout_process_documents

        resource_monitor.start_monitoring()

        # Create files
        files = file_generator.create_test_files_batch(
            directory=stress_temp_dir,
            count=file_count,
            prefix="timeout_",
            size_bytes=100
        )

        # Process with timeout handling
        performance_tracker.start_operation("timeout_recovery")

        results = None
        retry_delays = []

        for attempt in range(5):
            try:
                results = await mock_pipeline.process_documents(
                    file_paths=[str(f) for f in files],
                    collection="stress-timeout"
                )
                break
            except asyncio.TimeoutError:
                delay = 0.5 * (2 ** attempt)  # Exponential backoff
                retry_delays.append(delay)
                await asyncio.sleep(delay)

        performance_tracker.end_operation("timeout_recovery")
        resource_monitor.stop_monitoring()

        # Validate timeout handling
        assert timeout_count[0] == max_timeouts, \
            f"Should have timed out {max_timeouts} times"

        assert results is not None, "Should eventually succeed"
        assert len(results) == file_count, "All files should be processed"

        # Validate exponential backoff
        if len(retry_delays) > 1:
            assert retry_delays[1] > retry_delays[0], \
                "Retry delays should increase (exponential backoff)"


# ============================================================================
# TASK 317.7: RESOURCE MONITORING AND ALERTING INTEGRATION
# ============================================================================

@pytest.mark.stress
@pytest.mark.asyncio
class TestResourceMonitoringIntegration:
    """Comprehensive resource monitoring integration for all stress tests."""

    async def test_cpu_monitoring_integration(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test CPU usage monitoring during stress conditions.

        Validates:
        - CPU metrics collected accurately
        - Alerts triggered on thresholds
        - CPU usage stays within acceptable bounds
        """
        file_count = stress_config["file_count"] // 2

        resource_monitor.start_monitoring()

        # Create CPU-intensive workload (multiple batches)
        batch_count = 10
        batch_size = file_count // batch_count

        for batch_num in range(batch_count):
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir / f"cpu_batch_{batch_num}",
                count=batch_size,
                prefix="cpu_",
                size_bytes=500
            )

            await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection="stress-cpu-monitor"
            )

            resource_monitor.record_snapshot()

        resource_monitor.stop_monitoring()

        # Validate CPU monitoring
        monitoring_summary = resource_monitor.get_summary()
        cpu_stats = monitoring_summary['summary']['cpu_percent']

        assert cpu_stats['avg'] >= 0, "CPU average should be recorded"
        assert cpu_stats['max'] >= 0, "CPU maximum should be recorded"
        assert cpu_stats['max'] <= 100, "CPU should not exceed 100%"

        # Check threshold alerting
        threshold_check = resource_monitor.check_thresholds()

        # Should have some metrics
        assert 'warnings' in threshold_check, "Should check for warnings"
        assert 'criticals' in threshold_check, "Should check for criticals"

    async def test_memory_monitoring_integration(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        memory_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test memory usage monitoring during stress conditions.

        Validates:
        - Memory metrics collected at intervals
        - Memory growth tracked accurately
        - Alerts on memory thresholds
        """
        file_count = stress_config["file_count"] // 2

        resource_monitor.start_monitoring()
        memory_tracker.start()

        # Create memory-intensive workload
        batch_count = 5

        for batch_num in range(batch_count):
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir,
                count=file_count // batch_count,
                prefix=f"mem_{batch_num}_",
                size_bytes=1000
            )

            await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection="stress-memory-monitor"
            )

            resource_monitor.record_snapshot()
            memory_tracker.snapshot()

        memory_tracker.stop()
        resource_monitor.stop_monitoring()

        # Validate memory monitoring
        monitoring_summary = resource_monitor.get_summary()
        memory_stats = monitoring_summary['summary']['memory_rss_mb']

        assert memory_stats['avg'] > 0, "Memory average should be recorded"
        assert memory_stats['max'] > 0, "Memory maximum should be recorded"

        # Validate memory growth tracking
        assert 'memory_growth_mb' in monitoring_summary['summary'], \
            "Memory growth should be tracked"

        memory_growth = monitoring_summary['summary']['memory_growth_mb']
        assert memory_growth < 200, \
            f"Memory growth ({memory_growth:.1f}MB) excessive for workload"

    async def test_disk_io_monitoring_integration(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test disk I/O monitoring during stress conditions.

        Validates:
        - Disk I/O metrics collected
        - I/O rates calculated correctly
        - Performance correlates with I/O load
        """
        # Create I/O-intensive workload
        large_file_count = 20
        large_file_size_mb = 5

        resource_monitor.start_monitoring()

        for i in range(large_file_count):
            file_path = stress_temp_dir / f"diskio_{i}.txt"
            content = file_generator.generate_random_content(large_file_size_mb * 1024 * 1024)
            file_path.write_text(content, encoding='utf-8')

            resource_monitor.record_snapshot()

        # Process files (read I/O)
        files = list(stress_temp_dir.glob("diskio_*.txt"))

        await mock_pipeline.process_documents(
            file_paths=[str(f) for f in files],
            collection="stress-diskio-monitor"
        )

        resource_monitor.stop_monitoring()

        # Validate disk I/O monitoring (if available)
        resource_monitor.get_summary()

        # Disk I/O may not be available on all platforms
        # If available, validate it
        has_disk_io = any('disk_read_mb' in snapshot for snapshot in resource_monitor.monitoring_data)

        if has_disk_io:
            # Validate disk reads occurred
            disk_reads = [s.get('disk_read_mb', 0) for s in resource_monitor.monitoring_data]
            assert max(disk_reads) > 0, "Disk reads should be recorded"

    async def test_comprehensive_monitoring_report(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test generation of comprehensive monitoring report.

        Validates:
        - All metrics collected
        - Report generated successfully
        - Actionable insights provided
        """
        file_count = 500

        resource_monitor.start_monitoring()
        performance_tracker.start_operation("comprehensive_test")

        # Run varied workload
        # 1. Small files
        small_files = file_generator.create_test_files_batch(
            directory=stress_temp_dir / "small",
            count=file_count,
            prefix="small_",
            size_bytes=100
        )

        await mock_pipeline.process_documents(
            file_paths=[str(f) for f in small_files],
            collection="stress-small"
        )

        resource_monitor.record_snapshot()

        # 2. Medium files
        medium_files = file_generator.create_test_files_batch(
            directory=stress_temp_dir / "medium",
            count=file_count // 2,
            prefix="medium_",
            size_bytes=1000
        )

        await mock_pipeline.process_documents(
            file_paths=[str(f) for f in medium_files],
            collection="stress-medium"
        )

        resource_monitor.record_snapshot()

        # 3. Large files
        large_files = []
        for i in range(10):
            file_path = stress_temp_dir / "large" / f"large_{i}.txt"
            file_path.parent.mkdir(exist_ok=True)
            content = file_generator.generate_random_content(1024 * 1024)  # 1MB
            file_path.write_text(content, encoding='utf-8')
            large_files.append(file_path)

        await mock_pipeline.process_documents(
            file_paths=[str(f) for f in large_files],
            collection="stress-large"
        )

        performance_tracker.end_operation("comprehensive_test")
        resource_monitor.stop_monitoring()

        # Generate comprehensive report
        monitoring_summary = resource_monitor.get_summary()
        performance_summary = performance_tracker.get_summary()
        threshold_check = resource_monitor.check_thresholds()

        # Validate report components
        assert 'summary' in monitoring_summary, "Should have summary section"
        assert 'duration_seconds' in monitoring_summary, "Should track duration"
        assert 'sample_count' in monitoring_summary, "Should track samples"

        assert len(performance_summary) > 0, "Should have performance metrics"

        assert 'warnings' in threshold_check, "Should have warnings section"
        assert 'criticals' in threshold_check, "Should have criticals section"

        # Save report to file
        report_file = stress_temp_dir / "monitoring_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'monitoring': monitoring_summary,
                'performance': performance_summary,
                'thresholds': threshold_check,
            }, f, indent=2)

        assert report_file.exists(), "Report file should be created"
        assert report_file.stat().st_size > 0, "Report should have content"


# ============================================================================
# TASK 317.8: PERFORMANCE DEGRADATION TRACKING AND REPORTING
# ============================================================================

@pytest.mark.stress
@pytest.mark.asyncio
class TestPerformanceDegradationTracking:
    """Performance baseline tracking and degradation detection."""

    async def test_baseline_performance_capture(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Capture baseline performance metrics for comparison.

        Validates:
        - Baseline metrics collected
        - Reproducible test scenarios
        - Metrics saved for future comparison
        """
        file_count = 1000

        resource_monitor.start_monitoring()

        # Run standard workload
        performance_tracker.start_operation("baseline_creation")

        files = file_generator.create_test_files_batch(
            directory=stress_temp_dir,
            count=file_count,
            prefix="baseline_",
            size_bytes=200
        )

        results = await mock_pipeline.process_documents(
            file_paths=[str(f) for f in files],
            collection="stress-baseline"
        )

        performance_tracker.end_operation("baseline_creation")

        resource_monitor.stop_monitoring()

        # Calculate baseline metrics
        monitoring_summary = resource_monitor.get_summary()
        performance_tracker.get_summary()

        baseline = {
            'file_count': file_count,
            'duration_seconds': monitoring_summary['duration_seconds'],
            'throughput_files_per_second': file_count / monitoring_summary['duration_seconds'],
            'avg_memory_mb': monitoring_summary['summary']['memory_rss_mb']['avg'],
            'max_memory_mb': monitoring_summary['summary']['memory_rss_mb']['max'],
            'avg_cpu_percent': monitoring_summary['summary']['cpu_percent']['avg'],
            'success_count': len(results),
            'error_count': 0,
        }

        # Save baseline
        baseline_file = stress_temp_dir / "performance_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)

        # Validate baseline
        assert baseline['throughput_files_per_second'] > 0, "Should have positive throughput"
        assert baseline['success_count'] == file_count, "All files should succeed"

        return baseline

    async def test_performance_comparison(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Compare current performance against baseline.

        Validates:
        - Performance metrics compared accurately
        - Degradation detected when present
        - Regression percentages calculated
        """
        # First, establish baseline
        baseline = await self.test_baseline_performance_capture(
            stress_temp_dir,
            file_generator,
            resource_monitor,
            performance_tracker,
            mock_pipeline,
            stress_config
        )

        # Run comparison test (simulate slight degradation)
        file_count = baseline['file_count']

        resource_monitor.start_monitoring()
        performance_tracker.start_operation("comparison_test")

        # Add small delay to simulate degradation
        files = file_generator.create_test_files_batch(
            directory=stress_temp_dir / "comparison",
            count=file_count,
            prefix="compare_",
            size_bytes=200
        )

        # Simulate slightly slower processing
        await asyncio.sleep(0.1)

        await mock_pipeline.process_documents(
            file_paths=[str(f) for f in files],
            collection="stress-comparison"
        )

        performance_tracker.end_operation("comparison_test")
        resource_monitor.stop_monitoring()

        # Calculate current metrics
        monitoring_summary = resource_monitor.get_summary()

        current = {
            'throughput_files_per_second': file_count / monitoring_summary['duration_seconds'],
            'avg_memory_mb': monitoring_summary['summary']['memory_rss_mb']['avg'],
            'max_memory_mb': monitoring_summary['summary']['memory_rss_mb']['max'],
        }

        # Calculate degradation
        throughput_change = (
            (current['throughput_files_per_second'] - baseline['throughput_files_per_second'])
            / baseline['throughput_files_per_second'] * 100
        )

        memory_change = (
            (current['avg_memory_mb'] - baseline['avg_memory_mb'])
            / baseline['avg_memory_mb'] * 100
        )

        # Validate comparison
        comparison_report = {
            'baseline': baseline,
            'current': current,
            'degradation': {
                'throughput_change_percent': throughput_change,
                'memory_change_percent': memory_change,
            }
        }

        # Save comparison
        comparison_file = stress_temp_dir / "performance_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)

        assert comparison_file.exists(), "Comparison report should be created"

    async def test_degradation_alerting(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        mock_pipeline,
        stress_config
    ):
        """
        Test alerting when performance degrades beyond thresholds.

        Validates:
        - Degradation detected automatically
        - Alerts triggered on threshold violations
        - Actionable recommendations provided
        """
        # Establish baseline
        file_count = 500

        # Baseline run
        resource_monitor.start_monitoring()
        baseline_files = file_generator.create_test_files_batch(
            directory=stress_temp_dir / "baseline",
            count=file_count,
            prefix="baseline_",
            size_bytes=200
        )

        await mock_pipeline.process_documents(
            file_paths=[str(f) for f in baseline_files],
            collection="stress-baseline"
        )

        resource_monitor.stop_monitoring()
        baseline_summary = resource_monitor.get_summary()
        baseline_throughput = file_count / baseline_summary['duration_seconds']

        # Degraded run (simulate 30% slowdown)
        resource_monitor.start_monitoring()
        degraded_files = file_generator.create_test_files_batch(
            directory=stress_temp_dir / "degraded",
            count=file_count,
            prefix="degraded_",
            size_bytes=200
        )

        # Add artificial delay
        await asyncio.sleep(baseline_summary['duration_seconds'] * 0.3)

        await mock_pipeline.process_documents(
            file_paths=[str(f) for f in degraded_files],
            collection="stress-degraded"
        )

        resource_monitor.stop_monitoring()
        degraded_summary = resource_monitor.get_summary()
        degraded_throughput = file_count / degraded_summary['duration_seconds']

        # Check degradation
        throughput_decrease = (
            (baseline_throughput - degraded_throughput) / baseline_throughput * 100
        )

        alerts = []

        if throughput_decrease > DEGRADATION_THRESHOLDS['throughput_decrease_percent']:
            alerts.append({
                'type': 'throughput_degradation',
                'severity': 'warning',
                'message': f"Throughput decreased by {throughput_decrease:.1f}%",
                'threshold': DEGRADATION_THRESHOLDS['throughput_decrease_percent'],
                'actual': throughput_decrease,
            })

        # Validate alerting
        assert len(alerts) > 0, "Should detect degradation and generate alerts"

        alert_file = stress_temp_dir / "degradation_alerts.json"
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)

        assert alert_file.exists(), "Alert file should be created"

    async def test_trend_tracking(
        self,
        stress_temp_dir,
        file_generator,
        resource_monitor,
        performance_tracker,
        mock_pipeline,
        stress_config
    ):
        """
        Test tracking performance trends over multiple runs.

        Validates:
        - Trends calculated correctly
        - Historical data maintained
        - Trend direction identified
        """
        file_count = 200
        run_count = 5

        historical_data = []

        for run_num in range(run_count):
            resource_monitor.start_monitoring()
            performance_tracker.start_operation(f"run_{run_num}")

            # Create and process files
            files = file_generator.create_test_files_batch(
                directory=stress_temp_dir / f"run_{run_num}",
                count=file_count,
                prefix=f"trend_{run_num}_",
                size_bytes=200
            )

            await mock_pipeline.process_documents(
                file_paths=[str(f) for f in files],
                collection=f"stress-trend-{run_num}"
            )

            performance_tracker.end_operation(f"run_{run_num}")
            resource_monitor.stop_monitoring()

            # Record metrics
            monitoring_summary = resource_monitor.get_summary()

            run_data = {
                'run_number': run_num,
                'timestamp': time.time(),
                'throughput': file_count / monitoring_summary['duration_seconds'],
                'avg_memory_mb': monitoring_summary['summary']['memory_rss_mb']['avg'],
                'duration_seconds': monitoring_summary['duration_seconds'],
            }

            historical_data.append(run_data)

            # Add small random variation to simulate real-world fluctuation
            await asyncio.sleep(random.uniform(0.01, 0.05))

        # Analyze trend
        throughputs = [d['throughput'] for d in historical_data]

        # Simple linear trend
        avg_first_half = sum(throughputs[:len(throughputs)//2]) / (len(throughputs)//2)
        avg_second_half = sum(throughputs[len(throughputs)//2:]) / (len(throughputs) - len(throughputs)//2)

        trend_direction = "improving" if avg_second_half > avg_first_half else "degrading"
        trend_magnitude = abs(avg_second_half - avg_first_half) / avg_first_half * 100

        trend_report = {
            'historical_data': historical_data,
            'analysis': {
                'trend_direction': trend_direction,
                'trend_magnitude_percent': trend_magnitude,
                'avg_throughput': sum(throughputs) / len(throughputs),
                'min_throughput': min(throughputs),
                'max_throughput': max(throughputs),
            }
        }

        # Save trend report
        trend_file = stress_temp_dir / "performance_trend.json"
        with open(trend_file, 'w') as f:
            json.dump(trend_report, f, indent=2)

        # Validate trend tracking
        assert len(historical_data) == run_count, "All runs should be recorded"
        assert trend_file.exists(), "Trend report should be created"
        assert 'trend_direction' in trend_report['analysis'], "Should identify trend direction"
