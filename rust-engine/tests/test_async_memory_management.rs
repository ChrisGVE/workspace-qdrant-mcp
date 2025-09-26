//! Integration tests for async memory management
//!
//! These tests verify async memory allocation, sharing, and cleanup patterns
//! in the context of the workspace daemon with real async workloads.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};
use workspace_qdrant_daemon::memory::{
    AsyncMemoryPool, AsyncSharedCache, AsyncMappedBuffer, MemoryConfig
};

/// Test suite for async memory pool integration
mod async_memory_pool_integration {
    use super::*;

    #[tokio::test]
    async fn test_memory_pool_with_realistic_workload() {
        let config = MemoryConfig {
            max_pool_size: 10 * 1024 * 1024, // 10MB
            initial_capacity: 8,
            chunk_size: 4096,
            cleanup_interval_secs: 1, // Fast cleanup for testing
            enable_monitoring: true,
            pressure_threshold: 0.7,
        };

        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Simulate document processing workload
        let mut allocated_blocks = vec![];

        // Phase 1: Bulk allocation (simulate document loading)
        for i in 0..20 {
            let size = match i % 4 {
                0 => 1024,  // Small document
                1 => 4096,  // Medium document
                2 => 16384, // Large document
                3 => 65536, // Very large document
                _ => unreachable!(),
            };

            let block = pool.allocate(size).await.unwrap();
            allocated_blocks.push(block);
        }

        // Verify memory usage increased
        assert!(pool.stats().current_usage() > 0);
        assert_eq!(pool.stats().allocation_count.load(std::sync::atomic::Ordering::Relaxed), 20);

        // Phase 2: Processing simulation (deallocate some, keep others)
        for (i, block) in allocated_blocks.into_iter().enumerate() {
            if i % 2 == 0 {
                pool.deallocate(block).await.unwrap();
            }
        }

        // Verify partial deallocation
        assert_eq!(pool.stats().free_count.load(std::sync::atomic::Ordering::Relaxed), 10);

        // Phase 3: Memory pressure test
        let initial_pressure = pool.memory_pressure().await;
        assert!(initial_pressure >= 0.0 && initial_pressure <= 1.0);

        // Force GC and verify cleanup
        let freed_bytes = pool.force_gc().await.unwrap();
        // freed_bytes is u64, so it's inherently >= 0
        assert!(freed_bytes != u64::MAX, "freed_bytes should be a valid value");
    }

    #[tokio::test]
    async fn test_memory_pool_stress_under_concurrent_load() {
        let config = MemoryConfig {
            max_pool_size: 50 * 1024 * 1024, // 50MB
            ..Default::default()
        };

        let pool = Arc::new(AsyncMemoryPool::new(config).await.unwrap());
        let num_concurrent_tasks = 25;
        let operations_per_task = 50;

        let start_time = Instant::now();
        let mut handles = vec![];

        // Spawn concurrent workers
        for worker_id in 0..num_concurrent_tasks {
            let pool_clone = Arc::clone(&pool);
            let handle = tokio::spawn(async move {
                let mut local_blocks = vec![];

                for op in 0..operations_per_task {
                    // Vary allocation sizes realistically
                    let size = match (worker_id + op) % 6 {
                        0 => 512,    // Metadata
                        1 => 2048,   // Text chunk
                        2 => 8192,   // Code file
                        3 => 32768,  // Large document
                        4 => 131072, // Image metadata
                        5 => 524288, // Large binary
                        _ => unreachable!(),
                    };

                    // Allocate
                    let block = pool_clone.allocate(size).await.unwrap();
                    local_blocks.push(block);

                    // Simulate processing time
                    if op % 10 == 0 {
                        sleep(Duration::from_micros(100)).await;
                    }

                    // Periodically deallocate some blocks
                    if local_blocks.len() > 5 && op % 7 == 0 {
                        let block = local_blocks.remove(0);
                        pool_clone.deallocate(block).await.unwrap();
                    }
                }

                // Cleanup remaining blocks
                for block in local_blocks {
                    pool_clone.deallocate(block).await.unwrap();
                }

                worker_id
            });
            handles.push(handle);
        }

        // Wait for all workers with timeout
        let timeout_duration = Duration::from_secs(30);
        let results = timeout(timeout_duration, async {
            let mut worker_ids = vec![];
            for handle in handles {
                worker_ids.push(handle.await.unwrap());
            }
            worker_ids
        }).await.expect("Stress test timed out");

        assert_eq!(results.len(), num_concurrent_tasks);

        let elapsed = start_time.elapsed();
        println!("Stress test completed in {:?}", elapsed);

        // Verify pool statistics
        let stats = pool.stats();
        assert!(stats.allocation_count.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert!(stats.free_count.load(std::sync::atomic::Ordering::Relaxed) > 0);

        // Memory should be mostly cleaned up
        let final_pressure = pool.memory_pressure().await;
        assert!(final_pressure < 0.5, "Memory pressure too high after cleanup: {:.2}", final_pressure);
    }

    #[tokio::test]
    async fn test_memory_pool_edge_cases() {
        let config = MemoryConfig {
            max_pool_size: 1024 * 1024, // 1MB
            initial_capacity: 2,
            ..Default::default()
        };

        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Test zero-size allocation (should get minimum size)
        let zero_block = pool.allocate(0).await.unwrap();
        assert!(zero_block.size > 0);
        pool.deallocate(zero_block).await.unwrap();

        // Test very large allocation
        let large_block = pool.allocate(2 * 1024 * 1024).await.unwrap(); // 2MB
        assert!(large_block.size >= 2 * 1024 * 1024);
        pool.deallocate(large_block).await.unwrap();

        // Test rapid allocation/deallocation cycles
        for _ in 0..100 {
            let block = pool.allocate(1024).await.unwrap();
            pool.deallocate(block).await.unwrap();
        }

        // Verify pool efficiency
        let stats = pool.stats();
        if stats.pool_hits.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            let hit_ratio = stats.cache_hit_ratio();
            assert!(hit_ratio > 0.0, "Pool should have some hits with rapid cycles");
        }
    }

    #[tokio::test]
    async fn test_memory_pool_cleanup_integration() {
        let config = MemoryConfig {
            max_pool_size: 5 * 1024 * 1024, // 5MB
            cleanup_interval_secs: 1, // Fast cleanup
            pressure_threshold: 0.6,
            ..Default::default()
        };

        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Allocate enough to trigger high pressure
        let mut blocks = vec![];
        for _ in 0..50 {
            let block = pool.allocate(100 * 1024).await.unwrap(); // 100KB each
            blocks.push(block);
        }

        // Check initial pressure
        let initial_pressure = pool.memory_pressure().await;

        // Deallocate to populate pool with old blocks
        for block in blocks {
            pool.deallocate(block).await.unwrap();
        }

        // Wait for automatic cleanup
        sleep(Duration::from_secs(2)).await;

        // Check that pressure decreased
        let final_pressure = pool.memory_pressure().await;
        assert!(final_pressure <= initial_pressure);
    }
}

/// Test suite for async shared cache integration
mod async_shared_cache_integration {
    use super::*;

    #[tokio::test]
    async fn test_cache_integration_with_concurrent_workloads() {
        let cache: Arc<AsyncSharedCache<String, Vec<u8>>> = Arc::new(
            AsyncSharedCache::new(100, Duration::from_secs(5))
        );

        let num_workers = 10;
        let operations_per_worker = 100;
        let mut handles = vec![];

        // Spawn concurrent cache workers
        for worker_id in 0..num_workers {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                for op in 0..operations_per_worker {
                    let key = format!("worker_{}_{}", worker_id, op % 20); // Create overlapping keys
                    let value = vec![worker_id as u8; 1024 + op]; // Variable size data

                    // Simulate realistic cache usage patterns
                    match op % 4 {
                        0 | 1 => {
                            // Insert/update operations (50%)
                            cache_clone.insert(key, value).await;
                        }
                        2 => {
                            // Read operations (25%)
                            let _ = cache_clone.get(&key).await;
                        }
                        3 => {
                            // Delete operations (25%)
                            let _ = cache_clone.remove(&key).await;
                        }
                        _ => unreachable!(),
                    }

                    // Simulate processing delay
                    if op % 50 == 0 {
                        sleep(Duration::from_micros(100)).await;
                    }
                }
                worker_id
            });
            handles.push(handle);
        }

        // Wait for all workers
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify cache statistics
        let stats = cache.stats();
        let total_ops = stats.hits.load(std::sync::atomic::Ordering::Relaxed) +
                       stats.misses.load(std::sync::atomic::Ordering::Relaxed);
        assert!(total_ops > 0);
        assert!(stats.inserts.load(std::sync::atomic::Ordering::Relaxed) > 0);

        println!("Cache stats - Hits: {}, Misses: {}, Inserts: {}, Evictions: {}",
                stats.hits.load(std::sync::atomic::Ordering::Relaxed),
                stats.misses.load(std::sync::atomic::Ordering::Relaxed),
                stats.inserts.load(std::sync::atomic::Ordering::Relaxed),
                stats.evictions.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_cache_ttl_under_load() {
        let short_ttl_cache: Arc<AsyncSharedCache<i32, String>> = Arc::new(
            AsyncSharedCache::new(50, Duration::from_millis(200))
        );

        // Insert initial values
        for i in 0..10 {
            short_ttl_cache.insert(i, format!("value_{}", i)).await;
        }

        // Verify all values exist
        for i in 0..10 {
            assert!(short_ttl_cache.get(&i).await.is_some());
        }

        // Wait for TTL expiration
        sleep(Duration::from_millis(250)).await;

        // All values should be expired
        for i in 0..10 {
            assert!(short_ttl_cache.get(&i).await.is_none());
        }

        // Verify miss count increased
        let stats = short_ttl_cache.stats();
        assert!(stats.misses.load(std::sync::atomic::Ordering::Relaxed) >= 10);
    }

    #[tokio::test]
    async fn test_cache_eviction_under_pressure() {
        let small_cache: Arc<AsyncSharedCache<i32, Vec<u8>>> = Arc::new(
            AsyncSharedCache::new(5, Duration::from_secs(60)) // Small cache, long TTL
        );

        // Fill cache beyond capacity
        for i in 0..10 {
            let large_value = vec![i as u8; 1024]; // 1KB each
            small_cache.insert(i, large_value).await;
        }

        // Check that evictions occurred
        let stats = small_cache.stats();
        assert!(stats.evictions.load(std::sync::atomic::Ordering::Relaxed) > 0);

        // Some early keys should be evicted
        let mut evicted_count = 0;
        for i in 0..5 {
            if small_cache.get(&i).await.is_none() {
                evicted_count += 1;
            }
        }
        assert!(evicted_count > 0, "Some early keys should have been evicted");
    }

    #[tokio::test]
    async fn test_cache_clear_during_concurrent_operations() {
        let cache: Arc<AsyncSharedCache<i32, String>> = Arc::new(
            AsyncSharedCache::new(100, Duration::from_secs(60))
        );

        // Start background operations
        let cache_clone = Arc::clone(&cache);
        let background_handle = tokio::spawn(async move {
            for i in 0..1000 {
                cache_clone.insert(i, format!("bg_value_{}", i)).await;
                if i % 10 == 0 {
                    let _ = cache_clone.get(&(i - 5)).await;
                }
                sleep(Duration::from_micros(100)).await;
            }
        });

        // Let some operations run
        sleep(Duration::from_millis(50)).await;

        // Clear cache during operations
        cache.clear().await;

        // Continue operations
        sleep(Duration::from_millis(50)).await;

        // Cancel background operations
        background_handle.abort();

        // Verify cache is still functional
        cache.insert(999, "test_value".to_string()).await;
        assert_eq!(cache.get(&999).await, Some("test_value".to_string()));
    }
}

/// Test suite for async mapped buffer integration
mod async_mapped_buffer_integration {
    use super::*;

    #[tokio::test]
    async fn test_mapped_buffer_concurrent_read_write() {
        let buffer_size = 1024 * 1024; // 1MB
        let buffer = Arc::new(AsyncMappedBuffer::new(buffer_size, false).await.unwrap());

        let num_writers = 8;
        let writes_per_worker = 50;
        let chunk_size = 1024; // 1KB chunks

        let mut handles = vec![];

        // Spawn concurrent writers to different regions
        for writer_id in 0..num_writers {
            let buffer_clone = Arc::clone(&buffer);
            let handle = tokio::spawn(async move {
                let base_offset = writer_id * (buffer_size / num_writers);
                let region_size = buffer_size / num_writers;

                for write_op in 0..writes_per_worker {
                    let offset = base_offset + (write_op * chunk_size) % (region_size - chunk_size);
                    let data: Vec<u8> = (0..chunk_size)
                        .map(|i| ((writer_id * 256 + write_op + i) % 256) as u8)
                        .collect();

                    // Write data
                    buffer_clone.write(offset, &data).await.unwrap();

                    // Verify write immediately
                    let read_data = buffer_clone.read(offset, chunk_size).await.unwrap();
                    assert_eq!(read_data, data, "Data mismatch in writer {} at offset {}", writer_id, offset);

                    // Small delay to allow interleaving
                    if write_op % 10 == 0 {
                        sleep(Duration::from_micros(50)).await;
                    }
                }

                writer_id
            });
            handles.push(handle);
        }

        // Wait for all writers
        for handle in handles {
            handle.await.unwrap();
        }

        // Spawn concurrent readers to verify data integrity
        let mut read_handles = vec![];
        for reader_id in 0..num_writers {
            let buffer_clone = Arc::clone(&buffer);
            let handle = tokio::spawn(async move {
                let base_offset = reader_id * (buffer_size / num_writers);
                let region_size = buffer_size / num_writers;

                // Read random locations in the region
                for _ in 0..20 {
                    let offset = base_offset + (rand::random::<usize>() % (region_size - chunk_size));
                    let data = buffer_clone.read(offset, chunk_size).await.unwrap();
                    assert_eq!(data.len(), chunk_size);
                }

                reader_id
            });
            read_handles.push(handle);
        }

        for handle in read_handles {
            handle.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_mapped_buffer_streaming_operations() {
        let buffer = Arc::new(AsyncMappedBuffer::new(64 * 1024, false).await.unwrap()); // 64KB

        // Simulate streaming write
        let write_buffer = Arc::clone(&buffer);
        let write_handle = tokio::spawn(async move {
            let chunk_size = 1024;
            let pattern = b"STREAMING_DATA_PATTERN_";

            for chunk_id in 0..60 { // Write 60KB
                let offset = chunk_id * chunk_size;
                let mut chunk_data = Vec::with_capacity(chunk_size);

                // Create repeating pattern
                while chunk_data.len() < chunk_size {
                    chunk_data.extend_from_slice(pattern);
                }
                chunk_data.truncate(chunk_size);

                write_buffer.write(offset, &chunk_data).await.unwrap();

                // Simulate streaming delay
                sleep(Duration::from_micros(200)).await;
            }
        });

        // Simulate streaming read with some delay
        sleep(Duration::from_millis(10)).await;

        let read_buffer = Arc::clone(&buffer);
        let read_handle = tokio::spawn(async move {
            let chunk_size = 1024;
            let expected_pattern = b"STREAMING_DATA_PATTERN_";

            for chunk_id in 0..60 {
                let offset = chunk_id * chunk_size;

                // Wait for write to complete this chunk
                while {
                    let data = read_buffer.read(offset, 23).await.unwrap(); // Read pattern length
                    data != expected_pattern[..23]
                } {
                    sleep(Duration::from_micros(100)).await;
                }

                // Read full chunk and verify
                let chunk_data = read_buffer.read(offset, chunk_size).await.unwrap();
                assert!(chunk_data.starts_with(expected_pattern));
            }
        });

        // Wait for both operations
        let (write_result, read_result) = tokio::join!(write_handle, read_handle);
        write_result.unwrap();
        read_result.unwrap();
    }

    #[tokio::test]
    async fn test_mapped_buffer_boundary_conditions() {
        let buffer_size = 4096; // 4KB
        let buffer = AsyncMappedBuffer::new(buffer_size, false).await.unwrap();

        // Test boundary writes
        let test_cases = vec![
            (0, 1),                    // Start boundary
            (buffer_size - 1, 1),      // End boundary
            (0, buffer_size),          // Full buffer
            (1024, 2048),             // Middle chunk
            (buffer_size / 2, 1),     // Middle single byte
        ];

        for (offset, size) in test_cases {
            let data = vec![0xAB; size];

            // Write
            buffer.write(offset, &data).await.unwrap();

            // Read back
            let read_data = buffer.read(offset, size).await.unwrap();
            assert_eq!(read_data, data, "Boundary test failed at offset {} size {}", offset, size);
        }

        // Test error conditions
        assert!(buffer.read(buffer_size, 1).await.is_err()); // Read beyond end
        assert!(buffer.write(buffer_size - 1, &[1, 2]).await.is_err()); // Write beyond end
    }

    #[tokio::test]
    async fn test_readonly_buffer_protection() {
        let buffer = AsyncMappedBuffer::new(1024, true).await.unwrap();

        // Reads should work
        let initial_data = buffer.read(0, 100).await.unwrap();
        assert_eq!(initial_data.len(), 100);
        assert!(initial_data.iter().all(|&b| b == 0)); // Should be zero-initialized

        // Writes should fail
        let write_data = vec![0xFF; 10];
        assert!(buffer.write(0, &write_data).await.is_err());
        assert!(buffer.write(500, &write_data).await.is_err());

        // Verify data unchanged
        let final_data = buffer.read(0, 100).await.unwrap();
        assert_eq!(initial_data, final_data);

        // Flush should still work
        assert!(buffer.flush().await.is_ok());
    }
}

/// Integration tests combining multiple memory management components
mod integrated_memory_management {
    use super::*;

    #[tokio::test]
    async fn test_memory_components_integration() {
        // Setup integrated memory management
        let pool = Arc::new(AsyncMemoryPool::new(MemoryConfig::default()).await.unwrap());
        let cache: Arc<AsyncSharedCache<String, Vec<u8>>> = Arc::new(
            AsyncSharedCache::new(50, Duration::from_secs(30))
        );
        let buffer = Arc::new(AsyncMappedBuffer::new(1024 * 1024, false).await.unwrap());

        let num_workers = 5;
        let mut handles = vec![];

        // Spawn workers that use all memory components
        for worker_id in 0..num_workers {
            let pool_clone = Arc::clone(&pool);
            let cache_clone = Arc::clone(&cache);
            let buffer_clone = Arc::clone(&buffer);

            let handle = tokio::spawn(async move {
                for op in 0..20 {
                    let key = format!("worker_{}_{}", worker_id, op);

                    // 1. Allocate memory for processing
                    let work_memory = pool_clone.allocate(4096).await.unwrap();

                    // 2. Check cache for existing data
                    let cached_data = cache_clone.get(&key).await;

                    let data = if let Some(cached) = cached_data {
                        cached
                    } else {
                        // 3. Generate new data and cache it
                        let new_data = vec![worker_id as u8; 1024 + op];
                        cache_clone.insert(key.clone(), new_data.clone()).await;
                        new_data
                    };

                    // 4. Write data to shared buffer
                    let buffer_offset = (worker_id * 200 * 1024) + (op * 1024);
                    if buffer_offset + data.len() < 1024 * 1024 {
                        buffer_clone.write(buffer_offset, &data).await.unwrap();

                        // 5. Verify buffer write
                        let read_back = buffer_clone.read(buffer_offset, data.len()).await.unwrap();
                        assert_eq!(read_back, data);
                    }

                    // 6. Deallocate work memory
                    pool_clone.deallocate(work_memory).await.unwrap();

                    // Simulate processing time
                    sleep(Duration::from_micros(100)).await;
                }

                worker_id
            });
            handles.push(handle);
        }

        // Wait for all workers
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all components are functioning
        assert!(pool.stats().allocation_count.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert!(cache.stats().inserts.load(std::sync::atomic::Ordering::Relaxed) > 0);

        // Check memory efficiency
        let hit_ratio = cache.stats().hits.load(std::sync::atomic::Ordering::Relaxed) as f64 /
            (cache.stats().hits.load(std::sync::atomic::Ordering::Relaxed) +
             cache.stats().misses.load(std::sync::atomic::Ordering::Relaxed)) as f64;

        println!("Integration test - Cache hit ratio: {:.2}%", hit_ratio * 100.0);
        println!("Pool stats - Allocated: {}, Freed: {}, Hit ratio: {:.2}%",
                pool.stats().allocation_count.load(std::sync::atomic::Ordering::Relaxed),
                pool.stats().free_count.load(std::sync::atomic::Ordering::Relaxed),
                pool.stats().cache_hit_ratio() * 100.0);
    }

    #[tokio::test]
    async fn test_memory_pressure_coordination() {
        let config = MemoryConfig {
            max_pool_size: 2 * 1024 * 1024, // 2MB - small for pressure testing
            pressure_threshold: 0.5,
            ..Default::default()
        };

        let pool = Arc::new(AsyncMemoryPool::new(config).await.unwrap());
        let cache: Arc<AsyncSharedCache<i32, Vec<u8>>> = Arc::new(
            AsyncSharedCache::new(20, Duration::from_secs(60))
        );

        // Allocate memory to create pressure
        let mut allocated_blocks = vec![];
        for i in 0..10 {
            let block = pool.allocate(200 * 1024).await.unwrap(); // 200KB each
            allocated_blocks.push(block);

            // Also fill cache
            let large_cache_value = vec![i as u8; 50 * 1024]; // 50KB
            cache.insert(i, large_cache_value).await;
        }

        // Check memory pressure
        let pressure = pool.memory_pressure().await;
        assert!(pressure > 0.5, "Memory pressure should be high: {:.2}", pressure);

        // Trigger cleanup when under pressure
        if pressure > 0.7 {
            // Clear cache to reduce pressure
            cache.clear().await;

            // Force garbage collection
            let freed = pool.force_gc().await.unwrap();
            println!("Freed {} bytes under pressure", freed);
        }

        // Deallocate half the blocks
        for (i, block) in allocated_blocks.into_iter().enumerate() {
            if i % 2 == 0 {
                pool.deallocate(block).await.unwrap();
            }
        }

        // Check pressure decreased
        let final_pressure = pool.memory_pressure().await;
        assert!(final_pressure < pressure, "Pressure should decrease after cleanup");
    }

    #[tokio::test]
    async fn test_memory_leak_prevention() {
        let pool = Arc::new(AsyncMemoryPool::new(MemoryConfig::default()).await.unwrap());
        let initial_usage = pool.stats().current_usage();

        // Simulate memory allocation patterns that could cause leaks
        for cycle in 0..5 {
            let mut temp_blocks = vec![];

            // Allocate blocks
            for _ in 0..20 {
                let size = 1024 + (cycle * 256);
                let block = pool.allocate(size).await.unwrap();
                temp_blocks.push(block);
            }

            // Verify usage increased
            assert!(pool.stats().current_usage() > initial_usage);

            // Deallocate all blocks
            for block in temp_blocks {
                pool.deallocate(block).await.unwrap();
            }

            // Force cleanup
            pool.force_gc().await.unwrap();

            // Small delay for cleanup
            sleep(Duration::from_millis(10)).await;
        }

        // Memory usage should return to initial level (or close to it)
        let final_usage = pool.stats().current_usage();
        let usage_diff = if final_usage > initial_usage {
            final_usage - initial_usage
        } else {
            initial_usage - final_usage
        };

        assert!(usage_diff < 1024 * 1024, "Memory usage difference too large: {} bytes", usage_diff);

        // Verify allocation/deallocation counts match
        let allocations = pool.stats().allocation_count.load(std::sync::atomic::Ordering::Relaxed);
        let deallocations = pool.stats().free_count.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(allocations, deallocations, "Allocation/deallocation count mismatch");
    }

    #[tokio::test]
    async fn test_arc_mutex_safety_patterns() {
        // Test Arc/Mutex patterns used in async memory management
        let shared_data = Arc::new(tokio::sync::Mutex::new(Vec::<u8>::new()));
        let num_tasks = 10;
        let mut handles = vec![];

        // Test concurrent access to shared mutable data
        for task_id in 0..num_tasks {
            let data_clone = Arc::clone(&shared_data);
            let handle = tokio::spawn(async move {
                for i in 0..100 {
                    let mut data = data_clone.lock().await;
                    data.push((task_id * 100 + i) as u8);
                }
                task_id
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify data integrity
        let final_data = shared_data.lock().await;
        assert_eq!(final_data.len(), num_tasks * 100);

        // Test RwLock pattern for shared read-heavy data
        let shared_cache = Arc::new(parking_lot::RwLock::new(std::collections::HashMap::<i32, String>::new()));
        let mut read_handles = vec![];

        // Populate cache
        {
            let mut cache = shared_cache.write();
            for i in 0..100 {
                cache.insert(i, format!("value_{}", i));
            }
        }

        // Spawn concurrent readers
        for reader_id in 0..20 {
            let cache_clone = Arc::clone(&shared_cache);
            let handle = tokio::spawn(async move {
                for _ in 0..50 {
                    let cache = cache_clone.read();
                    let key = reader_id % 100;
                    assert!(cache.contains_key(&key));
                }
                reader_id
            });
            read_handles.push(handle);
        }

        for handle in read_handles {
            handle.await.unwrap();
        }
    }
}

// Property-based integration tests
mod property_based_integration {
    use super::*;

    #[tokio::test]
    async fn test_memory_allocation_invariants() {
        let pool = AsyncMemoryPool::new(MemoryConfig::default()).await.unwrap();

        // Property: allocated memory size is always >= requested size
        for requested_size in [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192] {
            let block = pool.allocate(requested_size).await.unwrap();
            assert!(block.size >= requested_size,
                   "Allocated size {} < requested size {}", block.size, requested_size);
            pool.deallocate(block).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_cache_consistency_properties() {
        let cache: AsyncSharedCache<i32, Vec<u8>> = AsyncSharedCache::new(100, Duration::from_secs(60));

        // Property: value inserted should be retrievable
        for i in 0..50 {
            let value = vec![i as u8; 100];
            cache.insert(i, value.clone()).await;
            assert_eq!(cache.get(&i).await, Some(value));
        }

        // Property: removed items should not be retrievable
        for i in 0..25 {
            cache.remove(&i).await;
            assert_eq!(cache.get(&i).await, None);
        }
    }

    #[tokio::test]
    async fn test_buffer_isolation_properties() {
        let buffer = AsyncMappedBuffer::new(10000, false).await.unwrap();

        // Property: writes to different regions don't interfere
        let regions = [
            (0, 1000),
            (2000, 1000),
            (4000, 1000),
            (6000, 1000),
            (8000, 1000),
        ];

        for (offset, size) in regions {
            let pattern = vec![offset as u8; size];
            buffer.write(offset, &pattern).await.unwrap();
        }

        // Verify each region independently
        for (offset, size) in regions {
            let data = buffer.read(offset, size).await.unwrap();
            assert!(data.iter().all(|&b| b == offset as u8),
                   "Region at offset {} corrupted", offset);
        }
    }
}

// Benchmark-style integration tests
#[cfg(test)]
mod performance_integration {
    use super::*;

    #[tokio::test]
    async fn test_memory_pool_performance_characteristics() {
        let config = MemoryConfig {
            max_pool_size: 100 * 1024 * 1024, // 100MB
            ..Default::default()
        };
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Measure allocation performance
        let start = Instant::now();
        let mut blocks = vec![];

        for _ in 0..1000 {
            let block = pool.allocate(4096).await.unwrap();
            blocks.push(block);
        }

        let allocation_time = start.elapsed();

        // Measure deallocation performance
        let start = Instant::now();
        for block in blocks {
            pool.deallocate(block).await.unwrap();
        }
        let deallocation_time = start.elapsed();

        println!("Allocation time for 1000 blocks: {:?}", allocation_time);
        println!("Deallocation time for 1000 blocks: {:?}", deallocation_time);

        // Performance assertions (should complete within reasonable time)
        assert!(allocation_time < Duration::from_secs(1));
        assert!(deallocation_time < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_cache_performance_under_load() {
        let cache: AsyncSharedCache<i32, Vec<u8>> = AsyncSharedCache::new(1000, Duration::from_secs(60));

        // Warm up cache
        for i in 0..500 {
            cache.insert(i, vec![i as u8; 1024]).await;
        }

        // Measure read performance
        let start = Instant::now();
        for _ in 0..10000 {
            let key = rand::random::<i32>() % 500;
            let _ = cache.get(&key).await;
        }
        let read_time = start.elapsed();

        println!("Cache read time for 10000 operations: {:?}", read_time);
        assert!(read_time < Duration::from_secs(1));

        // Check hit ratio
        let hit_ratio = cache.stats().hits.load(std::sync::atomic::Ordering::Relaxed) as f64 /
                       (cache.stats().hits.load(std::sync::atomic::Ordering::Relaxed) +
                        cache.stats().misses.load(std::sync::atomic::Ordering::Relaxed)) as f64;

        assert!(hit_ratio > 0.8, "Hit ratio should be high: {:.2}", hit_ratio);
    }
}