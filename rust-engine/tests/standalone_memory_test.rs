//! Standalone tests for async memory management module
//!
//! This file tests the async memory management module in isolation,
//! avoiding dependencies on other modules that may have compilation issues.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

// Import only the memory module types
use workspace_qdrant_daemon::memory::{
    AsyncMemoryPool, AsyncSharedCache, AsyncMappedBuffer, MemoryConfig
};

#[tokio::test]
async fn test_async_memory_pool_basic_operations() {
    let config = MemoryConfig::default();
    let pool = AsyncMemoryPool::new(config).await.unwrap();

    // Test allocation
    let block = pool.allocate(1024).await.unwrap();
    assert!(block.size >= 1024);
    assert_eq!(block.data.len(), block.size);

    // Verify stats updated
    assert!(pool.stats().current_usage() > 0);
    assert_eq!(pool.stats().allocation_count.load(Ordering::Relaxed), 1);

    // Test deallocation
    pool.deallocate(block).await.unwrap();
    assert_eq!(pool.stats().free_count.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn test_async_memory_pool_concurrent_access() {
    let config = MemoryConfig::default();
    let pool = Arc::new(AsyncMemoryPool::new(config).await.unwrap());

    let mut handles = vec![];

    // Spawn 10 concurrent allocation/deallocation tasks
    for i in 0..10 {
        let pool_clone = Arc::clone(&pool);
        let handle = tokio::spawn(async move {
            let size = 1024 + (i * 256);
            let block = pool_clone.allocate(size).await.unwrap();
            sleep(Duration::from_millis(10)).await;
            pool_clone.deallocate(block).await.unwrap();
            i
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all operations completed
    let stats = pool.stats();
    assert_eq!(stats.allocation_count.load(Ordering::Relaxed), 10);
    assert_eq!(stats.free_count.load(Ordering::Relaxed), 10);
}

#[tokio::test]
async fn test_memory_pool_pooling_efficiency() {
    let config = MemoryConfig {
        initial_capacity: 4,
        ..Default::default()
    };
    let pool = AsyncMemoryPool::new(config).await.unwrap();

    // Allocate and deallocate to populate pool
    let block1 = pool.allocate(1024).await.unwrap();
    pool.deallocate(block1).await.unwrap();

    // Second allocation should hit pool
    let _block2 = pool.allocate(1024).await.unwrap();

    // Verify pool hits occurred
    assert!(pool.stats().pool_hits.load(Ordering::Relaxed) > 0);
}

#[tokio::test]
async fn test_memory_pressure_and_gc() {
    let config = MemoryConfig {
        max_pool_size: 1024 * 1024, // 1MB
        pressure_threshold: 0.7,
        ..Default::default()
    };
    let pool = AsyncMemoryPool::new(config).await.unwrap();

    // Simulate memory usage
    pool.stats().current_usage.store(800 * 1024, Ordering::Relaxed); // 800KB

    let pressure = pool.memory_pressure().await;
    assert!((pressure - 0.78125).abs() < 0.01); // 800/1024 ≈ 0.78125

    // Test garbage collection
    let freed = pool.force_gc().await.unwrap();
    assert!(freed >= 0);
}

#[tokio::test]
async fn test_async_shared_cache_basic_operations() {
    let cache: AsyncSharedCache<String, i32> = AsyncSharedCache::new(10, Duration::from_secs(60));

    // Test insertion and retrieval
    cache.insert("key1".to_string(), 42).await;
    assert_eq!(cache.get(&"key1".to_string()).await, Some(42));

    // Test non-existent key
    assert_eq!(cache.get(&"nonexistent".to_string()).await, None);

    // Verify stats
    let stats = cache.stats();
    assert_eq!(stats.hits.load(Ordering::Relaxed), 1);
    assert_eq!(stats.misses.load(Ordering::Relaxed), 1);
    assert_eq!(stats.inserts.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn test_cache_eviction_policy() {
    let cache: AsyncSharedCache<i32, String> = AsyncSharedCache::new(3, Duration::from_secs(60));

    // Fill cache to capacity
    cache.insert(1, "value1".to_string()).await;
    cache.insert(2, "value2".to_string()).await;
    cache.insert(3, "value3".to_string()).await;

    // This should trigger eviction
    cache.insert(4, "value4".to_string()).await;

    let stats = cache.stats();
    assert!(stats.evictions.load(Ordering::Relaxed) > 0);
    assert_eq!(stats.inserts.load(Ordering::Relaxed), 4);
}

#[tokio::test]
async fn test_cache_ttl_expiration() {
    let cache: AsyncSharedCache<String, i32> = AsyncSharedCache::new(10, Duration::from_millis(100));

    cache.insert("key".to_string(), 42).await;
    assert_eq!(cache.get(&"key".to_string()).await, Some(42));

    // Wait for TTL expiration
    sleep(Duration::from_millis(150)).await;
    assert_eq!(cache.get(&"key".to_string()).await, None);
}

#[tokio::test]
async fn test_cache_remove_and_clear() {
    let cache: AsyncSharedCache<String, i32> = AsyncSharedCache::new(10, Duration::from_secs(60));

    cache.insert("key1".to_string(), 1).await;
    cache.insert("key2".to_string(), 2).await;

    // Test removal
    assert_eq!(cache.remove(&"key1".to_string()).await, Some(1));
    assert_eq!(cache.get(&"key1".to_string()).await, None);

    // Test clear
    cache.clear().await;
    assert_eq!(cache.get(&"key2".to_string()).await, None);
}

#[tokio::test]
async fn test_async_mapped_buffer_read_write() {
    let buffer = AsyncMappedBuffer::new(1024, false).await.unwrap();

    // Test basic properties
    assert_eq!(buffer.size(), 1024);
    assert!(!buffer.is_readonly());

    // Test write and read
    let data = vec![1, 2, 3, 4, 5];
    buffer.write(0, &data).await.unwrap();

    let read_data = buffer.read(0, 5).await.unwrap();
    assert_eq!(read_data, data);

    // Test partial read
    let partial = buffer.read(2, 2).await.unwrap();
    assert_eq!(partial, vec![3, 4]);
}

#[tokio::test]
async fn test_mapped_buffer_bounds_checking() {
    let buffer = AsyncMappedBuffer::new(100, false).await.unwrap();

    // Test read beyond bounds
    let result = buffer.read(90, 20).await;
    assert!(result.is_err());

    // Test write beyond bounds
    let data = vec![1, 2, 3];
    let result = buffer.write(99, &data).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_readonly_buffer() {
    let buffer = AsyncMappedBuffer::new(100, true).await.unwrap();
    assert!(buffer.is_readonly());

    // Reads should work
    let data = buffer.read(0, 10).await.unwrap();
    assert_eq!(data.len(), 10);

    // Writes should fail
    let write_data = vec![0xFF; 5];
    assert!(buffer.write(0, &write_data).await.is_err());

    // Flush should still work
    assert!(buffer.flush().await.is_ok());
}

#[tokio::test]
async fn test_concurrent_buffer_access() {
    let buffer = Arc::new(AsyncMappedBuffer::new(1000, false).await.unwrap());
    let num_writers = 5;
    let mut handles = vec![];

    // Spawn concurrent writers to different regions
    for i in 0..num_writers {
        let buffer_clone = Arc::clone(&buffer);
        let handle = tokio::spawn(async move {
            let offset = i * 200; // Non-overlapping regions
            let data = vec![i as u8; 100];

            buffer_clone.write(offset, &data).await.unwrap();

            // Verify write
            let read_data = buffer_clone.read(offset, 100).await.unwrap();
            assert_eq!(read_data, data);
            i
        });
        handles.push(handle);
    }

    // Wait for all writers
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all regions independently
    for i in 0..num_writers {
        let offset = i * 200;
        let data = buffer.read(offset, 100).await.unwrap();
        assert!(data.iter().all(|&b| b == i as u8));
    }
}

#[tokio::test]
async fn test_memory_stats_accuracy() {
    let config = MemoryConfig::default();
    let pool = AsyncMemoryPool::new(config).await.unwrap();

    let initial_usage = pool.stats().current_usage();
    let initial_peak = pool.stats().peak_usage();

    // Allocate some memory
    let block1 = pool.allocate(1024).await.unwrap();
    let block2 = pool.allocate(2048).await.unwrap();

    // Check usage increased
    assert!(pool.stats().current_usage() > initial_usage);
    assert!(pool.stats().peak_usage() >= pool.stats().current_usage());

    // Deallocate
    pool.deallocate(block1).await.unwrap();
    pool.deallocate(block2).await.unwrap();

    // Usage should decrease but peak should remain
    assert!(pool.stats().current_usage() <= initial_usage);
    assert!(pool.stats().peak_usage() >= initial_peak);
}

#[tokio::test]
async fn test_cache_hit_ratio_calculation() {
    let cache: AsyncSharedCache<i32, String> = AsyncSharedCache::new(5, Duration::from_secs(60));

    // Insert some values
    for i in 0..3 {
        cache.insert(i, format!("value{}", i)).await;
    }

    // Create hits and misses
    for i in 0..3 {
        let _ = cache.get(&i).await; // Hit
    }
    for i in 10..13 {
        let _ = cache.get(&i).await; // Miss
    }

    let stats = cache.stats();
    let hits = stats.hits.load(Ordering::Relaxed);
    let misses = stats.misses.load(Ordering::Relaxed);
    let total = hits + misses;

    assert_eq!(hits, 3);
    assert_eq!(misses, 3);
    assert_eq!(total, 6);

    let hit_ratio = hits as f64 / total as f64;
    assert_eq!(hit_ratio, 0.5);
}

#[tokio::test]
async fn test_memory_allocation_size_rounding() {
    let config = MemoryConfig::default();
    let pool = AsyncMemoryPool::new(config).await.unwrap();

    // Test various sizes get rounded to appropriate pool sizes
    let test_cases = [
        (50, 64),     // Should round up to 64
        (200, 256),   // Should round up to 256
        (1000, 1024), // Should round up to 1024
        (5000, 16384), // Should round up to 16384
    ];

    for (requested, expected_min) in test_cases {
        let block = pool.allocate(requested).await.unwrap();
        assert!(block.size >= requested);
        assert!(block.size >= expected_min);
        pool.deallocate(block).await.unwrap();
    }
}

#[tokio::test]
async fn test_integrated_memory_workflow() {
    // Test a realistic workflow using all memory components
    let pool = Arc::new(AsyncMemoryPool::new(MemoryConfig::default()).await.unwrap());
    let cache: Arc<AsyncSharedCache<String, Vec<u8>>> = Arc::new(
        AsyncSharedCache::new(10, Duration::from_secs(30))
    );
    let buffer = Arc::new(AsyncMappedBuffer::new(4096, false).await.unwrap());

    // Simulate document processing workflow
    let key = "document_123".to_string();

    // 1. Allocate work memory
    let work_memory = pool.allocate(2048).await.unwrap();
    assert!(work_memory.size >= 2048);

    // 2. Check cache
    let cached_data = cache.get(&key).await;
    assert!(cached_data.is_none());

    // 3. Process data and cache result
    let processed_data = vec![0xAB; 1024];
    cache.insert(key.clone(), processed_data.clone()).await;

    // 4. Write to shared buffer
    buffer.write(0, &processed_data).await.unwrap();

    // 5. Verify cache hit on subsequent access
    let cached_result = cache.get(&key).await;
    assert_eq!(cached_result, Some(processed_data.clone()));

    // 6. Verify buffer data
    let buffer_data = buffer.read(0, processed_data.len()).await.unwrap();
    assert_eq!(buffer_data, processed_data);

    // 7. Clean up
    pool.deallocate(work_memory).await.unwrap();

    // Verify stats
    assert_eq!(pool.stats().allocation_count.load(Ordering::Relaxed), 1);
    assert_eq!(pool.stats().free_count.load(Ordering::Relaxed), 1);
    assert_eq!(cache.stats().hits.load(Ordering::Relaxed), 1);
    assert_eq!(cache.stats().misses.load(Ordering::Relaxed), 1);
}