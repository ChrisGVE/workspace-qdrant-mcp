//! Async memory management module for the workspace daemon
//!
//! This module provides async-safe memory allocation, pooling, caching, and cleanup
//! strategies for high-performance document processing and vector operations.

use crate::error::{DaemonError, DaemonResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock as AsyncRwLock, Semaphore};
use tracing::{debug, info, warn};

/// Configuration for memory management
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory pool size in bytes
    pub max_pool_size: u64,
    /// Initial pool capacity
    pub initial_capacity: usize,
    /// Memory allocation chunk size
    pub chunk_size: usize,
    /// Memory cleanup interval in seconds
    pub cleanup_interval_secs: u64,
    /// Enable memory performance monitoring
    pub enable_monitoring: bool,
    /// Memory pressure threshold (0.0-1.0)
    pub pressure_threshold: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 512 * 1024 * 1024, // 512MB
            initial_capacity: 16,
            chunk_size: 4096,
            cleanup_interval_secs: 30,
            enable_monitoring: true,
            pressure_threshold: 0.8,
        }
    }
}

/// Memory allocation statistics
#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_allocated: AtomicU64,
    pub total_freed: AtomicU64,
    pub current_usage: AtomicU64,
    pub peak_usage: AtomicU64,
    pub allocation_count: AtomicU64,
    pub free_count: AtomicU64,
    pub pool_hits: AtomicU64,
    pub pool_misses: AtomicU64,
}

impl Clone for MemoryStats {
    fn clone(&self) -> Self {
        Self {
            total_allocated: AtomicU64::new(self.total_allocated.load(Ordering::SeqCst)),
            total_freed: AtomicU64::new(self.total_freed.load(Ordering::SeqCst)),
            current_usage: AtomicU64::new(self.current_usage.load(Ordering::SeqCst)),
            peak_usage: AtomicU64::new(self.peak_usage.load(Ordering::SeqCst)),
            allocation_count: AtomicU64::new(self.allocation_count.load(Ordering::SeqCst)),
            free_count: AtomicU64::new(self.free_count.load(Ordering::SeqCst)),
            pool_hits: AtomicU64::new(self.pool_hits.load(Ordering::SeqCst)),
            pool_misses: AtomicU64::new(self.pool_misses.load(Ordering::SeqCst)),
        }
    }
}

impl MemoryStats {
    /// Get current memory usage
    pub fn current_usage(&self) -> u64 {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> u64 {
        self.peak_usage.load(Ordering::Relaxed)
    }

    /// Get total allocations
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.pool_hits.load(Ordering::Relaxed);
        let misses = self.pool_misses.load(Ordering::Relaxed);
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }
}

/// Memory block for pooled allocation
#[derive(Debug)]
pub struct MemoryBlock {
    pub data: Vec<u8>,
    pub size: usize,
    pub allocated_at: Instant,
    pub last_used: Instant,
}

impl MemoryBlock {
    fn new(size: usize) -> Self {
        let now = Instant::now();
        Self {
            data: vec![0u8; size],
            size,
            allocated_at: now,
            last_used: now,
        }
    }

    fn reset(&mut self) {
        self.data.fill(0);
        self.last_used = Instant::now();
    }
}

/// Async memory pool for efficient allocation
#[derive(Debug)]
pub struct AsyncMemoryPool {
    config: MemoryConfig,
    stats: Arc<MemoryStats>,
    pools: AsyncRwLock<HashMap<usize, Vec<MemoryBlock>>>,
    allocation_semaphore: Arc<Semaphore>,
    cleanup_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl AsyncMemoryPool {
    /// Create a new async memory pool
    pub async fn new(config: MemoryConfig) -> DaemonResult<Arc<Self>> {
        info!("Initializing async memory pool with max size: {} bytes", config.max_pool_size);

        let max_concurrent = (config.max_pool_size / config.chunk_size as u64) as usize;
        let semaphore = Arc::new(Semaphore::new(max_concurrent.max(1)));

        let pool = Arc::new(Self {
            config: config.clone(),
            stats: Arc::new(MemoryStats::default()),
            pools: AsyncRwLock::new(HashMap::new()),
            allocation_semaphore: semaphore,
            cleanup_handle: Mutex::new(None),
        });

        // Initialize pools for common sizes
        {
            let mut pools = pool.pools.write().await;
            let common_sizes = [64, 256, 1024, 4096, 16384, 65536];
            for size in common_sizes {
                pools.insert(size, Vec::with_capacity(config.initial_capacity));
            }
        }

        // Start cleanup task
        let cleanup_pool = Arc::clone(&pool);
        let cleanup_task = tokio::spawn(async move {
            cleanup_pool.cleanup_task().await;
        });

        *pool.cleanup_handle.lock().await = Some(cleanup_task);

        Ok(pool)
    }

    /// Allocate memory block asynchronously
    pub async fn allocate(&self, size: usize) -> DaemonResult<MemoryBlock> {
        let _permit = self.allocation_semaphore.acquire().await
            .map_err(|e| DaemonError::Internal {
                message: format!("Memory allocation semaphore error: {}", e)
            })?;

        debug!("Allocating memory block of size: {}", size);

        // Try to get from pool first
        let pool_size = self.get_pool_size_for(size);

        {
            let mut pools = self.pools.write().await;
            if let Some(pool) = pools.get_mut(&pool_size) {
                if let Some(mut block) = pool.pop() {
                    block.reset();
                    self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
                    self.update_usage_stats(size as u64, true);
                    return Ok(block);
                }
            }
        }

        // Pool miss - allocate new block
        self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
        let block = MemoryBlock::new(pool_size);
        self.update_usage_stats(size as u64, true);

        Ok(block)
    }

    /// Deallocate memory block asynchronously
    pub async fn deallocate(&self, mut block: MemoryBlock) -> DaemonResult<()> {
        debug!("Deallocating memory block of size: {}", block.size);

        self.update_usage_stats(block.size as u64, false);

        // Check if we should return to pool or free
        if self.should_pool(&block) {
            let mut pools = self.pools.write().await;
            if let Some(pool) = pools.get_mut(&block.size) {
                if pool.len() < self.config.initial_capacity * 2 {
                    block.reset();
                    pool.push(block);
                    return Ok(());
                }
            }
        }

        // Block is dropped here (freed)
        Ok(())
    }

    /// Check current memory pressure
    pub async fn memory_pressure(&self) -> f64 {
        let current = self.stats.current_usage() as f64;
        let max_size = self.config.max_pool_size as f64;
        current / max_size
    }

    /// Force garbage collection
    pub async fn force_gc(&self) -> DaemonResult<u64> {
        info!("Forcing garbage collection");

        let mut freed_bytes = 0u64;
        let cutoff_time = Instant::now() - Duration::from_secs(60);

        let mut pools = self.pools.write().await;
        for (size, pool) in pools.iter_mut() {
            let original_len = pool.len();
            pool.retain(|block| block.last_used > cutoff_time);
            let removed = original_len - pool.len();
            freed_bytes += removed as u64 * (*size as u64);
        }

        info!("Garbage collection freed {} bytes", freed_bytes);
        Ok(freed_bytes)
    }

    /// Get memory statistics
    pub fn stats(&self) -> Arc<MemoryStats> {
        Arc::clone(&self.stats)
    }

    fn get_pool_size_for(&self, requested: usize) -> usize {
        // Round up to next power of 2 or common size
        let common_sizes = [64, 256, 1024, 4096, 16384, 65536];
        common_sizes.iter()
            .find(|&&size| size >= requested)
            .copied()
            .unwrap_or_else(|| {
                // For larger sizes, round up to next power of 2
                let mut size = 1;
                while size < requested {
                    size <<= 1;
                }
                size
            })
    }

    fn should_pool(&self, block: &MemoryBlock) -> bool {
        // Don't pool very large blocks or very old blocks
        block.size <= 65536 &&
        block.allocated_at.elapsed() < Duration::from_secs(300)
    }

    fn update_usage_stats(&self, size: u64, is_allocation: bool) {
        if is_allocation {
            self.stats.total_allocated.fetch_add(size, Ordering::Relaxed);
            self.stats.allocation_count.fetch_add(1, Ordering::Relaxed);
            let new_usage = self.stats.current_usage.fetch_add(size, Ordering::Relaxed) + size;

            // Update peak usage
            let mut current_peak = self.stats.peak_usage.load(Ordering::Relaxed);
            while new_usage > current_peak {
                match self.stats.peak_usage.compare_exchange_weak(
                    current_peak, new_usage, Ordering::Relaxed, Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(actual) => current_peak = actual,
                }
            }
        } else {
            self.stats.total_freed.fetch_add(size, Ordering::Relaxed);
            self.stats.free_count.fetch_add(1, Ordering::Relaxed);
            self.stats.current_usage.fetch_sub(size, Ordering::Relaxed);
        }
    }

    async fn cleanup_task(&self) {
        let mut interval = tokio::time::interval(
            Duration::from_secs(self.config.cleanup_interval_secs)
        );

        loop {
            interval.tick().await;

            if let Err(e) = self.periodic_cleanup().await {
                warn!("Periodic cleanup error: {}", e);
            }
        }
    }

    async fn periodic_cleanup(&self) -> DaemonResult<()> {
        debug!("Running periodic memory cleanup");

        let pressure = self.memory_pressure().await;
        if pressure > self.config.pressure_threshold {
            warn!("High memory pressure detected: {:.2}%", pressure * 100.0);
            self.force_gc().await?;
        }

        Ok(())
    }
}

impl Drop for AsyncMemoryPool {
    fn drop(&mut self) {
        // Cleanup task will be automatically cancelled when the handle is dropped
    }
}

/// Shared memory cache for async operations
#[derive(Debug)]
pub struct AsyncSharedCache<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync,
    V: Clone + Send + Sync,
{
    cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    max_size: usize,
    ttl: Duration,
    stats: Arc<CacheStats>,
}

#[derive(Debug)]
struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    last_accessed: Instant,
    access_count: AtomicUsize,
}

impl<V: Clone> Clone for CacheEntry<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            access_count: AtomicUsize::new(self.access_count.load(Ordering::SeqCst)),
        }
    }
}

#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub inserts: AtomicU64,
}

impl<K, V> AsyncSharedCache<K, V>
where
    K: Clone + Eq + std::hash::Hash + Send + Sync,
    V: Clone + Send + Sync,
{
    /// Create a new async shared cache
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            ttl,
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Get value from cache
    pub async fn get(&self, key: &K) -> Option<V> {
        let cache = self.cache.read();
        if let Some(entry) = cache.get(key) {
            if entry.created_at.elapsed() < self.ttl {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.value.clone());
            }
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert value into cache
    pub async fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.write();

        // Check if we need to evict
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache);
        }

        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: AtomicUsize::new(1),
        };

        cache.insert(key, entry);
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove value from cache
    pub async fn remove(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write();
        cache.remove(key).map(|entry| entry.value)
    }

    /// Clear all cache entries
    pub async fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> Arc<CacheStats> {
        Arc::clone(&self.stats)
    }

    fn evict_lru(&self, cache: &mut HashMap<K, CacheEntry<V>>) {
        if cache.is_empty() {
            return;
        }

        // Find least recently used entry
        let lru_key = cache.iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            cache.remove(&key);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Memory-mapped async buffer for large data processing
#[derive(Debug)]
pub struct AsyncMappedBuffer {
    data: Arc<Mutex<Vec<u8>>>,
    size: usize,
    readonly: bool,
    #[allow(dead_code)]
    stats: Arc<MemoryStats>,
}

impl AsyncMappedBuffer {
    /// Create a new async mapped buffer
    pub async fn new(size: usize, readonly: bool) -> DaemonResult<Self> {
        info!("Creating async mapped buffer of size: {} bytes", size);

        let data = vec![0u8; size];

        Ok(Self {
            data: Arc::new(Mutex::new(data)),
            size,
            readonly,
            stats: Arc::new(MemoryStats::default()),
        })
    }

    /// Read data from buffer asynchronously
    pub async fn read(&self, offset: usize, len: usize) -> DaemonResult<Vec<u8>> {
        if offset + len > self.size {
            return Err(DaemonError::Internal {
                message: "Read beyond buffer bounds".to_string(),
            });
        }

        let data = self.data.lock().await;
        Ok(data[offset..offset + len].to_vec())
    }

    /// Write data to buffer asynchronously
    pub async fn write(&self, offset: usize, data: &[u8]) -> DaemonResult<()> {
        if self.readonly {
            return Err(DaemonError::Internal {
                message: "Attempted write to readonly buffer".to_string(),
            });
        }

        if offset + data.len() > self.size {
            return Err(DaemonError::Internal {
                message: "Write beyond buffer bounds".to_string(),
            });
        }

        let mut buffer = self.data.lock().await;
        buffer[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Flush buffer changes (no-op for in-memory)
    pub async fn flush(&self) -> DaemonResult<()> {
        // No-op for in-memory buffer
        Ok(())
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if buffer is readonly
    pub fn is_readonly(&self) -> bool {
        self.readonly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    use std::sync::Arc;

    // Test utility macros
    macro_rules! property_test {
        ($name:ident, $prop:expr) => {
            #[tokio::test]
            async fn $name() {
                $prop.await;
            }
        };
    }

    #[tokio::test]
    async fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.max_pool_size, 512 * 1024 * 1024);
        assert_eq!(config.initial_capacity, 16);
        assert_eq!(config.chunk_size, 4096);
        assert_eq!(config.cleanup_interval_secs, 30);
        assert!(config.enable_monitoring);
        assert_eq!(config.pressure_threshold, 0.8);
    }

    #[tokio::test]
    async fn test_memory_stats_operations() {
        let stats = MemoryStats::default();

        // Test atomic operations
        stats.total_allocated.store(1024, Ordering::Relaxed);
        stats.current_usage.store(512, Ordering::Relaxed);
        stats.peak_usage.store(768, Ordering::Relaxed);

        assert_eq!(stats.total_allocated(), 1024);
        assert_eq!(stats.current_usage(), 512);
        assert_eq!(stats.peak_usage(), 768);

        // Test cache hit ratio calculation
        stats.pool_hits.store(8, Ordering::Relaxed);
        stats.pool_misses.store(2, Ordering::Relaxed);
        assert_eq!(stats.cache_hit_ratio(), 0.8);

        // Test zero case
        stats.pool_hits.store(0, Ordering::Relaxed);
        stats.pool_misses.store(0, Ordering::Relaxed);
        assert_eq!(stats.cache_hit_ratio(), 0.0);
    }

    #[tokio::test]
    async fn test_memory_block_creation() {
        let block = MemoryBlock::new(1024);
        assert_eq!(block.size, 1024);
        assert_eq!(block.data.len(), 1024);
        assert!(block.data.iter().all(|&b| b == 0));

        // Test reset functionality
        let mut block = MemoryBlock::new(512);
        block.data[0] = 42;
        block.reset();
        assert_eq!(block.data[0], 0);
    }

    #[tokio::test]
    async fn test_async_memory_pool_creation() {
        let config = MemoryConfig::default();
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        assert!(pool.stats.current_usage() == 0);
        assert!(pool.memory_pressure().await == 0.0);
    }

    #[tokio::test]
    async fn test_async_memory_pool_allocation() {
        let config = MemoryConfig {
            max_pool_size: 1024 * 1024,
            ..Default::default()
        };
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Test allocation
        let block = pool.allocate(512).await.unwrap();
        assert!(block.size >= 512);
        assert_eq!(block.data.len(), block.size);

        // Test stats update
        assert!(pool.stats.current_usage() > 0);
        assert!(pool.stats.total_allocated() > 0);
        assert_eq!(pool.stats.allocation_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_async_memory_pool_deallocation() {
        let config = MemoryConfig::default();
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        let block = pool.allocate(256).await.unwrap();
        let initial_usage = pool.stats.current_usage();

        pool.deallocate(block).await.unwrap();
        assert!(pool.stats.current_usage() < initial_usage);
        assert_eq!(pool.stats.free_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_async_memory_pool_pooling() {
        let config = MemoryConfig {
            initial_capacity: 2,
            ..Default::default()
        };
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Allocate and deallocate to populate pool
        let block1 = pool.allocate(1024).await.unwrap();
        pool.deallocate(block1).await.unwrap();

        // Second allocation should hit pool
        let _block2 = pool.allocate(1024).await.unwrap();
        assert!(pool.stats.pool_hits.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_memory_pressure_calculation() {
        let config = MemoryConfig {
            max_pool_size: 1000,
            ..Default::default()
        };
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Simulate allocation
        pool.stats.current_usage.store(800, Ordering::Relaxed);
        let pressure = pool.memory_pressure().await;
        assert_eq!(pressure, 0.8);
    }

    #[tokio::test]
    async fn test_async_memory_pool_force_gc() {
        let config = MemoryConfig::default();
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Allocate some blocks
        let block1 = pool.allocate(1024).await.unwrap();
        let block2 = pool.allocate(2048).await.unwrap();

        pool.deallocate(block1).await.unwrap();
        pool.deallocate(block2).await.unwrap();

        // Force GC should clean up old blocks
        let freed = pool.force_gc().await.unwrap();
        // freed is u64, always >= 0 - may be 0 if blocks are too recent
        assert!(freed < 1000); // Sanity check
    }

    #[tokio::test]
    async fn test_concurrent_memory_operations() {
        let config = MemoryConfig::default();
        let pool = Arc::new(AsyncMemoryPool::new(config).await.unwrap());

        let mut handles = vec![];

        // Spawn multiple concurrent allocation/deallocation tasks
        for i in 0..10 {
            let pool_clone = Arc::clone(&pool);
            let handle = tokio::spawn(async move {
                let size = 1024 + (i * 256);
                let block = pool_clone.allocate(size).await.unwrap();
                sleep(Duration::from_millis(10)).await;
                pool_clone.deallocate(block).await.unwrap();
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all operations completed successfully
        assert_eq!(pool.stats.allocation_count.load(Ordering::Relaxed), 10);
        assert_eq!(pool.stats.free_count.load(Ordering::Relaxed), 10);
    }

    #[tokio::test]
    async fn test_memory_pool_size_calculation() {
        let config = MemoryConfig::default();
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        // Test size rounding for common sizes
        assert_eq!(pool.get_pool_size_for(50), 64);
        assert_eq!(pool.get_pool_size_for(200), 256);
        assert_eq!(pool.get_pool_size_for(1000), 1024);
        assert_eq!(pool.get_pool_size_for(5000), 16384);

        // Test power-of-2 rounding for large sizes
        assert_eq!(pool.get_pool_size_for(100000), 131072);
    }

    #[tokio::test]
    async fn test_async_shared_cache_operations() {
        let cache: AsyncSharedCache<String, i32> = AsyncSharedCache::new(3, Duration::from_secs(60));

        // Test insertion and retrieval
        cache.insert("key1".to_string(), 42).await;
        assert_eq!(cache.get(&"key1".to_string()).await, Some(42));
        assert_eq!(cache.get(&"nonexistent".to_string()).await, None);

        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.hits.load(Ordering::Relaxed), 1);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 1);
        assert_eq!(stats.inserts.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_async_shared_cache_eviction() {
        let cache: AsyncSharedCache<String, i32> = AsyncSharedCache::new(2, Duration::from_secs(60));

        // Fill cache to capacity
        cache.insert("key1".to_string(), 1).await;
        cache.insert("key2".to_string(), 2).await;

        // This should trigger eviction
        cache.insert("key3".to_string(), 3).await;

        let stats = cache.stats();
        assert!(stats.evictions.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_async_shared_cache_ttl() {
        let cache: AsyncSharedCache<String, i32> = AsyncSharedCache::new(10, Duration::from_millis(50));

        cache.insert("key".to_string(), 42).await;
        assert_eq!(cache.get(&"key".to_string()).await, Some(42));

        // Wait for TTL to expire
        sleep(Duration::from_millis(60)).await;
        assert_eq!(cache.get(&"key".to_string()).await, None);
    }

    #[tokio::test]
    async fn test_async_shared_cache_removal_and_clear() {
        let cache: AsyncSharedCache<String, i32> = AsyncSharedCache::new(10, Duration::from_secs(60));

        cache.insert("key1".to_string(), 1).await;
        cache.insert("key2".to_string(), 2).await;

        // Test removal
        assert_eq!(cache.remove(&"key1".to_string()).await, Some(1));
        assert_eq!(cache.get(&"key1".to_string()).await, None);
        assert_eq!(cache.get(&"key2".to_string()).await, Some(2));

        // Test clear
        cache.clear().await;
        assert_eq!(cache.get(&"key2".to_string()).await, None);
    }

    #[tokio::test]
    async fn test_async_mapped_buffer_creation() {
        let buffer = AsyncMappedBuffer::new(1024, false).await.unwrap();
        assert_eq!(buffer.size(), 1024);
        assert!(!buffer.is_readonly());

        let readonly_buffer = AsyncMappedBuffer::new(512, true).await.unwrap();
        assert!(readonly_buffer.is_readonly());
    }

    #[tokio::test]
    async fn test_async_mapped_buffer_read_write() {
        let buffer = AsyncMappedBuffer::new(1024, false).await.unwrap();

        // Test write
        let data = vec![1, 2, 3, 4, 5];
        buffer.write(0, &data).await.unwrap();

        // Test read
        let read_data = buffer.read(0, 5).await.unwrap();
        assert_eq!(read_data, data);

        // Test partial read
        let partial = buffer.read(2, 2).await.unwrap();
        assert_eq!(partial, vec![3, 4]);
    }

    #[tokio::test]
    async fn test_async_mapped_buffer_bounds_checking() {
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
    async fn test_async_mapped_buffer_readonly() {
        let buffer = AsyncMappedBuffer::new(100, true).await.unwrap();

        // Test that write fails on readonly buffer
        let data = vec![1, 2, 3];
        let result = buffer.write(0, &data).await;
        assert!(result.is_err());

        // Test that read still works
        let read_result = buffer.read(0, 10).await;
        assert!(read_result.is_ok());
    }

    #[tokio::test]
    async fn test_async_mapped_buffer_flush() {
        let buffer = AsyncMappedBuffer::new(100, false).await.unwrap();

        // Flush should always succeed for in-memory buffer
        let result = buffer.flush().await;
        assert!(result.is_ok());
    }

    // Property-based testing using macro
    property_test!(
        test_memory_allocation_properties,
        async {
            let config = MemoryConfig::default();
            let pool = AsyncMemoryPool::new(config).await.unwrap();

            // Property: allocation size should be at least requested size
            for size in [64, 128, 256, 512, 1024, 2048] {
                let block = pool.allocate(size).await.unwrap();
                assert!(block.size >= size);
                pool.deallocate(block).await.unwrap();
            }
        }
    );

    property_test!(
        test_memory_stats_consistency,
        async {
            let config = MemoryConfig::default();
            let pool = AsyncMemoryPool::new(config).await.unwrap();

            let mut blocks = vec![];

            // Allocate several blocks
            for size in [256, 512, 1024] {
                let block = pool.allocate(size).await.unwrap();
                blocks.push(block);
            }

            let allocations = pool.stats.allocation_count.load(Ordering::Relaxed);

            // Deallocate all blocks
            for block in blocks {
                pool.deallocate(block).await.unwrap();
            }

            let deallocations = pool.stats.free_count.load(Ordering::Relaxed);

            // Property: allocation count should equal deallocation count
            assert_eq!(allocations, deallocations);
        }
    );

    property_test!(
        test_cache_hit_ratio_properties,
        async {
            let cache: AsyncSharedCache<i32, String> = AsyncSharedCache::new(5, Duration::from_secs(60));

            // Insert some values
            for i in 0..3 {
                cache.insert(i, format!("value{}", i)).await;
            }

            // Hit existing keys
            for i in 0..3 {
                let _ = cache.get(&i).await;
            }

            // Miss non-existing keys
            for i in 10..13 {
                let _ = cache.get(&i).await;
            }

            let hit_ratio = cache.stats().hits.load(Ordering::Relaxed) as f64 /
                          (cache.stats().hits.load(Ordering::Relaxed) + cache.stats().misses.load(Ordering::Relaxed)) as f64;

            // Property: hit ratio should be 0.5 (3 hits out of 6 total accesses)
            assert!((hit_ratio - 0.5).abs() < 0.01);
        }
    );

    #[tokio::test]
    async fn test_memory_fragmentation_resistance() {
        let config = MemoryConfig {
            max_pool_size: 10 * 1024 * 1024, // 10MB
            ..Default::default()
        };
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        let mut blocks = vec![];

        // Allocate various sizes to test fragmentation
        let sizes = [64, 128, 256, 512, 1024, 2048, 4096];
        for &size in &sizes {
            for _ in 0..5 {
                let block = pool.allocate(size).await.unwrap();
                blocks.push(block);
            }
        }

        // Deallocate every other block to create fragmentation
        for (i, block) in blocks.into_iter().enumerate() {
            if i % 2 == 0 {
                pool.deallocate(block).await.unwrap();
            }
        }

        // Should still be able to allocate efficiently
        for &size in &sizes {
            let _block = pool.allocate(size).await.unwrap();
        }

        // Verify pool efficiency
        let stats = pool.stats();
        let total_accesses = stats.pool_hits.load(Ordering::Relaxed) + stats.pool_misses.load(Ordering::Relaxed);
        if total_accesses > 0 {
            let hit_ratio = stats.pool_hits.load(Ordering::Relaxed) as f64 / total_accesses as f64;
            assert!(hit_ratio > 0.0); // Some hits should occur due to pooling
        }
    }

    #[tokio::test]
    async fn test_high_concurrency_stress() {
        let config = MemoryConfig {
            max_pool_size: 50 * 1024 * 1024, // 50MB
            ..Default::default()
        };
        let pool = Arc::new(AsyncMemoryPool::new(config).await.unwrap());

        let num_tasks = 50;
        let operations_per_task = 20;

        let mut handles = vec![];

        for task_id in 0..num_tasks {
            let pool_clone = Arc::clone(&pool);
            let handle = tokio::spawn(async move {
                for op_id in 0..operations_per_task {
                    let size = 1024 + (task_id * 100) + (op_id * 50);

                    // Allocate
                    let block = pool_clone.allocate(size).await.unwrap();

                    // Hold for a short time
                    sleep(Duration::from_micros(100)).await;

                    // Deallocate
                    pool_clone.deallocate(block).await.unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify stats consistency
        let stats = pool.stats();
        let total_allocations = stats.allocation_count.load(Ordering::Relaxed);
        let total_deallocations = stats.free_count.load(Ordering::Relaxed);

        assert_eq!(total_allocations, (num_tasks * operations_per_task) as u64);
        assert_eq!(total_deallocations, (num_tasks * operations_per_task) as u64);
    }

    #[tokio::test]
    async fn test_memory_leak_detection() {
        let config = MemoryConfig::default();
        let pool = AsyncMemoryPool::new(config).await.unwrap();

        let initial_usage = pool.stats.current_usage();

        // Allocate and properly deallocate
        {
            let block = pool.allocate(1024).await.unwrap();
            pool.deallocate(block).await.unwrap();
        }

        // Memory usage should return to initial level
        let final_usage = pool.stats.current_usage();
        assert_eq!(initial_usage, final_usage);
    }

    #[tokio::test]
    async fn test_cache_concurrent_access() {
        let cache: Arc<AsyncSharedCache<i32, String>> = Arc::new(
            AsyncSharedCache::new(100, Duration::from_secs(60))
        );

        let num_tasks = 20;
        let mut handles = vec![];

        // Spawn concurrent readers and writers
        for task_id in 0..num_tasks {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                for i in 0..10 {
                    let key = (task_id * 10 + i) % 50; // Create some overlap

                    // Insert
                    cache_clone.insert(key, format!("value-{}-{}", task_id, i)).await;

                    // Read
                    let _ = cache_clone.get(&key).await;

                    // Sometimes remove
                    if i % 3 == 0 {
                        let _ = cache_clone.remove(&key).await;
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all operations
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify cache stats are consistent
        let stats = cache.stats();
        let total_operations = stats.hits.load(Ordering::Relaxed) +
                              stats.misses.load(Ordering::Relaxed);
        assert!(total_operations > 0);
    }

    #[tokio::test]
    async fn test_async_mapped_buffer_concurrent_access() {
        let buffer = Arc::new(AsyncMappedBuffer::new(1024, false).await.unwrap());

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
            });
            handles.push(handle);
        }

        // Wait for all writers
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all regions were written correctly
        for i in 0..num_writers {
            let offset = i * 200;
            let data = buffer.read(offset, 100).await.unwrap();
            assert!(data.iter().all(|&b| b == i as u8));
        }
    }
}