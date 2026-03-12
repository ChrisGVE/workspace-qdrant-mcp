//! LRU phrase embedding cache for the EmbeddingGenerator.
//!
//! Caches dense vectors for short phrases (≤5 words, ≤64 characters) to avoid
//! re-embedding the same keyword/tag phrases across documents. Common programming
//! keywords, stdlib names, and domain terms appear repeatedly — the LRU cache
//! captures these with negligible memory overhead (~6 MB at 4096 entries).

use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

/// Maximum word count for a phrase to be eligible for caching.
const MAX_CACHE_WORDS: usize = 5;
/// Maximum character length for a phrase to be eligible for caching.
const MAX_CACHE_CHARS: usize = 64;

/// LRU cache for short-phrase dense embeddings.
pub(super) struct PhraseCache {
    inner: Arc<Mutex<LruCache<String, Vec<f32>>>>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl PhraseCache {
    /// Create a new `PhraseCache` with the given capacity (number of entries).
    ///
    /// Capacity is clamped to at least 1 to satisfy `NonZeroUsize`.
    pub(super) fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).expect("capacity >= 1");
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(cap))),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns true if the phrase is eligible for caching (short enough).
    pub(super) fn is_cacheable(phrase: &str) -> bool {
        let trimmed = phrase.trim();
        trimmed.len() <= MAX_CACHE_CHARS && trimmed.split_whitespace().count() <= MAX_CACHE_WORDS
    }

    /// Look up a phrase in the cache. Returns `None` on cache miss.
    pub(super) async fn get(&self, phrase: &str) -> Option<Vec<f32>> {
        if !Self::is_cacheable(phrase) {
            return None;
        }
        let key = phrase.trim().to_lowercase();
        let result = self.inner.lock().await.get(&key).cloned();
        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Insert a phrase → vector pair into the cache (only if cacheable).
    pub(super) async fn put(&self, phrase: &str, vector: Vec<f32>) {
        if !Self::is_cacheable(phrase) {
            return;
        }
        let key = phrase.trim().to_lowercase();
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.inner.lock().await.put(key, vector);
    }

    /// Returns `(hits, misses, current_size)`.
    pub(super) async fn stats(&self) -> (u64, u64, usize) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let size = self.inner.lock().await.len();

        let total = hits + misses;
        if total > 0 && total % 1000 == 0 {
            let hit_pct = hits * 100 / total;
            debug!(
                hits = hits,
                misses = misses,
                size = size,
                hit_pct = hit_pct,
                "phrase cache stats"
            );
        }

        (hits, misses, size)
    }

    /// Clear all cached entries and reset counters.
    pub(super) async fn clear(&self) {
        self.inner.lock().await.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_hit_and_miss() {
        let cache = PhraseCache::new(16);
        let vec = vec![1.0f32, 0.0, 0.0];

        // Miss on first lookup
        assert!(cache.get("hello").await.is_none());

        // Insert then hit
        cache.put("hello", vec.clone()).await;
        let result = cache.get("hello").await;
        assert_eq!(result, Some(vec.clone()));

        let (hits, misses, size) = cache.stats().await;
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(size, 1);
    }

    #[tokio::test]
    async fn test_cache_normalizes_key() {
        let cache = PhraseCache::new(16);
        let vec = vec![0.5f32, 0.5];

        // Insert with mixed case
        cache.put("Rust Programming", vec.clone()).await;

        // Look up with lowercase — should hit
        let result = cache.get("rust programming").await;
        assert_eq!(result, Some(vec));
    }

    #[tokio::test]
    async fn test_long_phrase_not_cached() {
        let cache = PhraseCache::new(16);
        let long = "a b c d e f"; // 6 words — exceeds MAX_CACHE_WORDS

        cache.put(long, vec![1.0]).await;
        assert!(
            cache.get(long).await.is_none(),
            "6-word phrase must not be cached"
        );
    }

    #[tokio::test]
    async fn test_long_char_phrase_not_cached() {
        let cache = PhraseCache::new(16);
        let long_char = "a".repeat(65); // 65 chars — exceeds MAX_CACHE_CHARS

        cache.put(&long_char, vec![1.0]).await;
        assert!(
            cache.get(&long_char).await.is_none(),
            "65-char phrase must not be cached"
        );
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let cache = PhraseCache::new(2); // capacity = 2

        cache.put("alpha", vec![1.0]).await;
        cache.put("beta", vec![2.0]).await;

        // Access alpha to make it recently used
        let _ = cache.get("alpha").await;

        // Insert gamma — should evict beta (least recently used)
        cache.put("gamma", vec![3.0]).await;

        assert!(
            cache.get("alpha").await.is_some(),
            "alpha should survive LRU eviction"
        );
        assert!(
            cache.get("gamma").await.is_some(),
            "gamma should be present"
        );
        // beta may have been evicted (LRU after alpha was accessed)
    }

    #[tokio::test]
    async fn test_clear_resets_cache() {
        let cache = PhraseCache::new(16);
        cache.put("foo", vec![1.0]).await;
        cache.clear().await;

        assert!(
            cache.get("foo").await.is_none(),
            "cache should be empty after clear"
        );
        let (hits, misses, size) = cache.stats().await;
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
        assert_eq!(size, 0);
    }
}
