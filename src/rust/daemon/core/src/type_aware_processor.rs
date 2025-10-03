//! Type-Aware Processing Module
//!
//! Provides collection type-specific performance optimizations for queue processing.
//! Matches Python configuration in collection_type_config.py to ensure consistency
//! across the system.
//!
//! Collection types:
//! - SYSTEM: Infrastructure collections (__ prefix) - smaller batch, lower concurrency
//! - LIBRARY: Library documentation (_ prefix) - medium batch, medium concurrency
//! - PROJECT: Project-scoped ({project}-{suffix}) - larger batch, higher concurrency
//! - GLOBAL: System-wide collections - largest batch, highest concurrency
//! - UNKNOWN: Unclassified collections - default medium settings

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Performance settings for a collection type
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionTypeSettings {
    /// Number of items to process in each batch
    pub batch_size: i32,

    /// Maximum number of concurrent operations for this type
    pub max_concurrent_operations: usize,

    /// Priority weight (1=lowest, 5=highest) for queue ordering
    pub priority_weight: i32,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl CollectionTypeSettings {
    /// Create settings for SYSTEM collection type
    /// System collections (__ prefix) - CLI-writable, LLM-readable
    pub fn system() -> Self {
        Self {
            batch_size: 50,
            max_concurrent_operations: 3,
            priority_weight: 4,
            cache_ttl_seconds: 600,
        }
    }

    /// Create settings for LIBRARY collection type
    /// Library collections (_ prefix) - CLI-managed, MCP-readonly
    pub fn library() -> Self {
        Self {
            batch_size: 100,
            max_concurrent_operations: 5,
            priority_weight: 3,
            cache_ttl_seconds: 900,
        }
    }

    /// Create settings for PROJECT collection type
    /// Project collections ({project}-{suffix}) - user-created, project-scoped
    pub fn project() -> Self {
        Self {
            batch_size: 150,
            max_concurrent_operations: 10,
            priority_weight: 2,
            cache_ttl_seconds: 300,
        }
    }

    /// Create settings for GLOBAL collection type
    /// Global collections - system-wide, always available
    pub fn global() -> Self {
        Self {
            batch_size: 200,
            max_concurrent_operations: 8,
            priority_weight: 5,
            cache_ttl_seconds: 1800,
        }
    }

    /// Create default settings for UNKNOWN collection types
    pub fn unknown() -> Self {
        Self {
            batch_size: 100,
            max_concurrent_operations: 5,
            priority_weight: 1,
            cache_ttl_seconds: 300,
        }
    }
}

/// Get performance settings for a collection type
///
/// # Arguments
/// * `collection_type` - Optional collection type string ("system", "library", "project", "global")
///
/// # Returns
/// Settings for the specified type, or default settings for unknown/None types
pub fn get_settings_for_type(collection_type: Option<&str>) -> CollectionTypeSettings {
    match collection_type {
        Some("system") => CollectionTypeSettings::system(),
        Some("library") => CollectionTypeSettings::library(),
        Some("project") => CollectionTypeSettings::project(),
        Some("global") => CollectionTypeSettings::global(),
        _ => CollectionTypeSettings::unknown(),
    }
}

/// Concurrent operation tracker for type-aware rate limiting
#[derive(Debug, Clone)]
pub struct ConcurrentOperationTracker {
    /// Map of collection type to current concurrent operation count
    counts: Arc<RwLock<HashMap<String, usize>>>,
}

impl ConcurrentOperationTracker {
    /// Create a new concurrent operation tracker
    pub fn new() -> Self {
        Self {
            counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Try to acquire a slot for the given collection type
    ///
    /// # Arguments
    /// * `collection_type` - The collection type to acquire for
    ///
    /// # Returns
    /// `true` if slot was acquired, `false` if concurrent limit reached
    pub async fn try_acquire(&self, collection_type: Option<&str>) -> bool {
        let type_str = collection_type.unwrap_or("unknown").to_string();
        let settings = get_settings_for_type(collection_type);
        let max_concurrent = settings.max_concurrent_operations;

        let mut counts = self.counts.write().await;
        let current = counts.get(&type_str).copied().unwrap_or(0);

        if current < max_concurrent {
            counts.insert(type_str, current + 1);
            true
        } else {
            false
        }
    }

    /// Release a slot for the given collection type
    ///
    /// # Arguments
    /// * `collection_type` - The collection type to release for
    pub async fn release(&self, collection_type: Option<&str>) {
        let type_str = collection_type.unwrap_or("unknown").to_string();
        let mut counts = self.counts.write().await;

        if let Some(current) = counts.get_mut(&type_str) {
            *current = current.saturating_sub(1);
            if *current == 0 {
                counts.remove(&type_str);
            }
        }
    }

    /// Get current concurrent operation count for a type
    ///
    /// # Arguments
    /// * `collection_type` - The collection type to query
    ///
    /// # Returns
    /// Current number of concurrent operations for this type
    pub async fn get_count(&self, collection_type: Option<&str>) -> usize {
        let type_str = collection_type.unwrap_or("unknown").to_string();
        let counts = self.counts.read().await;
        counts.get(&type_str).copied().unwrap_or(0)
    }

    /// Get all concurrent operation counts
    ///
    /// # Returns
    /// HashMap of collection type to concurrent operation count
    pub async fn get_all_counts(&self) -> HashMap<String, usize> {
        let counts = self.counts.read().await;
        counts.clone()
    }
}

impl Default for ConcurrentOperationTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for automatic release of concurrent operation slot
pub struct ConcurrentOperationGuard<'a> {
    tracker: &'a ConcurrentOperationTracker,
    collection_type: Option<String>,
}

impl<'a> ConcurrentOperationGuard<'a> {
    /// Create a new guard
    pub fn new(tracker: &'a ConcurrentOperationTracker, collection_type: Option<&str>) -> Self {
        Self {
            tracker,
            collection_type: collection_type.map(|s| s.to_string()),
        }
    }
}

impl<'a> Drop for ConcurrentOperationGuard<'a> {
    fn drop(&mut self) {
        // Release slot when guard is dropped
        let tracker = self.tracker.clone();
        let collection_type = self.collection_type.clone();

        // Spawn async task to release the slot
        tokio::spawn(async move {
            tracker.release(collection_type.as_deref()).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_settings() {
        let settings = CollectionTypeSettings::system();
        assert_eq!(settings.batch_size, 50);
        assert_eq!(settings.max_concurrent_operations, 3);
        assert_eq!(settings.priority_weight, 4);
        assert_eq!(settings.cache_ttl_seconds, 600);
    }

    #[test]
    fn test_library_settings() {
        let settings = CollectionTypeSettings::library();
        assert_eq!(settings.batch_size, 100);
        assert_eq!(settings.max_concurrent_operations, 5);
        assert_eq!(settings.priority_weight, 3);
        assert_eq!(settings.cache_ttl_seconds, 900);
    }

    #[test]
    fn test_project_settings() {
        let settings = CollectionTypeSettings::project();
        assert_eq!(settings.batch_size, 150);
        assert_eq!(settings.max_concurrent_operations, 10);
        assert_eq!(settings.priority_weight, 2);
        assert_eq!(settings.cache_ttl_seconds, 300);
    }

    #[test]
    fn test_global_settings() {
        let settings = CollectionTypeSettings::global();
        assert_eq!(settings.batch_size, 200);
        assert_eq!(settings.max_concurrent_operations, 8);
        assert_eq!(settings.priority_weight, 5);
        assert_eq!(settings.cache_ttl_seconds, 1800);
    }

    #[test]
    fn test_unknown_settings() {
        let settings = CollectionTypeSettings::unknown();
        assert_eq!(settings.batch_size, 100);
        assert_eq!(settings.max_concurrent_operations, 5);
        assert_eq!(settings.priority_weight, 1);
        assert_eq!(settings.cache_ttl_seconds, 300);
    }

    #[test]
    fn test_get_settings_for_type() {
        assert_eq!(
            get_settings_for_type(Some("system")),
            CollectionTypeSettings::system()
        );
        assert_eq!(
            get_settings_for_type(Some("library")),
            CollectionTypeSettings::library()
        );
        assert_eq!(
            get_settings_for_type(Some("project")),
            CollectionTypeSettings::project()
        );
        assert_eq!(
            get_settings_for_type(Some("global")),
            CollectionTypeSettings::global()
        );
        assert_eq!(
            get_settings_for_type(Some("unknown_type")),
            CollectionTypeSettings::unknown()
        );
        assert_eq!(
            get_settings_for_type(None),
            CollectionTypeSettings::unknown()
        );
    }

    #[tokio::test]
    async fn test_concurrent_tracker_acquire_release() {
        let tracker = ConcurrentOperationTracker::new();

        // Should be able to acquire up to limit
        assert!(tracker.try_acquire(Some("system")).await);
        assert_eq!(tracker.get_count(Some("system")).await, 1);

        assert!(tracker.try_acquire(Some("system")).await);
        assert_eq!(tracker.get_count(Some("system")).await, 2);

        assert!(tracker.try_acquire(Some("system")).await);
        assert_eq!(tracker.get_count(Some("system")).await, 3);

        // Should fail to acquire beyond limit (system max is 3)
        assert!(!tracker.try_acquire(Some("system")).await);
        assert_eq!(tracker.get_count(Some("system")).await, 3);

        // Release one slot
        tracker.release(Some("system")).await;
        assert_eq!(tracker.get_count(Some("system")).await, 2);

        // Should be able to acquire again
        assert!(tracker.try_acquire(Some("system")).await);
        assert_eq!(tracker.get_count(Some("system")).await, 3);
    }

    #[tokio::test]
    async fn test_concurrent_tracker_multiple_types() {
        let tracker = ConcurrentOperationTracker::new();

        // Acquire for different types
        assert!(tracker.try_acquire(Some("system")).await);
        assert!(tracker.try_acquire(Some("library")).await);
        assert!(tracker.try_acquire(Some("project")).await);

        assert_eq!(tracker.get_count(Some("system")).await, 1);
        assert_eq!(tracker.get_count(Some("library")).await, 1);
        assert_eq!(tracker.get_count(Some("project")).await, 1);

        // Get all counts
        let all_counts = tracker.get_all_counts().await;
        assert_eq!(all_counts.get("system"), Some(&1));
        assert_eq!(all_counts.get("library"), Some(&1));
        assert_eq!(all_counts.get("project"), Some(&1));
    }

    #[tokio::test]
    async fn test_concurrent_tracker_none_type() {
        let tracker = ConcurrentOperationTracker::new();

        // None should use "unknown" type
        assert!(tracker.try_acquire(None).await);
        assert_eq!(tracker.get_count(None).await, 1);

        tracker.release(None).await;
        assert_eq!(tracker.get_count(None).await, 0);
    }

    #[tokio::test]
    async fn test_concurrent_tracker_saturating_sub() {
        let tracker = ConcurrentOperationTracker::new();

        // Release without acquire should not panic (saturating_sub)
        tracker.release(Some("system")).await;
        assert_eq!(tracker.get_count(Some("system")).await, 0);
    }
}
