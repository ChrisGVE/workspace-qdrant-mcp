//! Compiled glob patterns for efficient file matching with LRU cache

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use glob::Pattern;
use lru::LruCache;

use super::config::WatcherConfig;
use super::WatchingError;

/// Compiled patterns for efficient matching with LRU cache
#[derive(Debug)]
pub(super) struct CompiledPatterns {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
    /// LRU cache for pattern matching results (path -> should_process)
    cache: std::sync::Mutex<LruCache<PathBuf, bool>>,
}

impl CompiledPatterns {
    pub(super) fn new(config: &WatcherConfig) -> Result<Self, WatchingError> {
        let include = config
            .include_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()?;

        let exclude = config
            .exclude_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()?;

        // Create LRU cache with 10K capacity for pattern match results
        let cache = std::sync::Mutex::new(LruCache::new(NonZeroUsize::new(10_000).unwrap()));

        Ok(Self {
            include,
            exclude,
            cache,
        })
    }

    pub(super) fn should_process(&self, path: &Path) -> bool {
        // Check cache first for fast path
        {
            let mut cache_lock = self.cache.lock().unwrap();
            if let Some(&cached_result) = cache_lock.get(path) {
                return cached_result;
            }
        }

        // Fast-path exclusion checks before expensive glob matching
        // These common patterns are checked via simple string operations
        let path_str = path.to_string_lossy();

        // Fast-path suffix checks for common temporary files
        if path_str.ends_with(".tmp")
            || path_str.ends_with(".swp")
            || path_str.ends_with(".bak")
            || path_str.ends_with("~")
        {
            let mut cache_lock = self.cache.lock().unwrap();
            cache_lock.push(path.to_path_buf(), false);
            return false;
        }

        // Fast-path prefix/component checks for common excluded directories
        if path_str.contains("/.git/")
            || path_str.contains("/node_modules/")
            || path_str.contains("/target/")
            || path_str.contains("/__pycache__/")
            || path_str.contains("/.svn/")
            || path_str.contains("/.pytest_cache/")
        {
            let mut cache_lock = self.cache.lock().unwrap();
            cache_lock.push(path.to_path_buf(), false);
            return false;
        }

        // Fast-path filename checks for common system files
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename == ".DS_Store" || filename == "Thumbs.db" {
                let mut cache_lock = self.cache.lock().unwrap();
                cache_lock.push(path.to_path_buf(), false);
                return false;
            }
        }

        // Fall back to glob pattern matching for more complex patterns
        // Check exclude patterns first (more specific)
        for pattern in &self.exclude {
            if pattern.matches(&path_str) {
                let mut cache_lock = self.cache.lock().unwrap();
                cache_lock.push(path.to_path_buf(), false);
                return false;
            }
        }

        // If no include patterns, allow all
        if self.include.is_empty() {
            let mut cache_lock = self.cache.lock().unwrap();
            cache_lock.push(path.to_path_buf(), true);
            return true;
        }

        // Check include patterns
        for pattern in &self.include {
            if pattern.matches(&path_str) {
                let mut cache_lock = self.cache.lock().unwrap();
                cache_lock.push(path.to_path_buf(), true);
                return true;
            }
        }

        // No match, exclude by default
        let mut cache_lock = self.cache.lock().unwrap();
        cache_lock.push(path.to_path_buf(), false);
        false
    }

    /// Get cache statistics for monitoring
    pub(super) fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}
