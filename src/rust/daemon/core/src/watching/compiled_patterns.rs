//! Compiled glob patterns for efficient file matching with LRU cache

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use glob::Pattern;
use lru::LruCache;

use super::config::WatcherConfig;
use super::WatchingError;

/// LRU cache capacity for pattern match results.
///
/// Evaluated as a `const` so the nonzero invariant is checked at compile time
/// rather than via a runtime `.unwrap()` — satisfying the panic-free protocol.
const CACHE_CAPACITY: NonZeroUsize = match NonZeroUsize::new(10_000) {
    Some(n) => n,
    // 10_000 is a nonzero literal, so this arm is statically unreachable.
    None => unreachable!(),
};

/// Compiled patterns for efficient matching with LRU cache
#[derive(Debug)]
pub(super) struct CompiledPatterns {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
    /// LRU cache for pattern matching results (path -> should_process)
    cache: Mutex<LruCache<PathBuf, bool>>,
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
        let cache = Mutex::new(LruCache::new(CACHE_CAPACITY));

        Ok(Self {
            include,
            exclude,
            cache,
        })
    }

    /// Lock the match-result cache, recovering the guard if the mutex was
    /// poisoned by a panicking thread.
    ///
    /// The cache is a non-critical performance accelerator: a poisoned lock
    /// only means some other thread panicked mid-update, leaving the cache in
    /// a logically valid (just possibly stale) state. Recovering the guard
    /// keeps the file-watching hot path alive instead of cascading the panic
    /// into every subsequent `should_process` call.
    fn lock_cache(&self) -> MutexGuard<'_, LruCache<PathBuf, bool>> {
        self.cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(super) fn should_process(&self, path: &Path) -> bool {
        // Check cache first for fast path
        {
            let mut cache_lock = self.lock_cache();
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
            let mut cache_lock = self.lock_cache();
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
            let mut cache_lock = self.lock_cache();
            cache_lock.push(path.to_path_buf(), false);
            return false;
        }

        // Fast-path filename checks for common system files
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename == ".DS_Store" || filename == "Thumbs.db" {
                let mut cache_lock = self.lock_cache();
                cache_lock.push(path.to_path_buf(), false);
                return false;
            }
        }

        // Fall back to glob pattern matching for more complex patterns
        // Check exclude patterns first (more specific)
        for pattern in &self.exclude {
            if pattern.matches(&path_str) {
                let mut cache_lock = self.lock_cache();
                cache_lock.push(path.to_path_buf(), false);
                return false;
            }
        }

        // If no include patterns, allow all
        if self.include.is_empty() {
            let mut cache_lock = self.lock_cache();
            cache_lock.push(path.to_path_buf(), true);
            return true;
        }

        // Check include patterns
        for pattern in &self.include {
            if pattern.matches(&path_str) {
                let mut cache_lock = self.lock_cache();
                cache_lock.push(path.to_path_buf(), true);
                return true;
            }
        }

        // No match, exclude by default
        let mut cache_lock = self.lock_cache();
        cache_lock.push(path.to_path_buf(), false);
        false
    }

    /// Get cache statistics for monitoring
    pub(super) fn cache_len(&self) -> usize {
        self.lock_cache().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Build a config with explicit include/exclude lists; everything else
    /// is left at WatcherConfig::default().
    fn cfg(includes: &[&str], excludes: &[&str]) -> WatcherConfig {
        WatcherConfig {
            include_patterns: includes.iter().map(|s| s.to_string()).collect(),
            exclude_patterns: excludes.iter().map(|s| s.to_string()).collect(),
            ..WatcherConfig::default()
        }
    }

    #[test]
    fn new_returns_error_for_invalid_glob() {
        let bad = cfg(&["[unterminated"], &[]);
        assert!(CompiledPatterns::new(&bad).is_err());
    }

    #[test]
    fn empty_include_allows_all_unless_excluded() {
        let p = CompiledPatterns::new(&cfg(&[], &[])).unwrap();
        assert!(p.should_process(Path::new("anything.txt")));
        assert!(p.should_process(Path::new("README")));
    }

    #[test]
    fn empty_include_still_honours_exclude() {
        let p = CompiledPatterns::new(&cfg(&[], &["*.log"])).unwrap();
        assert!(!p.should_process(Path::new("app.log")));
        assert!(p.should_process(Path::new("README.md")));
    }

    #[test]
    fn exclude_takes_precedence_over_include() {
        let p = CompiledPatterns::new(&cfg(&["*.rs"], &["secret*.rs"])).unwrap();
        assert!(p.should_process(Path::new("main.rs")));
        assert!(!p.should_process(Path::new("secret_key.rs")));
    }

    #[test]
    fn default_excludes_when_include_does_not_match() {
        // Include is restrictive; non-matching files are skipped.
        let p = CompiledPatterns::new(&cfg(&["*.md"], &[])).unwrap();
        assert!(p.should_process(Path::new("notes.md")));
        assert!(!p.should_process(Path::new("script.sh")));
    }

    #[test]
    fn fast_path_excludes_temp_file_suffixes() {
        // Empty patterns + permissive includes → fall-throughs would
        // allow these; the fast path must reject them by suffix alone.
        let p = CompiledPatterns::new(&cfg(&[], &[])).unwrap();
        assert!(!p.should_process(Path::new("foo.tmp")));
        assert!(!p.should_process(Path::new("foo.swp")));
        assert!(!p.should_process(Path::new("foo.bak")));
        assert!(!p.should_process(Path::new("foo~")));
    }

    #[test]
    fn fast_path_excludes_known_subdirs() {
        let p = CompiledPatterns::new(&cfg(&[], &[])).unwrap();
        // Always-forward-slash component check (works on all platforms
        // because the substring comparison uses literal `/` regardless of
        // the host path separator).
        assert!(!p.should_process(Path::new("/repo/.git/HEAD")));
        assert!(!p.should_process(Path::new("/repo/node_modules/pkg/index.js")));
        assert!(!p.should_process(Path::new("/repo/target/debug/build.log")));
        assert!(!p.should_process(Path::new("/repo/src/__pycache__/m.pyc")));
        assert!(!p.should_process(Path::new("/repo/.svn/entries")));
        assert!(!p.should_process(Path::new("/repo/.pytest_cache/v/cache/lastfailed")));
    }

    #[test]
    fn fast_path_excludes_system_files_by_basename() {
        let p = CompiledPatterns::new(&cfg(&[], &[])).unwrap();
        assert!(!p.should_process(Path::new("/some/dir/.DS_Store")));
        assert!(!p.should_process(Path::new("/some/dir/Thumbs.db")));
    }

    #[test]
    fn lru_cache_records_decisions_for_repeated_lookups() {
        let p = CompiledPatterns::new(&cfg(&["*.rs"], &["*.tmp"])).unwrap();
        assert_eq!(p.cache_len(), 0);

        p.should_process(Path::new("a.rs"));
        p.should_process(Path::new("b.tmp"));
        p.should_process(Path::new("c.rs"));
        assert_eq!(p.cache_len(), 3);

        // Repeated lookups hit the cache rather than expanding it.
        p.should_process(Path::new("a.rs"));
        p.should_process(Path::new("b.tmp"));
        assert_eq!(p.cache_len(), 3);
    }

    #[test]
    fn should_process_survives_poisoned_cache_lock() {
        use std::sync::Arc;

        // Poison the internal cache mutex from a panicking thread, then verify
        // the hot path keeps working instead of cascading the panic.
        let p = Arc::new(CompiledPatterns::new(&cfg(&["*.rs"], &["*.tmp"])).unwrap());

        let poisoner = {
            let p = Arc::clone(&p);
            std::thread::spawn(move || {
                let _guard = p.cache.lock().unwrap();
                panic!("intentional panic to poison the cache mutex");
            })
        };
        assert!(poisoner.join().is_err(), "poisoner thread should have panicked");
        assert!(p.cache.is_poisoned(), "cache mutex should be poisoned");

        // These calls would panic with a bare `.lock().unwrap()`; with
        // poison recovery they succeed and continue caching.
        assert!(p.should_process(Path::new("main.rs")));
        assert!(!p.should_process(Path::new("scratch.tmp")));
        assert_eq!(p.cache_len(), 2);
    }
}
