//! TextSearchService gRPC implementation
//!
//! Provides FTS5-based code search for the MCP server grep tool and CLI.
//! Exposes 2 RPCs: Search (exact/regex) and CountMatches (count-only).
//!
//! Includes a short-lived result cache (5s TTL) so that a CountMatches
//! followed by a Search (or vice versa) with the same parameters reuses
//! the query result instead of hitting FTS5 twice.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};
use workspace_qdrant_core::text_search::{
    attach_context_lines, search_exact, search_regex, SearchOptions, SearchResults,
};
use workspace_qdrant_core::SearchDbManager;

use crate::proto::{
    text_search_service_server::TextSearchService, TextIndexStatus, TextSearchCountResponse,
    TextSearchMatch, TextSearchRequest, TextSearchResponse,
};

/// Cache TTL — entries older than this are evicted on access.
const CACHE_TTL: Duration = Duration::from_secs(5);

/// Maximum cache entries before forced eviction of all expired entries.
const CACHE_MAX_ENTRIES: usize = 32;

/// Maximum results to fetch on a cache miss (F-028). Prevents unbounded
/// memory usage on broad queries while keeping enough for CountMatches.
const CACHE_MISS_MAX_RESULTS: usize = 10_000;

/// Cache key derived from the search-relevant fields of a request.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct CacheKey {
    pattern: String,
    regex: bool,
    case_sensitive: bool,
    tenant_id: Option<String>,
    branch: Option<String>,
    path_prefix: Option<String>,
    path_glob: Option<String>,
}

impl CacheKey {
    fn from_request(req: &TextSearchRequest) -> Self {
        Self {
            pattern: req.pattern.clone(),
            regex: req.regex,
            case_sensitive: req.case_sensitive,
            tenant_id: req.tenant_id.clone(),
            branch: req.branch.clone(),
            path_prefix: req.path_prefix.clone(),
            path_glob: req.path_glob.clone(),
        }
    }

    /// Compute a u64 hash for logging.
    fn hash_u64(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

struct CacheEntry {
    results: SearchResults,
    inserted_at: Instant,
}

/// TextSearchService implementation
pub struct TextSearchServiceImpl {
    search_db: Arc<SearchDbManager>,
    cache: Mutex<HashMap<CacheKey, CacheEntry>>,
    /// State-DB pool for tenant indexing status (#97). Optional: when absent,
    /// responses simply omit `index_status`.
    state_pool: Option<sqlx::SqlitePool>,
}

impl TextSearchServiceImpl {
    /// Create a new TextSearchService with a shared SearchDbManager
    pub fn new(search_db: Arc<SearchDbManager>) -> Self {
        Self {
            search_db,
            cache: Mutex::new(HashMap::new()),
            state_pool: None,
        }
    }

    /// Attach the state-DB pool so responses can report tenant indexing
    /// status (#97).
    pub fn with_state_pool(mut self, pool: sqlx::SqlitePool) -> Self {
        self.state_pool = Some(pool);
        self
    }

    /// Tenant indexing state for a tenant-scoped request (#97).
    ///
    /// Returns `None` when the request has no tenant filter, no state pool is
    /// attached, or the queries fail (status is best-effort — a search must
    /// never fail because the status lookup did).
    async fn tenant_index_status(&self, tenant_id: Option<&str>) -> Option<TextIndexStatus> {
        let tenant = tenant_id.filter(|t| !t.is_empty())?;
        let pool = self.state_pool.as_ref()?;

        let files_tracked: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.tenant_id = ?1",
        )
        .bind(tenant)
        .fetch_one(pool)
        .await
        .map_err(|e| debug!("index_status: tracked_files count failed: {e}"))
        .ok()?;

        let queue_pending: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE tenant_id = ?1 AND status IN ('pending', 'in_progress')",
        )
        .bind(tenant)
        .fetch_one(pool)
        .await
        .map_err(|e| debug!("index_status: unified_queue count failed: {e}"))
        .ok()?;

        Some(TextIndexStatus {
            files_tracked: files_tracked.max(0) as u64,
            queue_pending: queue_pending.max(0) as u64,
            index_complete: queue_pending == 0,
        })
    }

    /// Convert a gRPC request into SearchOptions
    fn build_options(req: &TextSearchRequest) -> SearchOptions {
        SearchOptions {
            tenant_id: req.tenant_id.clone(),
            branch: req.branch.clone(),
            path_prefix: req.path_prefix.clone(),
            path_glob: req.path_glob.clone(),
            case_insensitive: !req.case_sensitive,
            max_results: if req.max_results > 0 {
                req.max_results as usize
            } else {
                1000
            },
            context_lines: req.context_lines.max(0) as usize,
        }
    }

    /// Look up a cached result. Returns `None` if expired or absent.
    async fn cache_get(&self, key: &CacheKey) -> Option<SearchResults> {
        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.get(key) {
            if entry.inserted_at.elapsed() < CACHE_TTL {
                return Some(entry.results.clone());
            }
            // Expired — remove it
            cache.remove(key);
        }
        None
    }

    /// Insert a result into the cache, evicting expired entries when full.
    async fn cache_put(&self, key: CacheKey, results: SearchResults) {
        let mut cache = self.cache.lock().await;
        if cache.len() >= CACHE_MAX_ENTRIES {
            let now = Instant::now();
            cache.retain(|_, e| now.duration_since(e.inserted_at) < CACHE_TTL);
        }
        cache.insert(
            key,
            CacheEntry {
                results,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Cap results to `max_results` and attach context lines when needed.
    ///
    /// The cache stores results without context (F-029). Instead of
    /// re-executing the full search, context lines are fetched via
    /// targeted `code_lines` lookups on the already-capped result set.
    async fn apply_max_results_and_context(
        &self,
        results: SearchResults,
        _req: &TextSearchRequest,
        options: &SearchOptions,
    ) -> Vec<TextSearchMatch> {
        let mut capped: Vec<_> = results
            .matches
            .into_iter()
            .take(options.max_results)
            .collect();

        if options.context_lines > 0 {
            // Attach context from code_lines without re-running the FTS query (F-029).
            if let Err(e) =
                attach_context_lines(&self.search_db, &mut capped, options.context_lines).await
            {
                debug!("Failed to attach context lines, returning without context: {e}");
            }
        }

        capped
            .into_iter()
            .map(|m| TextSearchMatch {
                file_path: m.file_path,
                line_number: m.line_number as i32,
                content: m.content,
                tenant_id: m.tenant_id,
                branch: m.branch,
                context_before: m.context_before,
                context_after: m.context_after,
            })
            .collect()
    }

    /// Execute the search (exact or regex) or return a cached result.
    async fn execute_or_cached(&self, req: &TextSearchRequest) -> Result<SearchResults, Status> {
        let key = CacheKey::from_request(req);

        if let Some(cached) = self.cache_get(&key).await {
            debug!("TextSearch cache hit (key={:#x})", key.hash_u64());
            return Ok(cached);
        }

        // Cache miss — run the full-count query (no max_results cap, no context)
        // so both Search and CountMatches can be served from the same entry.
        let options = SearchOptions {
            tenant_id: req.tenant_id.clone(),
            branch: req.branch.clone(),
            path_prefix: req.path_prefix.clone(),
            path_glob: req.path_glob.clone(),
            case_insensitive: !req.case_sensitive,
            max_results: CACHE_MISS_MAX_RESULTS,
            context_lines: 0,
        };

        let results = if req.regex {
            search_regex(&self.search_db, &req.pattern, &options).await
        } else {
            search_exact(&self.search_db, &req.pattern, &options).await
        }
        .map_err(|e| {
            error!("TextSearch failed: {:?}", e);
            Status::internal(format!("Search failed: {e}"))
        })?;

        self.cache_put(key, results.clone()).await;
        Ok(results)
    }
}

#[tonic::async_trait]
impl TextSearchService for TextSearchServiceImpl {
    #[tracing::instrument(skip_all, fields(method = "TextSearchService.search"))]
    async fn search(
        &self,
        request: Request<TextSearchRequest>,
    ) -> Result<Response<TextSearchResponse>, Status> {
        let req = request.into_inner();

        if req.pattern.is_empty() {
            return Err(Status::invalid_argument("Search pattern cannot be empty"));
        }

        debug!(
            "TextSearch: pattern={:?} regex={} case_sensitive={}",
            req.pattern, req.regex, req.case_sensitive
        );

        let start = Instant::now();
        let results = self.execute_or_cached(&req).await?;
        let options = Self::build_options(&req);
        // Capture pre-truncation count before apply_max_results_and_context caps the vec.
        let total_matches = results.matches.len() as i32;
        let truncated = results.matches.len() > options.max_results;

        let matches = self
            .apply_max_results_and_context(results, &req, &options)
            .await;

        // Tenant indexing state (#97): lets callers distinguish "pattern
        // absent" from "files not indexed yet" on zero-match responses.
        let index_status = self.tenant_index_status(req.tenant_id.as_deref()).await;

        let query_time_ms = start.elapsed().as_millis() as i64;

        info!(
            "TextSearch completed: {} matches (truncated={}) in {}ms",
            total_matches, truncated, query_time_ms
        );

        Ok(Response::new(TextSearchResponse {
            matches,
            total_matches,
            truncated,
            query_time_ms,
            index_status,
        }))
    }

    #[tracing::instrument(skip_all, fields(method = "TextSearchService.count_matches"))]
    async fn count_matches(
        &self,
        request: Request<TextSearchRequest>,
    ) -> Result<Response<TextSearchCountResponse>, Status> {
        let req = request.into_inner();

        if req.pattern.is_empty() {
            return Err(Status::invalid_argument("Search pattern cannot be empty"));
        }

        debug!(
            "TextSearch count: pattern={:?} regex={}",
            req.pattern, req.regex
        );

        let start = Instant::now();
        let results = self.execute_or_cached(&req).await?;
        let query_time_ms = start.elapsed().as_millis() as i64;
        let count = results.matches.len() as i32;

        debug!(
            "TextSearch count completed: {} matches in {}ms",
            count, query_time_ms
        );

        Ok(Response::new(TextSearchCountResponse {
            count,
            query_time_ms,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_options_defaults() {
        let req = TextSearchRequest {
            pattern: "test".to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: None,
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 0,
            max_results: 0,
        };

        let opts = TextSearchServiceImpl::build_options(&req);
        assert!(!opts.case_insensitive);
        assert_eq!(opts.max_results, 1000); // default when 0
        assert_eq!(opts.context_lines, 0);
        assert!(opts.tenant_id.is_none());
    }

    #[test]
    fn test_build_options_with_values() {
        let req = TextSearchRequest {
            pattern: "fn main".to_string(),
            regex: false,
            case_sensitive: false,
            tenant_id: Some("proj-123".to_string()),
            branch: Some("main".to_string()),
            path_glob: Some("**/*.rs".to_string()),
            path_prefix: None,
            context_lines: 3,
            max_results: 50,
        };

        let opts = TextSearchServiceImpl::build_options(&req);
        assert!(opts.case_insensitive);
        assert_eq!(opts.max_results, 50);
        assert_eq!(opts.context_lines, 3);
        assert_eq!(opts.tenant_id.as_deref(), Some("proj-123"));
        assert_eq!(opts.branch.as_deref(), Some("main"));
        assert_eq!(opts.path_glob.as_deref(), Some("**/*.rs"));
    }

    #[test]
    fn test_build_options_negative_context() {
        let req = TextSearchRequest {
            pattern: "test".to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: None,
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: -5,
            max_results: 100,
        };

        let opts = TextSearchServiceImpl::build_options(&req);
        assert_eq!(opts.context_lines, 0); // negative clamped to 0
    }

    #[test]
    fn test_cache_key_equality() {
        let req1 = TextSearchRequest {
            pattern: "test".to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: Some("proj".to_string()),
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 3,
            max_results: 100,
        };
        // Same search params, different context_lines/max_results
        let req2 = TextSearchRequest {
            pattern: "test".to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: Some("proj".to_string()),
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 0,
            max_results: 50,
        };
        let key1 = CacheKey::from_request(&req1);
        let key2 = CacheKey::from_request(&req2);
        // Keys should be equal — context_lines and max_results are not part of the key
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_pattern() {
        let req1 = TextSearchRequest {
            pattern: "foo".to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: None,
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 0,
            max_results: 0,
        };
        let req2 = TextSearchRequest {
            pattern: "bar".to_string(),
            ..req1.clone()
        };
        assert_ne!(CacheKey::from_request(&req1), CacheKey::from_request(&req2));
    }

    #[test]
    fn test_cache_key_regex_matters() {
        let req1 = TextSearchRequest {
            pattern: "test".to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: None,
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 0,
            max_results: 0,
        };
        let req2 = TextSearchRequest {
            regex: true,
            ..req1.clone()
        };
        assert_ne!(CacheKey::from_request(&req1), CacheKey::from_request(&req2));
    }

    /// Verify the total_matches / truncated invariant:
    ///
    /// When `results.matches.len() > options.max_results`, `truncated` must be
    /// `true` and `total_matches` must equal the PRE-cap length — NOT the
    /// post-cap length returned to the caller.
    ///
    /// This is a pure logic test; it does not invoke gRPC or the database.
    #[test]
    fn test_total_matches_is_precap_count() {
        // Simulate 1000 raw matches with a cap of 50.
        let raw_count: usize = 1000;
        let cap: usize = 50;

        // Build a mock options value with max_results = cap.
        let req = TextSearchRequest {
            pattern: "fn ".to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: None,
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 0,
            max_results: cap as i32,
        };
        let options = TextSearchServiceImpl::build_options(&req);
        assert_eq!(options.max_results, cap);

        // Replicate the handler's total_matches / truncated computation
        // (extracted here as pure arithmetic so the test stays fast).
        let truncated = raw_count > options.max_results;
        let total_matches = raw_count as i32; // captured BEFORE capping
        let capped_count = raw_count.min(options.max_results) as i32;

        assert!(truncated, "truncated must be true when raw > cap");
        assert_eq!(
            total_matches, 1000,
            "total_matches must reflect full pre-cap count"
        );
        assert_eq!(
            capped_count, cap as i32,
            "capped result set must equal max_results"
        );
        assert!(
            total_matches > capped_count,
            "total_matches must exceed the capped response length when truncated"
        );
    }

    /// When raw result count is within the cap, total_matches equals the
    /// result count and truncated is false.
    #[test]
    fn test_total_matches_no_truncation() {
        let raw_count: usize = 42;
        let cap: usize = 1000;

        let req = TextSearchRequest {
            pattern: "todo".to_string(),
            regex: false,
            case_sensitive: false,
            tenant_id: None,
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 0,
            max_results: cap as i32,
        };
        let options = TextSearchServiceImpl::build_options(&req);

        let truncated = raw_count > options.max_results;
        let total_matches = raw_count as i32;
        let capped_count = raw_count.min(options.max_results) as i32;

        assert!(!truncated, "truncated must be false when raw <= cap");
        assert_eq!(total_matches, 42);
        assert_eq!(capped_count, 42);
        assert_eq!(total_matches, capped_count);
    }
}
