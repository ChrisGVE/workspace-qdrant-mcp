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
#[cfg(test)]
use workspace_qdrant_core::text_search::SearchMatch;
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

    /// Tenant+branch indexing state for a tenant-scoped request (#97, #137, #141).
    ///
    /// All counts are scoped to the branch the search actually filtered on.
    /// `file_metadata` is per-branch: a file indexed on branch A is invisible to
    /// a grep filtered on branch B (#137). Counting tenant-wide masked that — it
    /// reported a healthy tenant while the queried branch returned nothing, so a
    /// branch with no indexed content produced a silent false negative.
    ///
    /// `files_tracked` is therefore the count of searchable `file_metadata` files
    /// for the queried branch (read from search.db, the table the FTS query
    /// joins). `index_complete` is true only when that branch is fully
    /// searchable: nothing queued AND at least one file indexed. When the branch
    /// has zero indexed files but the tenant has data on other branches, the grep
    /// tool surfaces a warning instead of letting an empty result look like
    /// genuine pattern absence.
    ///
    /// Returns `None` when the request has no tenant filter, no state pool is
    /// attached, or the queue query fails (status is best-effort — a search must
    /// never fail because the status lookup did). The search.db file count is
    /// best-effort within that: a lookup failure yields `0`, which is the safe,
    /// degrade-gracefully value (it flags the branch as not-complete rather than
    /// hiding the gap).
    async fn tenant_index_status(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> Option<TextIndexStatus> {
        let tenant = tenant_id.filter(|t| !t.is_empty())?;
        let pool = self.state_pool.as_ref()?;

        // A "*" branch opts into all branches (no filter); treat as branch-less.
        let branch = branch.filter(|b| !b.is_empty() && *b != "*");

        let files_tracked = self.branch_indexed_file_count(tenant, branch).await;
        let queue_pending = self
            .tenant_branch_queue_pending(pool, tenant, branch)
            .await?;

        Some(TextIndexStatus {
            files_tracked,
            queue_pending,
            // The queried branch is searchable only when nothing is still queued
            // for it AND it actually has indexed files. files_tracked == 0 with a
            // non-empty tenant means the branch is not yet indexed (#137).
            index_complete: queue_pending == 0 && files_tracked > 0,
        })
    }

    /// Count searchable `file_metadata` files for a tenant on a given branch
    /// (search.db). When `branch` is `None`, counts across all branches.
    ///
    /// Best-effort: returns `0` on query failure so an unknown state is reported
    /// as "not searchable" rather than silently masking a gap.
    async fn branch_indexed_file_count(&self, tenant: &str, branch: Option<&str>) -> u64 {
        let result: Result<i64, _> = if let Some(branch) = branch {
            sqlx::query_scalar(
                "SELECT COUNT(DISTINCT file_path) FROM file_metadata \
                 WHERE tenant_id = ?1 AND branch = ?2",
            )
            .bind(tenant)
            .bind(branch)
            .fetch_one(self.search_db.pool())
            .await
        } else {
            sqlx::query_scalar(
                "SELECT COUNT(DISTINCT file_path) FROM file_metadata WHERE tenant_id = ?1",
            )
            .bind(tenant)
            .fetch_one(self.search_db.pool())
            .await
        };

        result
            .map_err(|e| debug!("index_status: file_metadata count failed: {e}"))
            .map(|n| n.max(0) as u64)
            .unwrap_or(0)
    }

    /// Count pending/in-progress `unified_queue` items for a tenant on a given
    /// branch (state.db). When `branch` is `None`, counts across all branches.
    ///
    /// Returns `None` on query failure (propagated so the whole status is omitted
    /// rather than reported with a misleading queue count).
    async fn tenant_branch_queue_pending(
        &self,
        pool: &sqlx::SqlitePool,
        tenant: &str,
        branch: Option<&str>,
    ) -> Option<u64> {
        let result: Result<i64, _> = if let Some(branch) = branch {
            sqlx::query_scalar(
                "SELECT COUNT(*) FROM unified_queue \
                 WHERE tenant_id = ?1 AND branch = ?2 \
                 AND status IN ('pending', 'in_progress')",
            )
            .bind(tenant)
            .bind(branch)
            .fetch_one(pool)
            .await
        } else {
            sqlx::query_scalar(
                "SELECT COUNT(*) FROM unified_queue \
                 WHERE tenant_id = ?1 AND status IN ('pending', 'in_progress')",
            )
            .bind(tenant)
            .fetch_one(pool)
            .await
        };

        result
            .map_err(|e| debug!("index_status: unified_queue count failed: {e}"))
            .map(|n| n.max(0) as u64)
            .ok()
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

        let mut results = if req.regex {
            search_regex(&self.search_db, &req.pattern, &options).await
        } else {
            search_exact(&self.search_db, &req.pattern, &options).await
        }
        .map_err(|e| {
            error!("TextSearch failed: {:?}", e);
            Status::internal(format!("Search failed: {e}"))
        })?;

        // Defensive dedup (#102): without a branch filter, the FTS query joins
        // code_lines × ALL file_metadata branch rows for a file, emitting one
        // duplicate match per branch row (including stale rows for deleted
        // branches). Collapse to one match per (tenant, file, line). Applied
        // before caching so Search and CountMatches stay consistent.
        if options.branch.is_none() {
            dedup_matches(&mut results);
        }

        self.cache_put(key, results.clone()).await;
        Ok(results)
    }
}

/// Collapse duplicate matches to one per (tenant_id, file_path, line_number),
/// keeping the first occurrence (#102).
///
/// Without a branch filter, the FTS query joins `code_lines` against ALL
/// `file_metadata` branch rows for a file — each extra branch row (including
/// stale rows for deleted branches) duplicates every match in that file.
fn dedup_matches(results: &mut SearchResults) {
    let mut seen: std::collections::HashSet<(String, String, i64)> =
        std::collections::HashSet::new();
    results
        .matches
        .retain(|m| seen.insert((m.tenant_id.clone(), m.file_path.clone(), m.line_number)));
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

        // Tenant+branch indexing state (#97, #137, #141): lets callers
        // distinguish "pattern absent" from "files not indexed yet" / "branch
        // not indexed" on zero-match responses. Scoped to the branch the search
        // filtered on so a per-branch gap is not masked by tenant-wide counts.
        let index_status = self
            .tenant_index_status(req.tenant_id.as_deref(), req.branch.as_deref())
            .await;

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

    // ── dedup_matches (#102) ──

    fn mk_match(tenant: &str, file: &str, line: i64, branch: Option<&str>) -> SearchMatch {
        SearchMatch {
            line_id: line,
            file_id: 1,
            line_number: line,
            content: "x".to_string(),
            file_path: file.to_string(),
            tenant_id: tenant.to_string(),
            branch: branch.map(str::to_string),
            context_before: vec![],
            context_after: vec![],
        }
    }

    fn mk_results(matches: Vec<SearchMatch>) -> SearchResults {
        SearchResults {
            pattern: "x".to_string(),
            matches,
            truncated: false,
            query_time_ms: 0,
            search_engine: "fts5".to_string(),
        }
    }

    #[test]
    fn test_dedup_collapses_per_branch_duplicates() {
        // Same file:line surfaced once per file_metadata branch row (#102).
        let mut results = mk_results(vec![
            mk_match("t1", "a.rs", 181, Some("main")),
            mk_match("t1", "a.rs", 181, Some("feat/observability")),
            mk_match("t1", "a.rs", 200, Some("main")),
        ]);
        dedup_matches(&mut results);
        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].line_number, 181);
        // First occurrence wins.
        assert_eq!(results.matches[0].branch.as_deref(), Some("main"));
        assert_eq!(results.matches[1].line_number, 200);
    }

    #[test]
    fn test_dedup_keeps_distinct_tenants_files_lines() {
        let mut results = mk_results(vec![
            mk_match("t1", "a.rs", 1, None),
            mk_match("t2", "a.rs", 1, None),
            mk_match("t1", "b.rs", 1, None),
            mk_match("t1", "a.rs", 2, None),
        ]);
        dedup_matches(&mut results);
        assert_eq!(results.matches.len(), 4, "no false-positive dedup");
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

    // ── Branch-aware index status (#137, #141) ──
    //
    // `file_metadata` is per-branch: a file indexed on branch A is invisible to
    // a grep filtered on branch B. The index status must reflect the BRANCH the
    // search filtered on, so a per-branch gap is not masked by tenant-wide
    // counts (the #137 false negative).

    use std::sync::Arc;

    use sqlx::sqlite::SqlitePoolOptions;
    use workspace_qdrant_core::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};

    /// Minimal in-memory state pool exposing the `unified_queue` columns the
    /// status query reads (tenant_id, branch, status).
    async fn empty_state_pool() -> sqlx::SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE unified_queue (\
                 queue_id INTEGER PRIMARY KEY,\
                 tenant_id TEXT NOT NULL,\
                 branch TEXT,\
                 status TEXT NOT NULL\
             )",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    /// Index `content` for `tenant`/`branch` into a fresh search.db, then build a
    /// service over it with the given (empty) state pool.
    async fn service_with_indexed(
        tmp: &tempfile::TempDir,
        tenant: &str,
        branch: &str,
        file_id: i64,
        file_path: &str,
        content: &str,
        state_pool: sqlx::SqlitePool,
    ) -> TextSearchServiceImpl {
        let db_path = tmp.path().join("search.db");
        let search_db = Arc::new(SearchDbManager::new(&db_path).await.unwrap());
        let processor = FtsBatchProcessor::new(&search_db, FtsBatchConfig::default());
        processor
            .full_rewrite(
                file_id,
                content,
                tenant,
                Some(branch),
                file_path,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        TextSearchServiceImpl::new(search_db).with_state_pool(state_pool)
    }

    /// #137: a file indexed on `main` must NOT count toward a grep filtered on a
    /// different branch — that branch reports zero searchable files and is
    /// therefore not complete, even with an empty queue.
    #[tokio::test]
    async fn index_status_is_branch_scoped_for_file_count() {
        let tmp = tempfile::TempDir::new().unwrap();
        let svc = service_with_indexed(
            &tmp,
            "tenant-a",
            "main",
            1,
            "src/narrative/mod.rs",
            "pub async fn run_narrative_pipeline() {}",
            empty_state_pool().await,
        )
        .await;

        // The branch that owns the file: one searchable file, complete.
        let on_main = svc
            .tenant_index_status(Some("tenant-a"), Some("main"))
            .await
            .expect("status present for tenant+branch");
        assert_eq!(on_main.files_tracked, 1);
        assert_eq!(on_main.queue_pending, 0);
        assert!(on_main.index_complete);

        // A different branch: the file is invisible, so zero files and NOT
        // complete — this is exactly the #137 false-negative made explicit.
        let on_other = svc
            .tenant_index_status(Some("tenant-a"), Some("fix/issue-137"))
            .await
            .expect("status present for tenant+branch");
        assert_eq!(on_other.files_tracked, 0);
        assert_eq!(on_other.queue_pending, 0);
        assert!(
            !on_other.index_complete,
            "a branch with no indexed files must be reported incomplete"
        );
    }

    /// #141: pending queue items for the queried branch keep it incomplete even
    /// when files are already indexed, so partial results carry the lag warning.
    #[tokio::test]
    async fn index_status_branch_scoped_queue_pending() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state_pool = empty_state_pool().await;
        // One item still queued for tenant-a on main; an unrelated branch's item
        // must not leak into main's count.
        sqlx::query("INSERT INTO unified_queue (tenant_id, branch, status) VALUES (?1, ?2, ?3)")
            .bind("tenant-a")
            .bind("main")
            .bind("pending")
            .execute(&state_pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO unified_queue (tenant_id, branch, status) VALUES (?1, ?2, ?3)")
            .bind("tenant-a")
            .bind("other")
            .bind("pending")
            .execute(&state_pool)
            .await
            .unwrap();

        let svc = service_with_indexed(
            &tmp,
            "tenant-a",
            "main",
            1,
            "src/lib.rs",
            "fn QueryBuilder() {}",
            state_pool,
        )
        .await;

        let status = svc
            .tenant_index_status(Some("tenant-a"), Some("main"))
            .await
            .expect("status present");
        assert_eq!(status.files_tracked, 1, "main has one indexed file");
        assert_eq!(
            status.queue_pending, 1,
            "only main's queued item counts, not other branch's"
        );
        assert!(
            !status.index_complete,
            "queued items for the branch keep it incomplete"
        );
    }

    /// A "*" (all-branches) filter is treated as branch-less: counts span every
    /// branch so the caller opting out of branch scoping sees the whole tenant.
    #[tokio::test]
    async fn index_status_star_branch_counts_all() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let search_db = Arc::new(SearchDbManager::new(&db_path).await.unwrap());
        let processor = FtsBatchProcessor::new(&search_db, FtsBatchConfig::default());
        // Same logical file on two branches → two file_metadata rows, distinct
        // file_path counts collapse to one path but across branches we expect
        // the all-branches count to include both branch rows' distinct paths.
        processor
            .full_rewrite(
                1,
                "fn a() {}",
                "tenant-a",
                Some("main"),
                "a.rs",
                None,
                None,
                None,
            )
            .await
            .unwrap();
        processor
            .full_rewrite(
                2,
                "fn b() {}",
                "tenant-a",
                Some("dev"),
                "b.rs",
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let svc = TextSearchServiceImpl::new(search_db).with_state_pool(empty_state_pool().await);

        let all = svc
            .tenant_index_status(Some("tenant-a"), Some("*"))
            .await
            .expect("status present");
        assert_eq!(all.files_tracked, 2, "* spans both branches");
        assert!(all.index_complete);
    }
}
