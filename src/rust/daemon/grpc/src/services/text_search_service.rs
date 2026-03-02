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
    search_exact, search_regex, SearchOptions, SearchResults,
};
use workspace_qdrant_core::SearchDbManager;

use crate::proto::{
    text_search_service_server::TextSearchService, TextSearchCountResponse, TextSearchMatch,
    TextSearchRequest, TextSearchResponse,
};

/// Cache TTL — entries older than this are evicted on access.
const CACHE_TTL: Duration = Duration::from_secs(5);

/// Maximum cache entries before forced eviction of all expired entries.
const CACHE_MAX_ENTRIES: usize = 32;

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
}

impl TextSearchServiceImpl {
    /// Create a new TextSearchService with a shared SearchDbManager
    pub fn new(search_db: Arc<SearchDbManager>) -> Self {
        Self {
            search_db,
            cache: Mutex::new(HashMap::new()),
        }
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

    /// Execute the search (exact or regex) or return a cached result.
    async fn execute_or_cached(
        &self,
        req: &TextSearchRequest,
    ) -> Result<SearchResults, Status> {
        let key = CacheKey::from_request(req);

        if let Some(cached) = self.cache_get(&key).await {
            debug!(
                "TextSearch cache hit (key={:#x})",
                key.hash_u64()
            );
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
            max_results: usize::MAX,
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
        let truncated = results.matches.len() > options.max_results;

        // Apply max_results and re-query for context if needed
        let capped: Vec<_> = results
            .matches
            .into_iter()
            .take(options.max_results)
            .collect();

        // If context_lines > 0, we need to fetch context for the matched lines.
        // The cached results were stored without context, so run the search
        // with context only for the capped results. For simplicity and to avoid
        // a separate per-line query, re-run with limits when context is needed.
        let matches: Vec<TextSearchMatch> = if options.context_lines > 0 {
            // Re-run search with context lines and max_results applied
            let result_with_ctx = if req.regex {
                search_regex(&self.search_db, &req.pattern, &options).await
            } else {
                search_exact(&self.search_db, &req.pattern, &options).await
            };
            match result_with_ctx {
                Ok(ctx_results) => ctx_results
                    .matches
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
                    .collect(),
                Err(_) => {
                    // Fallback: return without context
                    capped
                        .into_iter()
                        .map(|m| TextSearchMatch {
                            file_path: m.file_path,
                            line_number: m.line_number as i32,
                            content: m.content,
                            tenant_id: m.tenant_id,
                            branch: m.branch,
                            context_before: vec![],
                            context_after: vec![],
                        })
                        .collect()
                }
            }
        } else {
            capped
                .into_iter()
                .map(|m| TextSearchMatch {
                    file_path: m.file_path,
                    line_number: m.line_number as i32,
                    content: m.content,
                    tenant_id: m.tenant_id,
                    branch: m.branch,
                    context_before: vec![],
                    context_after: vec![],
                })
                .collect()
        };

        let query_time_ms = start.elapsed().as_millis() as i64;
        let total_matches = matches.len() as i32;

        info!(
            "TextSearch completed: {} matches (truncated={}) in {}ms",
            total_matches, truncated, query_time_ms
        );

        Ok(Response::new(TextSearchResponse {
            matches,
            total_matches,
            truncated,
            query_time_ms,
        }))
    }

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
}
