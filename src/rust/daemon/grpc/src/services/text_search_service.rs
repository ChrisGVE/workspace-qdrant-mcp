//! TextSearchService gRPC implementation
//!
//! Provides FTS5-based code search for the MCP server grep tool and CLI.
//! Exposes 2 RPCs: Search (exact/regex) and CountMatches (count-only).

use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{info, debug, error};
use workspace_qdrant_core::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, search_regex, SearchOptions};

use crate::proto::{
    text_search_service_server::TextSearchService,
    TextSearchRequest, TextSearchResponse, TextSearchCountResponse, TextSearchMatch,
};

/// TextSearchService implementation
pub struct TextSearchServiceImpl {
    search_db: Arc<SearchDbManager>,
}

impl TextSearchServiceImpl {
    /// Create a new TextSearchService with a shared SearchDbManager
    pub fn new(search_db: Arc<SearchDbManager>) -> Self {
        Self { search_db }
    }

    /// Convert a gRPC request into SearchOptions
    fn build_options(req: &TextSearchRequest) -> SearchOptions {
        SearchOptions {
            tenant_id: req.tenant_id.clone(),
            branch: req.branch.clone(),
            path_prefix: req.path_prefix.clone(),
            path_glob: req.path_glob.clone(),
            case_insensitive: !req.case_sensitive,
            max_results: if req.max_results > 0 { req.max_results as usize } else { 1000 },
            context_lines: req.context_lines.max(0) as usize,
        }
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

        debug!("TextSearch: pattern={:?} regex={} case_sensitive={}", req.pattern, req.regex, req.case_sensitive);

        let options = Self::build_options(&req);
        let start = std::time::Instant::now();

        let result = if req.regex {
            search_regex(&self.search_db, &req.pattern, &options).await
        } else {
            search_exact(&self.search_db, &req.pattern, &options).await
        };

        match result {
            Ok(results) => {
                let query_time_ms = start.elapsed().as_millis() as i64;
                let truncated = results.truncated;

                let matches: Vec<TextSearchMatch> = results.matches.into_iter().map(|m| {
                    TextSearchMatch {
                        file_path: m.file_path,
                        line_number: m.line_number as i32,
                        content: m.content,
                        tenant_id: m.tenant_id,
                        branch: m.branch,
                        context_before: m.context_before,
                        context_after: m.context_after,
                    }
                }).collect();

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
            Err(e) => {
                error!("TextSearch failed: {:?}", e);
                Err(Status::internal(format!("Search failed: {}", e)))
            }
        }
    }

    async fn count_matches(
        &self,
        request: Request<TextSearchRequest>,
    ) -> Result<Response<TextSearchCountResponse>, Status> {
        let req = request.into_inner();

        if req.pattern.is_empty() {
            return Err(Status::invalid_argument("Search pattern cannot be empty"));
        }

        debug!("TextSearch count: pattern={:?} regex={}", req.pattern, req.regex);

        // Use max_results=0 to signal we want all matches for counting,
        // but set a high limit to avoid unbounded results
        let mut options = Self::build_options(&req);
        // For count, we don't need context lines
        options.context_lines = 0;
        // Set a very high limit to get the true count
        options.max_results = usize::MAX;

        let start = std::time::Instant::now();

        let result = if req.regex {
            search_regex(&self.search_db, &req.pattern, &options).await
        } else {
            search_exact(&self.search_db, &req.pattern, &options).await
        };

        match result {
            Ok(results) => {
                let query_time_ms = start.elapsed().as_millis() as i64;
                let count = results.matches.len() as i32;

                debug!("TextSearch count completed: {} matches in {}ms", count, query_time_ms);

                Ok(Response::new(TextSearchCountResponse {
                    count,
                    query_time_ms,
                }))
            }
            Err(e) => {
                error!("TextSearch count failed: {:?}", e);
                Err(Status::internal(format!("Count failed: {}", e)))
            }
        }
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
}
