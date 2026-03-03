//! Main entry point for exact substring search.
//!
//! Orchestrates FTS5 trigram pre-filtering with INSTR verification,
//! scope binding, glob post-filtering, and optional context attachment.

use futures::TryStreamExt;
use sqlx::Row;
use tracing::debug;

use crate::search_db::{SearchDbManager, SearchDbError};
use super::super::escaping::{escape_fts5_pattern, compile_glob_matcher, resolve_path_filter};
use super::super::types::{SearchMatch, SearchOptions, SearchResults};
use super::context::attach_context_lines;
use super::query_builder::build_search_query;

/// Search code_lines for an exact substring pattern.
///
/// Uses a two-phase approach:
/// 1. FTS5 trigram MATCH for fast candidate selection
/// 2. INSTR verification for exact substring match
///
/// For patterns shorter than 3 characters, falls back to INSTR-only scan.
/// When `path_glob` is set, applies glob filtering in Rust after SQL results.
pub async fn search_exact(
    search_db: &SearchDbManager,
    pattern: &str,
    options: &SearchOptions,
) -> Result<SearchResults, SearchDbError> {
    let start = std::time::Instant::now();

    if pattern.is_empty() {
        return Ok(SearchResults {
            pattern: pattern.to_string(),
            matches: vec![],
            truncated: false,
            query_time_ms: 0,
            search_engine: "fts5".to_string(),
        });
    }

    // Resolve path_glob -> SQL prefix + glob matcher
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;

    let fts5_pattern = escape_fts5_pattern(pattern);

    // Build the SQL query dynamically based on options
    let (sql, use_fts) = build_search_query(&fts5_pattern, &effective_options);

    debug!(
        "FTS5 search: pattern={:?}, fts5={:?}, use_fts={}, tenant={:?}, branch={:?}, path_prefix={:?}, path_glob={:?}",
        pattern, fts5_pattern, use_fts,
        effective_options.tenant_id, effective_options.branch, effective_options.path_prefix,
        options.path_glob,
    );

    let pool = search_db.pool();
    let mut query = sqlx::query(&sql);

    // Bind parameters in order
    if use_fts {
        query = query.bind(fts5_pattern.as_ref().unwrap());
    }

    // INSTR pattern — case insensitive uses LOWER() in SQL so bind lowercase
    if options.case_insensitive {
        query = query.bind(pattern.to_lowercase());
    } else {
        query = query.bind(pattern);
    }

    // Bind optional scope filters
    if let Some(ref tid) = effective_options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = effective_options.branch {
        query = query.bind(branch);
    }
    if let Some(ref prefix) = effective_options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let max_results = if options.max_results > 0 {
        options.max_results
    } else {
        usize::MAX
    };

    let mut stream = query.fetch(pool);
    let mut matches = Vec::new();
    let mut truncated = false;

    while let Some(row) = stream.try_next().await? {
        let file_path: String = row.get("file_path");

        // Apply glob filter if set
        if let Some(ref matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }

        matches.push(SearchMatch {
            line_id: row.get("line_id"),
            file_id: row.get("file_id"),
            line_number: row.get("line_number"),
            content: row.get("content"),
            file_path,
            tenant_id: row.get("tenant_id"),
            branch: row.get("branch"),
            context_before: vec![],
            context_after: vec![],
        });

        if matches.len() >= max_results {
            truncated = true;
            break;
        }
    }
    // Drop the stream to release the connection before context queries
    drop(stream);

    // Attach context lines if requested
    if options.context_lines > 0 {
        attach_context_lines(search_db, &mut matches, options.context_lines).await?;
    }

    let query_time_ms = start.elapsed().as_millis() as u64;

    debug!(
        "FTS5 search complete: {} matches in {}ms (pattern={:?}, truncated={})",
        matches.len(), query_time_ms, pattern, truncated
    );

    Ok(SearchResults {
        pattern: pattern.to_string(),
        matches,
        truncated,
        query_time_ms,
        search_engine: "fts5".to_string(),
    })
}
