//! Core regex search execution: query dispatch and row streaming.

use futures::TryStreamExt;
use sqlx::Row;
use tracing::debug;

use super::query::{build_regex_search_query, fts5_exceeds_threshold};
use crate::search_db::{SearchDbError, SearchDbManager};
use crate::text_search::escaping::{compile_glob_matcher, resolve_path_filter};
use crate::text_search::exact_search::attach_context_lines;
use crate::text_search::regex_parser::{build_fts5_query, extract_literals_from_regex};
use crate::text_search::types::{SearchMatch, SearchOptions, SearchResults};

/// Search code_lines using a regex pattern with trigram acceleration.
///
/// ## Strategy
///
/// 1. Extract literal substrings from the regex for FTS5 pre-filtering
/// 2. Lightweight FTS5-only probe: if candidates exceed threshold, delegate
///    to `grep-searcher` for SIMD-accelerated file scanning
/// 3. Otherwise stream FTS5 candidates, verify with regex in Rust
/// 4. If no extractable literals: full table scan with regex in Rust
///
/// Case-insensitive mode uses `regex::RegexBuilder::case_insensitive(true)`.
/// When `path_glob` is set, applies glob filtering in Rust after SQL results.
pub async fn search_regex(
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

    let literals = extract_literals_from_regex(pattern);
    let fts5_query = build_fts5_query(&literals);
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;
    let re = regex::RegexBuilder::new(pattern)
        .case_insensitive(options.case_insensitive)
        .build()
        .map_err(|e| SearchDbError::InvalidPattern(format!("{}", e)))?;

    debug!(
        "Regex search: pattern={:?}, literals={:?}, fts5_query={:?}, tenant={:?}, path_glob={:?}",
        pattern, literals, fts5_query, effective_options.tenant_id, options.path_glob,
    );

    let pool = search_db.pool();

    // Grep fallback: if FTS5 candidate count exceeds threshold, delegate to
    // grep-searcher for SIMD-accelerated file scanning.
    if let Some(ref fts_q) = fts5_query {
        let threshold = crate::grep_search::GREP_FALLBACK_THRESHOLD;
        if fts5_exceeds_threshold(pool, fts_q, threshold).await? {
            debug!(
                "grep fallback: FTS5 candidates exceed threshold ({}), using grep",
                threshold
            );
            return crate::grep_search::search_regex_via_grep(search_db, pattern, options).await;
        }
    }

    let (matches, truncated, candidates_scanned) = collect_regex_matches(
        pool,
        &fts5_query,
        &effective_options,
        &re,
        glob_matcher.as_ref(),
        options.max_results,
    )
    .await?;

    let mut matches = matches;
    if options.context_lines > 0 {
        attach_context_lines(search_db, &mut matches, options.context_lines).await?;
    }

    let query_time_ms = start.elapsed().as_millis() as u64;
    debug!(
        "Regex search complete: {} matches in {}ms (pattern={:?}, fts_candidates={}, truncated={})",
        matches.len(),
        query_time_ms,
        pattern,
        candidates_scanned,
        truncated
    );

    Ok(SearchResults {
        pattern: pattern.to_string(),
        matches,
        truncated,
        query_time_ms,
        search_engine: "fts5".to_string(),
    })
}

/// Build, bind, and execute the SQL query, then stream rows applying regex
/// verification and optional glob filtering.
///
/// Returns `(matches, truncated, candidates_scanned)`.
pub(super) async fn collect_regex_matches(
    pool: &sqlx::SqlitePool,
    fts5_query: &Option<String>,
    options: &SearchOptions,
    re: &regex::Regex,
    glob_matcher: Option<&impl Fn(&str) -> bool>,
    max_results_hint: usize,
) -> Result<(Vec<SearchMatch>, bool, usize), SearchDbError> {
    let (sql, use_fts) = build_regex_search_query(fts5_query, options);
    let mut query = sqlx::query(&sql);

    if use_fts {
        query = query.bind(fts5_query.as_ref().unwrap());
    }
    if let Some(ref tid) = options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = options.branch {
        if branch != "*" {
            query = query.bind(branch);
        }
    }
    if let Some(ref prefix) = options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let max_results = if max_results_hint > 0 {
        max_results_hint
    } else {
        usize::MAX
    };
    let mut stream = query.fetch(pool);
    let mut matches = Vec::new();
    let mut truncated = false;
    let mut candidates_scanned: usize = 0;

    while let Some(row) = stream.try_next().await? {
        candidates_scanned += 1;
        let file_path: String = row.get("file_path");

        if let Some(matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }

        let content: String = row.get("content");
        if re.is_match(&content) {
            matches.push(SearchMatch {
                line_id: row.get("line_id"),
                file_id: row.get("file_id"),
                line_number: row.get("line_number"),
                content,
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
    }
    drop(stream);

    Ok((matches, truncated, candidates_scanned))
}
