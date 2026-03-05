//! grep-searcher fallback for high-frequency regex patterns.
//!
//! When FTS5 candidate count exceeds [`GREP_FALLBACK_THRESHOLD`], regex search
//! delegates to ripgrep's `grep-searcher` crate instead of streaming SQLite rows.
//! This avoids the ~3us/row SQLite fetch overhead on 10K+ candidate sets by
//! using SIMD-accelerated byte scanning over source files directly.

mod context_sink;
mod query;
mod scanner;
mod types;

#[cfg(test)]
mod tests;

use tracing::debug;

use crate::search_db::{SearchDbError, SearchDbManager};
use crate::text_search::{SearchOptions, SearchResults};

/// FTS5 candidate count above which we fall back to grep-searcher.
///
/// At 5K candidates, FTS5 row-fetch is ~15ms. Above this, grep-searcher's
/// SIMD-accelerated file scanning scales better. Below 5K, FTS5's in-process
/// streaming is faster than file I/O.
pub const GREP_FALLBACK_THRESHOLD: i64 = 5_000;

/// Search source files directly using ripgrep's `grep-searcher` crate.
///
/// Queries `file_metadata` for file paths matching scope filters, then scans
/// each file with `grep-searcher` for regex matches. Context lines are handled
/// natively by grep-searcher (zero additional I/O).
/// Build an empty SearchResults for early-return cases.
fn empty_grep_result(pattern: &str, query_time_ms: u64) -> SearchResults {
    SearchResults {
        pattern: pattern.to_string(),
        matches: vec![],
        truncated: false,
        query_time_ms,
        search_engine: "grep".to_string(),
    }
}

pub async fn search_regex_via_grep(
    search_db: &SearchDbManager,
    pattern: &str,
    options: &SearchOptions,
) -> Result<SearchResults, SearchDbError> {
    let start = std::time::Instant::now();

    if pattern.is_empty() {
        return Ok(empty_grep_result(pattern, 0));
    }

    // Resolve path_glob -> SQL prefix + glob matcher
    let (effective_options, glob_matcher) = query::resolve_and_compile(options)?;

    // Query file_metadata for matching file paths
    let file_paths =
        query::query_file_paths(search_db, &effective_options, glob_matcher.as_ref()).await?;

    debug!(
        "grep search: pattern={:?}, files={}, tenant={:?}, path_glob={:?}",
        pattern,
        file_paths.len(),
        effective_options.tenant_id,
        options.path_glob,
    );

    if file_paths.is_empty() {
        return Ok(empty_grep_result(
            pattern,
            start.elapsed().as_millis() as u64,
        ));
    }

    let max_results = if options.max_results > 0 {
        options.max_results
    } else {
        usize::MAX
    };

    let case_insensitive = options.case_insensitive;
    let context_lines = options.context_lines;
    let pattern_owned = pattern.to_string();

    // Run grep-searcher in a blocking task (it does synchronous file I/O)
    let (matches, truncated) = tokio::task::spawn_blocking(move || {
        scanner::grep_scan_files(
            &pattern_owned,
            case_insensitive,
            context_lines,
            max_results,
            &file_paths,
        )
    })
    .await
    .map_err(|e| {
        SearchDbError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("grep task join error: {}", e),
        ))
    })??;

    let query_time_ms = start.elapsed().as_millis() as u64;

    debug!(
        "grep search complete: {} matches in {}ms (pattern={:?}, truncated={})",
        matches.len(),
        query_time_ms,
        pattern,
        truncated
    );

    Ok(SearchResults {
        pattern: pattern.to_string(),
        matches,
        truncated,
        query_time_ms,
        search_engine: "grep".to_string(),
    })
}
