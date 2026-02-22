//! grep-searcher fallback for high-frequency regex patterns.
//!
//! When FTS5 candidate count exceeds [`GREP_FALLBACK_THRESHOLD`], regex search
//! delegates to ripgrep's `grep-searcher` crate instead of streaming SQLite rows.
//! This avoids the ~3us/row SQLite fetch overhead on 10K+ candidate sets by
//! using SIMD-accelerated byte scanning over source files directly.

use std::path::Path;

use grep_regex::RegexMatcherBuilder;
use grep_searcher::sinks::UTF8;
use grep_searcher::{Searcher, SearcherBuilder, Sink, SinkContextKind};
use sqlx::Row;
use tracing::debug;

use crate::search_db::{SearchDbManager, SearchDbError};
use crate::text_search::{
    compile_glob_matcher, resolve_path_filter, SearchMatch, SearchOptions, SearchResults,
};

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
pub async fn search_regex_via_grep(
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
            search_engine: "grep".to_string(),
        });
    }

    // Resolve path_glob -> SQL prefix + glob matcher
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;

    // Query file_metadata for matching file paths
    let file_paths = query_file_paths(search_db, &effective_options, glob_matcher.as_ref()).await?;

    debug!(
        "grep search: pattern={:?}, files={}, tenant={:?}, path_glob={:?}",
        pattern,
        file_paths.len(),
        effective_options.tenant_id,
        options.path_glob,
    );

    if file_paths.is_empty() {
        return Ok(SearchResults {
            pattern: pattern.to_string(),
            matches: vec![],
            truncated: false,
            query_time_ms: start.elapsed().as_millis() as u64,
            search_engine: "grep".to_string(),
        });
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
        grep_scan_files(
            &pattern_owned,
            case_insensitive,
            context_lines,
            max_results,
            &file_paths,
        )
    })
    .await
    .map_err(|e| {
        SearchDbError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("grep task join error: {}", e)))
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

/// File info from file_metadata needed for grep scanning.
#[derive(Debug, Clone)]
struct FileInfo {
    file_path: String,
    tenant_id: String,
    branch: Option<String>,
}

/// Query file_metadata for file paths matching scope filters.
async fn query_file_paths(
    search_db: &SearchDbManager,
    options: &SearchOptions,
    glob_matcher: Option<&Box<dyn Fn(&str) -> bool + Send + Sync>>,
) -> Result<Vec<FileInfo>, SearchDbError> {
    let mut sql = String::from(
        "SELECT file_path, tenant_id, branch FROM file_metadata WHERE 1=1",
    );
    let mut next_param = 1;

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(" AND file_path LIKE ?{} ESCAPE '\\'", next_param));
    }

    sql.push_str(" ORDER BY file_path");

    let pool = search_db.pool();
    let mut query = sqlx::query(&sql);

    if let Some(ref tid) = options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = options.branch {
        query = query.bind(branch);
    }
    if let Some(ref prefix) = options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let rows = query.fetch_all(pool).await?;

    let mut files = Vec::with_capacity(rows.len());
    for row in rows {
        let file_path: String = row.get("file_path");
        // Apply glob filter
        if let Some(matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }
        files.push(FileInfo {
            file_path,
            tenant_id: row.get("tenant_id"),
            branch: row.get("branch"),
        });
    }

    Ok(files)
}

/// Scan files with grep-searcher, collecting matches.
///
/// Uses `grep_regex::RegexMatcher` for SIMD-accelerated pattern matching
/// and `grep_searcher::Searcher` for context line handling.
fn grep_scan_files(
    pattern: &str,
    case_insensitive: bool,
    context_lines: usize,
    max_results: usize,
    files: &[FileInfo],
) -> Result<(Vec<SearchMatch>, bool), SearchDbError> {
    let matcher = RegexMatcherBuilder::new()
        .case_insensitive(case_insensitive)
        .build(pattern)
        .map_err(|e| SearchDbError::InvalidPattern(format!("{}", e)))?;

    let mut searcher = SearcherBuilder::new()
        .line_number(true)
        .before_context(context_lines)
        .after_context(context_lines)
        .build();

    let mut all_matches: Vec<SearchMatch> = Vec::new();
    let mut truncated = false;

    for file_info in files {
        let path = Path::new(&file_info.file_path);

        // Skip files that don't exist (may have been deleted since indexing)
        if !path.exists() {
            continue;
        }

        if context_lines > 0 {
            // With context: use custom Sink that tracks before/after lines
            let mut sink = ContextSink::new(file_info, context_lines);
            let result = searcher.search_path(&matcher, path, &mut sink);
            match result {
                Ok(()) => {}
                Err(e) => {
                    debug!("grep: skipping {}: {}", file_info.file_path, e);
                    continue;
                }
            }

            // Drain collected matches
            for m in sink.finish_collecting() {
                all_matches.push(m);
                if all_matches.len() >= max_results {
                    truncated = true;
                    break;
                }
            }
        } else {
            // No context: simple UTF8 sink
            let result = searcher.search_path(
                &matcher,
                path,
                UTF8(|line_number, content| {
                    all_matches.push(SearchMatch {
                        line_id: 0,
                        file_id: 0,
                        line_number: line_number as i64,
                        content: content.trim_end_matches('\n').to_string(),
                        file_path: file_info.file_path.clone(),
                        tenant_id: file_info.tenant_id.clone(),
                        branch: file_info.branch.clone(),
                        context_before: vec![],
                        context_after: vec![],
                    });

                    if all_matches.len() >= max_results {
                        truncated = true;
                        return Ok(false);
                    }
                    Ok(true)
                }),
            );

            if let Err(e) = result {
                debug!("grep: skipping {}: {}", file_info.file_path, e);
            }
        }

        if truncated {
            break;
        }
    }

    Ok((all_matches, truncated))
}

// ────────────────────────────────────────────────────────────────────────────
// Context-aware Sink implementation
// ────────────────────────────────────────────────────────────────────────────

/// Custom `Sink` that collects matches with before/after context lines.
///
/// grep-searcher interleaves context and match callbacks:
///   context(Before) → matched → context(After) → context_break → ...
///
/// We buffer context-before lines and attach context-after to the previous
/// match as they arrive.
struct ContextSink {
    file_path: String,
    tenant_id: String,
    branch: Option<String>,
    context_lines: usize,
    matches: Vec<SearchMatch>,
    /// Pending context-before lines for the next match.
    pending_before: Vec<String>,
    /// How many context-after lines we've added to the last match.
    after_count: usize,
}

impl ContextSink {
    fn new(file_info: &FileInfo, context_lines: usize) -> Self {
        Self {
            file_path: file_info.file_path.clone(),
            tenant_id: file_info.tenant_id.clone(),
            branch: file_info.branch.clone(),
            context_lines,
            matches: Vec::new(),
            pending_before: Vec::new(),
            after_count: 0,
        }
    }

    /// Consume and return all collected matches.
    fn finish_collecting(self) -> Vec<SearchMatch> {
        self.matches
    }
}

impl Sink for ContextSink {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &Searcher,
        mat: &grep_searcher::SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        let content = String::from_utf8_lossy(mat.bytes())
            .trim_end_matches('\n')
            .to_string();
        let line_number = mat.line_number().unwrap_or(0) as i64;

        let context_before = std::mem::take(&mut self.pending_before);
        self.after_count = 0;

        self.matches.push(SearchMatch {
            line_id: 0,
            file_id: 0,
            line_number,
            content,
            file_path: self.file_path.clone(),
            tenant_id: self.tenant_id.clone(),
            branch: self.branch.clone(),
            context_before,
            context_after: vec![],
        });

        Ok(true)
    }

    fn context(
        &mut self,
        _searcher: &Searcher,
        context: &grep_searcher::SinkContext<'_>,
    ) -> Result<bool, Self::Error> {
        let line = String::from_utf8_lossy(context.bytes())
            .trim_end_matches('\n')
            .to_string();

        match context.kind() {
            &SinkContextKind::Before => {
                self.pending_before.push(line);
                if self.pending_before.len() > self.context_lines {
                    self.pending_before.remove(0);
                }
            }
            &SinkContextKind::After => {
                if let Some(last) = self.matches.last_mut() {
                    if self.after_count < self.context_lines {
                        last.context_after.push(line);
                        self.after_count += 1;
                    }
                }
            }
            &SinkContextKind::Other => {
                self.pending_before.push(line);
                if self.pending_before.len() > self.context_lines {
                    self.pending_before.remove(0);
                }
            }
        }

        Ok(true)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Create temp files with known content and return their paths.
    fn setup_test_files(dir: &TempDir) -> Vec<String> {
        let file1 = dir.path().join("test1.rs");
        fs::write(
            &file1,
            "fn main() {\n    println!(\"hello\");\n    let x = 42;\n}\n",
        )
        .unwrap();

        let file2 = dir.path().join("test2.rs");
        fs::write(
            &file2,
            "use std::io;\nfn read_input() {\n    io::stdin();\n}\n",
        )
        .unwrap();

        vec![
            file1.to_string_lossy().to_string(),
            file2.to_string_lossy().to_string(),
        ]
    }

    fn make_file_infos(paths: &[String]) -> Vec<FileInfo> {
        paths
            .iter()
            .map(|p| FileInfo {
                file_path: p.clone(),
                tenant_id: "test-tenant".to_string(),
                branch: Some("main".to_string()),
            })
            .collect()
    }

    #[test]
    fn test_grep_scan_simple_match() {
        let dir = TempDir::new().unwrap();
        let paths = setup_test_files(&dir);
        let files = make_file_infos(&paths);

        let (matches, truncated) =
            grep_scan_files("println", false, 0, 100, &files).unwrap();

        assert!(!truncated);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].content.contains("println"));
        assert_eq!(matches[0].line_number, 2);
        assert_eq!(matches[0].tenant_id, "test-tenant");
    }

    #[test]
    fn test_grep_scan_regex_match() {
        let dir = TempDir::new().unwrap();
        let paths = setup_test_files(&dir);
        let files = make_file_infos(&paths);

        let (matches, _) =
            grep_scan_files(r"fn\s+\w+", false, 0, 100, &files).unwrap();

        assert_eq!(matches.len(), 2); // fn main, fn read_input
    }

    #[test]
    fn test_grep_scan_case_insensitive() {
        let dir = TempDir::new().unwrap();
        let paths = setup_test_files(&dir);
        let files = make_file_infos(&paths);

        let (matches, _) =
            grep_scan_files("PRINTLN", true, 0, 100, &files).unwrap();

        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_grep_scan_max_results_truncation() {
        let dir = TempDir::new().unwrap();
        let paths = setup_test_files(&dir);
        let files = make_file_infos(&paths);

        let (matches, truncated) =
            grep_scan_files(r"fn|let|use", false, 0, 2, &files).unwrap();

        assert_eq!(matches.len(), 2);
        assert!(truncated);
    }

    #[test]
    fn test_grep_scan_no_matches() {
        let dir = TempDir::new().unwrap();
        let paths = setup_test_files(&dir);
        let files = make_file_infos(&paths);

        let (matches, truncated) =
            grep_scan_files("NONEXISTENT_PATTERN_XYZ", false, 0, 100, &files).unwrap();

        assert!(matches.is_empty());
        assert!(!truncated);
    }

    #[test]
    fn test_grep_scan_missing_file_skipped() {
        let files = vec![FileInfo {
            file_path: "/nonexistent/file.rs".to_string(),
            tenant_id: "test".to_string(),
            branch: None,
        }];

        let (matches, _) =
            grep_scan_files("pattern", false, 0, 100, &files).unwrap();

        assert!(matches.is_empty());
    }

    #[test]
    fn test_grep_scan_empty_file_list() {
        let (matches, truncated) =
            grep_scan_files("pattern", false, 0, 100, &[]).unwrap();

        assert!(matches.is_empty());
        assert!(!truncated);
    }

    #[test]
    fn test_grep_scan_with_context_lines() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("ctx.rs");
        fs::write(
            &file,
            "line1\nline2\ntarget_line\nline4\nline5\n",
        )
        .unwrap();

        let files = vec![FileInfo {
            file_path: file.to_string_lossy().to_string(),
            tenant_id: "test".to_string(),
            branch: Some("main".to_string()),
        }];

        let (matches, _) =
            grep_scan_files("target_line", false, 2, 100, &files).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line_number, 3);
        assert_eq!(matches[0].context_before, vec!["line1", "line2"]);
        assert_eq!(matches[0].context_after, vec!["line4", "line5"]);
    }
}
