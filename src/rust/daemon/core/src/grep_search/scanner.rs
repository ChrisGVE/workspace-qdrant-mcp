/// Synchronous file scanner using ripgrep's `grep-searcher` crate.
use std::path::Path;

use grep_regex::RegexMatcherBuilder;
use grep_searcher::sinks::UTF8;
use grep_searcher::SearcherBuilder;
use tracing::debug;

use crate::search_db::SearchDbError;
use crate::text_search::SearchMatch;

use super::context_sink::ContextSink;
use super::types::FileInfo;

/// Scan files with grep-searcher, collecting matches.
///
/// Uses `grep_regex::RegexMatcher` for SIMD-accelerated pattern matching
/// and `grep_searcher::Searcher` for context line handling.
pub(super) fn grep_scan_files(
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
