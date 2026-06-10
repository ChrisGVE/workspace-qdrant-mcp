//! Terminal rendering for `wqm search` results (#125).
//!
//! Located at: `src/rust/cli/src/commands/search/render.rs`
//!
//! Formats a `SearchResponse` from the shared hybrid pipeline into the CLI's
//! standard key/value + list output.
//!
//! Neighbors: `mod.rs` (subcommand dispatch), `hybrid.rs` (pipeline glue).

use wqm_client::models::{SearchResponse, SearchResult};

use crate::output;

/// Print a full search response: header, results, or a degraded-status note.
pub fn print_response(resp: &SearchResponse) {
    if let Some(ref status) = resp.status {
        // Degraded or refused search (e.g. daemon fallback, group refusal).
        output::warning(format!(
            "Status: {}{}",
            status,
            resp.status_reason
                .as_deref()
                .map(|r| format!(" — {}", r))
                .unwrap_or_default()
        ));
    }

    if resp.results.is_empty() {
        output::info("No results.");
        return;
    }

    output::kv("Results", resp.results.len().to_string());
    if let Some(ref b) = resp.branch {
        output::kv("Branch", b);
    }
    output::kv("Collections", resp.collections_searched.join(", "));
    output::separator();

    for (i, r) in resp.results.iter().enumerate() {
        print_result(i + 1, r);
    }
}

/// Print one result: rank, score, collection, locator line, snippet.
fn print_result(rank: usize, r: &SearchResult) {
    output::info(format!(
        "{:>3}. [{:.3}] ({}) {}",
        rank,
        r.score,
        r.collection,
        result_locator(r)
    ));
    let snippet = first_snippet_line(&r.content);
    if !snippet.is_empty() {
        output::info(format!("     {}", snippet));
    }
}

/// Best human-readable locator for a result: title, then payload file_path,
/// then the point id.
fn result_locator(r: &SearchResult) -> String {
    if let Some(ref t) = r.title {
        if !t.is_empty() {
            return t.clone();
        }
    }
    if let Some(fp) = r.metadata.get("file_path").and_then(|v| v.as_str()) {
        if !fp.is_empty() {
            return fp.to_string();
        }
    }
    r.id.clone()
}

/// First non-empty line of the content, truncated for terminal display.
fn first_snippet_line(content: &str) -> String {
    const MAX: usize = 120;
    let line = content
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if line.chars().count() > MAX {
        let truncated: String = line.chars().take(MAX).collect();
        format!("{}…", truncated)
    } else {
        line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn result_with(title: Option<&str>, file_path: Option<&str>, content: &str) -> SearchResult {
        let mut metadata = HashMap::new();
        if let Some(fp) = file_path {
            metadata.insert("file_path".to_string(), serde_json::json!(fp));
        }
        SearchResult {
            id: "point-1".into(),
            score: 0.42,
            collection: "projects".into(),
            content: content.into(),
            title: title.map(str::to_string),
            metadata,
            provenance: None,
            parent_context: None,
            graph_context: None,
        }
    }

    #[test]
    fn locator_prefers_title() {
        let r = result_with(Some("My Doc"), Some("src/a.rs"), "x");
        assert_eq!(result_locator(&r), "My Doc");
    }

    #[test]
    fn locator_falls_back_to_file_path() {
        let r = result_with(None, Some("src/a.rs"), "x");
        assert_eq!(result_locator(&r), "src/a.rs");
    }

    #[test]
    fn locator_falls_back_to_id() {
        let r = result_with(None, None, "x");
        assert_eq!(result_locator(&r), "point-1");
    }

    #[test]
    fn snippet_takes_first_non_empty_line() {
        assert_eq!(first_snippet_line("\n\n  hello\nworld"), "hello");
    }

    #[test]
    fn snippet_truncates_long_lines() {
        let long = "x".repeat(200);
        let s = first_snippet_line(&long);
        assert!(s.chars().count() <= 121); // 120 + ellipsis
        assert!(s.ends_with('…'));
    }
}
