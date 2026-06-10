//! Semantic (hybrid dense+sparse) search support for the TUI Search page (#125).
//!
//! Located at: `src/rust/cli/src/tui/views/search_semantic.rs`
//!
//! Runs the shared `wqm_client::search` pipeline — the same code path the MCP
//! server and `wqm search` use: embeddings via the daemon's EmbeddingService,
//! dense + sparse Qdrant queries, RRF fusion. Called from the Search page's
//! background fetcher thread (its own tokio runtime — never the TUI thread).
//!
//! Neighbors: `search_data.rs` (fetcher + snapshot), `search_page.rs` (mode
//! state), `search_render.rs` (result table),
//! `commands/search/hybrid.rs` (the shared CLI pipeline entry point).

use wqm_client::models::SearchScope;
use wqm_client::search::options::{SearchInput, SearchOptions};

/// One semantic search hit, pre-flattened for table rendering.
#[derive(Debug, Clone)]
pub struct SemanticHit {
    /// RRF-fused relevance score.
    pub score: f64,
    /// Source collection (projects, libraries, rules, scratchpad).
    pub collection: String,
    /// Best locator: title, payload file_path, or point id.
    pub locator: String,
    /// First non-empty content line.
    pub snippet: String,
    /// Full content (for the preview popup).
    pub content: String,
}

/// Result limit for the TUI list (one screenful + scroll headroom).
const TUI_SEMANTIC_LIMIT: usize = 50;

/// Run a hybrid search for the given tenant and map results for the TUI.
pub async fn fetch_semantic(tenant_id: &str, query: &str) -> anyhow::Result<Vec<SemanticHit>> {
    let input = SearchInput {
        query: query.to_string(),
        limit: Some(TUI_SEMANTIC_LIMIT),
        scope: Some(SearchScope::Project),
        ..Default::default()
    };
    let opts = SearchOptions::from_input(input, None);

    let resp = crate::commands::search::hybrid::run_hybrid_search(&opts, Some(tenant_id)).await?;

    if let Some(status) = resp.status {
        // Degraded pipeline (embed failure → scroll fallback) — surface it
        // instead of silently showing unranked results.
        anyhow::bail!(
            "{}{}",
            status,
            resp.status_reason
                .map(|r| format!(": {}", r))
                .unwrap_or_default()
        );
    }

    Ok(resp.results.into_iter().map(to_hit).collect())
}

/// Flatten one pipeline result into a table-ready hit.
fn to_hit(r: wqm_client::models::SearchResult) -> SemanticHit {
    let locator = r
        .title
        .clone()
        .filter(|t| !t.is_empty())
        .or_else(|| {
            r.metadata
                .get("file_path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| r.id.clone());
    let snippet = r
        .content
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string();
    SemanticHit {
        score: r.score,
        collection: r.collection,
        locator,
        snippet,
        content: r.content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn result(
        title: Option<&str>,
        file_path: Option<&str>,
        content: &str,
    ) -> wqm_client::models::SearchResult {
        let mut metadata = HashMap::new();
        if let Some(fp) = file_path {
            metadata.insert("file_path".to_string(), serde_json::json!(fp));
        }
        wqm_client::models::SearchResult {
            id: "pt-9".into(),
            score: 0.7,
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
    fn to_hit_prefers_title() {
        let h = to_hit(result(Some("Doc"), Some("a.rs"), "body"));
        assert_eq!(h.locator, "Doc");
    }

    #[test]
    fn to_hit_falls_back_to_file_path_then_id() {
        assert_eq!(to_hit(result(None, Some("a.rs"), "x")).locator, "a.rs");
        assert_eq!(to_hit(result(None, None, "x")).locator, "pt-9");
    }

    #[test]
    fn to_hit_snippet_is_first_non_empty_line() {
        let h = to_hit(result(None, None, "\n\n  fn main() {\n}"));
        assert_eq!(h.snippet, "fn main() {");
    }
}
