//! Per-tenant symbol automaton for narrative EXPLAINS resolution.
//!
//! Builds an Aho-Corasick automaton over a tenant's REAL code-graph symbol
//! names so narrative extractors can resolve documentation references to
//! existing graph node ids — never to orphaned stub nodes with empty paths.
//!
//! A symbol name that maps to exactly one code node resolves to that node's
//! real id; names with zero or multiple matches are dropped by the consumer
//! (ambiguous references must not invent edges).

use std::collections::HashMap;
use std::sync::Arc;

use aho_corasick::{AhoCorasick, MatchKind};
use tokio::sync::Mutex;

use crate::graph::{GraphStore, SymbolRow};

/// Immutable, shareable index of a tenant's code symbols plus an Aho-Corasick
/// automaton for scanning narrative text for references to them.
///
/// Cheap to clone (everything is behind `Arc`). Built once per batch via
/// [`SymbolAutomaton::build`] and reused across every file in that batch.
#[derive(Clone)]
pub struct SymbolAutomaton {
    /// Aho-Corasick automaton over `patterns` (None when there are no symbols).
    automaton: Option<Arc<AhoCorasick>>,
    /// Distinct symbol names, indexed by Aho-Corasick pattern index.
    patterns: Arc<Vec<String>>,
    /// symbol_name → resolved code node ids. A name resolves only when it maps
    /// to exactly one node id (ambiguous names are dropped by the consumer).
    resolution: Arc<HashMap<String, Vec<String>>>,
}

impl SymbolAutomaton {
    /// Build an automaton from a tenant's resolved code symbols.
    ///
    /// Symbols shorter than `min_symbol_length` are excluded (too noisy to
    /// match reliably in prose). Duplicate names collapse to a single pattern
    /// whose resolution carries every node id sharing that name.
    pub fn build(symbols: &[SymbolRow], min_symbol_length: usize) -> Self {
        let mut resolution: HashMap<String, Vec<String>> = HashMap::new();
        for s in symbols {
            if s.symbol_name.len() < min_symbol_length {
                continue;
            }
            let ids = resolution.entry(s.symbol_name.clone()).or_default();
            if !ids.contains(&s.node_id) {
                ids.push(s.node_id.clone());
            }
        }

        let patterns: Vec<String> = resolution.keys().cloned().collect();

        let automaton = if patterns.is_empty() {
            None
        } else {
            // LeftmostLongest favors the longest symbol at each position, so
            // `HttpServer` wins over a hypothetical `Http` substring symbol.
            AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&patterns)
                .ok()
                .map(Arc::new)
        };

        Self {
            automaton,
            patterns: Arc::new(patterns),
            resolution: Arc::new(resolution),
        }
    }

    /// An empty automaton that matches nothing (no symbols available).
    pub fn empty() -> Self {
        Self {
            automaton: None,
            patterns: Arc::new(Vec::new()),
            resolution: Arc::new(HashMap::new()),
        }
    }

    /// Number of distinct symbol names in the automaton.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Whether the automaton holds no symbols.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Find all symbol matches in `content`.
    ///
    /// Returns `(symbol_name, byte_offset_of_match_start)` for every match,
    /// in scan order. The byte offset lets the consumer map a match to the
    /// section whose line range contains it.
    pub fn find_matches(&self, content: &str) -> Vec<(String, usize)> {
        let Some(ref ac) = self.automaton else {
            return Vec::new();
        };
        ac.find_iter(content)
            .map(|m| (self.patterns[m.pattern().as_usize()].clone(), m.start()))
            .collect()
    }

    /// Resolve a symbol name to a single code node id.
    ///
    /// Returns `Some(node_id)` only when exactly one node bears the name.
    /// Zero matches (unknown symbol) or more than one (ambiguous) yield `None`,
    /// so the caller drops the edge rather than inventing a target.
    pub fn resolve_unique(&self, symbol_name: &str) -> Option<&str> {
        match self.resolution.get(symbol_name) {
            Some(ids) if ids.len() == 1 => Some(ids[0].as_str()),
            _ => None,
        }
    }
}

impl Default for SymbolAutomaton {
    fn default() -> Self {
        Self::empty()
    }
}

/// Per-tenant cache of built [`SymbolAutomaton`]s.
///
/// The automaton is rebuilt only when a tenant's code-symbol count changes,
/// so successive files in the same batch reuse the same automaton (PERF-2:
/// build once per batch, not per file). The symbol count is a cheap, monotone
/// proxy for "the symbol set changed" — adequate because narrative extraction
/// runs after the code-graph phase writes the file's own symbols, so a new
/// file's symbols bump the count and trigger a rebuild.
#[derive(Default)]
pub struct AutomatonCache {
    inner: Mutex<HashMap<String, CachedAutomaton>>,
}

struct CachedAutomaton {
    /// Symbol count the automaton was built from (invalidation key).
    symbol_count: usize,
    automaton: SymbolAutomaton,
}

impl AutomatonCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the tenant's symbol automaton, rebuilding it from `graph_store`
    /// only when the tenant's symbol count has changed since the last build.
    ///
    /// On a graph-store query error an empty automaton is returned (narrative
    /// EXPLAINS resolution then yields no edges for this file).
    pub async fn get_or_build(
        &self,
        graph_store: &Arc<dyn GraphStore>,
        tenant_id: &str,
        min_symbol_length: usize,
    ) -> SymbolAutomaton {
        let symbols = match graph_store.query_code_symbols(tenant_id).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "AutomatonCache: failed to query code symbols for tenant {tenant_id}: {e}"
                );
                return SymbolAutomaton::empty();
            }
        };
        let count = symbols.len();

        let mut guard = self.inner.lock().await;
        if let Some(cached) = guard.get(tenant_id) {
            if cached.symbol_count == count {
                return cached.automaton.clone();
            }
        }
        let automaton = SymbolAutomaton::build(&symbols, min_symbol_length);
        guard.insert(
            tenant_id.to_string(),
            CachedAutomaton {
                symbol_count: count,
                automaton: automaton.clone(),
            },
        );
        automaton
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str, file: &str) -> SymbolRow {
        SymbolRow {
            symbol_name: name.to_string(),
            node_id: format!("{file}:{name}"),
            file_path: file.to_string(),
        }
    }

    #[test]
    fn builds_from_symbols_and_resolves_unique() {
        let symbols = vec![
            sym("validate_token", "auth.rs"),
            sym("HttpServer", "net.rs"),
        ];
        let auto = SymbolAutomaton::build(&symbols, 4);
        assert_eq!(auto.len(), 2);
        assert!(!auto.is_empty());
        assert_eq!(
            auto.resolve_unique("validate_token"),
            Some("auth.rs:validate_token")
        );
        assert_eq!(auto.resolve_unique("HttpServer"), Some("net.rs:HttpServer"));
        assert_eq!(auto.resolve_unique("unknown"), None);
    }

    #[test]
    fn ambiguous_symbol_does_not_resolve() {
        // Two distinct nodes share the same name.
        let symbols = vec![sym("handler", "a.rs"), sym("handler", "b.rs")];
        let auto = SymbolAutomaton::build(&symbols, 4);
        assert_eq!(auto.len(), 1);
        assert_eq!(auto.resolve_unique("handler"), None);
    }

    #[test]
    fn min_length_filter_excludes_short_names() {
        let symbols = vec![sym("io", "x.rs"), sym("parse_config", "y.rs")];
        let auto = SymbolAutomaton::build(&symbols, 4);
        assert_eq!(auto.len(), 1);
        assert_eq!(auto.resolve_unique("io"), None);
        assert_eq!(
            auto.resolve_unique("parse_config"),
            Some("y.rs:parse_config")
        );
    }

    #[test]
    fn find_matches_reports_offsets() {
        let symbols = vec![sym("parse_config", "y.rs")];
        let auto = SymbolAutomaton::build(&symbols, 4);
        let content = "Use parse_config first, then parse_config again.";
        let matches = auto.find_matches(content);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].0, "parse_config");
        assert_eq!(matches[0].1, 4);
        assert!(matches[1].1 > matches[0].1);
    }

    #[test]
    fn empty_automaton_matches_nothing() {
        let auto = SymbolAutomaton::empty();
        assert!(auto.is_empty());
        assert!(auto.find_matches("parse_config parse_config").is_empty());
        assert_eq!(auto.resolve_unique("parse_config"), None);
    }
}
