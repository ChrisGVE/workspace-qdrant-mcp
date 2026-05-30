//! parse_args permissive parsing tests (TS parity).
//!
//! Included from `search_tests_score.rs` via
//! `#[path = "search_tests_score_part2.rs"] mod parse_args_tests;`.

use serde_json::Value;

// ---------------------------------------------------------------------------
// parse_args permissive parsing (TS parity)
// ---------------------------------------------------------------------------

#[test]
fn parse_args_missing_query_defaults_to_empty_string() {
    // TS buildSearchOptions (search.ts:130): `query: (args?.['query'] as string) ?? ''`
    // When 'query' is absent, query defaults to ''.
    use crate::tools::search::options::SearchOptions;
    use serde_json::Map;
    let args: Map<String, Value> = Map::new();
    let input = SearchOptions::parse_args(&args).expect("missing query must not error");
    assert_eq!(input.query, "", "missing query must default to ''");
}

#[test]
fn parse_args_unrecognized_mode_silently_dropped() {
    // TS: mode only set when value is 'hybrid'|'semantic'|'keyword'.
    // Unknown values like 'fuzzy' must be silently ignored (mode=None → defaults to Hybrid).
    use crate::tools::search::options::SearchOptions;
    use serde_json::{json, Map};
    let mut args: Map<String, Value> = Map::new();
    args.insert("query".to_string(), json!("test"));
    args.insert("mode".to_string(), json!("fuzzy")); // unknown
    let input = SearchOptions::parse_args(&args).expect("unknown mode must not error");
    assert!(
        input.mode.is_none(),
        "unknown mode must be dropped (got {:?})",
        input.mode
    );
}

#[test]
fn parse_args_unrecognized_scope_silently_dropped() {
    // TS: scope only set when value is 'project'|'group'|'all'.
    // Unknown values like 'team' must be silently ignored (scope=None → defaults to Project).
    use crate::tools::search::options::SearchOptions;
    use serde_json::{json, Map};
    let mut args: Map<String, Value> = Map::new();
    args.insert("query".to_string(), json!("q"));
    args.insert("scope".to_string(), json!("team")); // unknown
    let input = SearchOptions::parse_args(&args).expect("unknown scope must not error");
    assert!(
        input.scope.is_none(),
        "unknown scope must be dropped (got {:?})",
        input.scope
    );
}

#[test]
fn parse_args_known_mode_and_scope_parsed_correctly() {
    use crate::tools::search::options::SearchOptions;
    use crate::tools::search::types::{SearchMode, SearchScope};
    use serde_json::{json, Map};
    let mut args: Map<String, Value> = Map::new();
    args.insert("query".to_string(), json!("hello"));
    args.insert("mode".to_string(), json!("semantic"));
    args.insert("scope".to_string(), json!("all"));
    let input = SearchOptions::parse_args(&args).expect("valid args must parse");
    assert_eq!(input.query, "hello");
    assert_eq!(input.mode, Some(SearchMode::Semantic));
    assert_eq!(input.scope, Some(SearchScope::All));
}
