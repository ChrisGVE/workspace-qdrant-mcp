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

// ---------------------------------------------------------------------------
// resolve_project_id precedence (GitHub #83): explicit > session > cwd-detect
// ---------------------------------------------------------------------------

#[test]
fn resolve_project_id_precedence_explicit_then_session_then_cwd() {
    use crate::server_types::SessionState;
    use crate::sqlite::{SharedStateManager, StateManager};
    use crate::tools::search::resolve_project_id;

    // Degraded state manager (path does not exist → no SQLite connection), so
    // the cwd-detection fallback resolves no tenant and returns None.
    let state = SharedStateManager::new(StateManager::open_at(
        "/nonexistent/wqm-resolve-project-id-test.db",
    ));

    let opts = crate::tools::search::options::SearchOptions {
        project_id: Some("explicit-pid".to_string()),
        ..super::super::opts_hybrid("q", 5)
    };
    let mut session = SessionState::new();
    session.project_id = Some("session-pid".to_string());

    // 1. Explicit opts.project_id wins over session and cwd.
    assert_eq!(
        resolve_project_id(&opts, &session, &state),
        Some("explicit-pid".to_string())
    );

    // 2. No explicit id → session.project_id is used.
    let opts_no_explicit = crate::tools::search::options::SearchOptions {
        project_id: None,
        ..super::super::opts_hybrid("q", 5)
    };
    assert_eq!(
        resolve_project_id(&opts_no_explicit, &session, &state),
        Some("session-pid".to_string())
    );

    // 3. Neither explicit nor session → cwd detection (degraded → None).
    let empty_session = SessionState::new();
    assert_eq!(
        resolve_project_id(&opts_no_explicit, &empty_session, &state),
        None
    );
}

#[test]
fn resolve_cwd_project_id_longest_prefix_picks_deeper_markerless_tenant() {
    use crate::sqlite::{SharedStateManager, StateManager};
    use crate::tools::search::resolve_cwd_project_id_locked;

    // Two registered projects, one nested inside the other, where the deeper
    // one has NO filesystem project marker (registration accepts raw paths).
    // cwd-direct longest-prefix matching must resolve a cwd under the nested
    // project to the DEEPER tenant — never the ancestor. A marker-based
    // project-root walk would skip the markerless nested dir and resolve the
    // wrong (ancestor) tenant. Regression guard for GitHub #83 (audit R2).
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path().join("repo");
    let nest = root.join("sandbox"); // deliberately NO marker file created
    let cwd = nest.join("src");
    std::fs::create_dir_all(&cwd).unwrap();

    let db_path = dir.path().join("state.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE watch_folders (
             tenant_id TEXT NOT NULL,
             path TEXT NOT NULL,
             collection TEXT NOT NULL DEFAULT 'projects'
         )",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_ROOT', ?1, 'projects')",
        rusqlite::params![root.to_str().unwrap()],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_NEST', ?1, 'projects')",
        rusqlite::params![nest.to_str().unwrap()],
    )
    .unwrap();
    drop(conn);

    let state = SharedStateManager::new(StateManager::open_at(&db_path));

    // cwd under the markerless nested project → deeper tenant.
    assert_eq!(
        resolve_cwd_project_id_locked(&cwd, &state),
        Some("T_NEST".to_string())
    );
    // The registered root itself → root tenant.
    assert_eq!(
        resolve_cwd_project_id_locked(&root, &state),
        Some("T_ROOT".to_string())
    );
    // An unrelated dir → no match.
    let other = tempfile::TempDir::new().unwrap();
    assert_eq!(resolve_cwd_project_id_locked(other.path(), &state), None);
}
