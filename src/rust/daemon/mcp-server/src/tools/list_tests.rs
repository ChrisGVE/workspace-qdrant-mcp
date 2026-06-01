//! Hermetic tests for the `list` MCP tool.
//!
//! Each test spins up an ephemeral WAL SQLite database, writes rows through
//! a writable connection, drops it, then exercises the read path through
//! `list_tool` — exactly matching the production flow.
//!
//! Included from `tools/list/mod.rs` via `#[cfg(test)] #[path = "../list_tests.rs"] mod tests;`.

use rusqlite::{params, Connection};
use tempfile::TempDir;

use crate::server_types::SessionState;
use crate::sqlite::{SharedStateManager, StateManager};
// DEFAULT_DEPTH / MAX_DEPTH / DEFAULT_LIMIT / MAX_LIMIT are used in part3 (§12 constants tests).
use crate::tools::list::{list_tool, ListInput};

// ---------------------------------------------------------------------------
// DB helpers — minimal schema matching production state.db
// ---------------------------------------------------------------------------

/// Schema shared by all tests.
const SCHEMA: &str = "
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;

    CREATE TABLE watch_folders (
        watch_id            TEXT PRIMARY KEY,
        tenant_id           TEXT NOT NULL,
        path                TEXT NOT NULL,
        collection          TEXT NOT NULL,
        git_remote_url      TEXT,
        remote_hash         TEXT,
        disambiguation_path TEXT,
        is_active           INTEGER NOT NULL DEFAULT 1,
        parent_watch_id     TEXT,
        submodule_path      TEXT,
        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at          TEXT,
        last_activity_at    TEXT
    );

    CREATE TABLE tracked_files (
        file_id         TEXT PRIMARY KEY,
        watch_folder_id TEXT NOT NULL,
        base_point      TEXT,
        relative_path   TEXT NOT NULL,
        file_type       TEXT,
        language        TEXT,
        extension       TEXT,
        is_test         INTEGER NOT NULL DEFAULT 0,
        branches        TEXT NOT NULL DEFAULT '[]',
        component       TEXT
    );

    CREATE TABLE project_components (
        component_id    TEXT PRIMARY KEY,
        watch_folder_id TEXT NOT NULL,
        component_name  TEXT NOT NULL,
        base_path       TEXT NOT NULL,
        source          TEXT NOT NULL DEFAULT 'cargo',
        patterns        TEXT
    );
";

struct TestDb {
    _dir: TempDir,
    path: std::path::PathBuf,
}

impl TestDb {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.db");
        let setup = Connection::open(&path).unwrap();
        setup.execute_batch(SCHEMA).unwrap();
        drop(setup);
        TestDb { _dir: dir, path }
    }

    fn write_conn(&self) -> Connection {
        Connection::open(&self.path).unwrap()
    }

    fn state_manager(&self) -> SharedStateManager {
        SharedStateManager::new(StateManager::open_at(&self.path))
    }

    fn insert_project(&self, watch_id: &str, tenant_id: &str, proj_path: &str) {
        let c = self.write_conn();
        c.execute(
            "INSERT INTO watch_folders
             (watch_id, tenant_id, path, collection, is_active)
             VALUES (?1, ?2, ?3, 'projects', 1)",
            params![watch_id, tenant_id, proj_path],
        )
        .unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_file(
        &self,
        id: &str,
        wfid: &str,
        rel_path: &str,
        lang: Option<&str>,
        ext: Option<&str>,
        is_test: i64,
        branches: &str,
        file_type: Option<&str>,
    ) {
        let c = self.write_conn();
        c.execute(
            "INSERT INTO tracked_files
             (file_id, watch_folder_id, relative_path, language, extension,
              is_test, branches, file_type)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![id, wfid, rel_path, lang, ext, is_test, branches, file_type],
        )
        .unwrap();
    }

    fn insert_component(&self, id: &str, wfid: &str, name: &str, base_path: &str, source: &str) {
        let c = self.write_conn();
        c.execute(
            "INSERT INTO project_components
             (component_id, watch_folder_id, component_name, base_path, source)
             VALUES (?1,?2,?3,?4,?5)",
            params![id, wfid, name, base_path, source],
        )
        .unwrap();
    }

    fn insert_submodule(
        &self,
        watch_id: &str,
        tenant_id: &str,
        parent_id: &str,
        sub_path: &str,
        url: Option<&str>,
    ) {
        let c = self.write_conn();
        c.execute(
            "INSERT INTO watch_folders
             (watch_id, tenant_id, path, collection, parent_watch_id, submodule_path, git_remote_url)
             VALUES (?1,?2,?3,'projects',?4,?5,?6)",
            params![watch_id, tenant_id, format!("/proj/{sub_path}"), parent_id, sub_path, url],
        )
        .unwrap();
    }
}

// ---------------------------------------------------------------------------
// Session state helpers
// ---------------------------------------------------------------------------

fn session_with_project(project_id: &str) -> SessionState {
    let mut s = SessionState::new();
    s.project_id = Some(project_id.to_string());
    s
}

fn session_with_branch(project_id: &str, branch: &str) -> SessionState {
    let mut s = session_with_project(project_id);
    s.current_branch = Some(branch.to_string());
    s
}

// ---------------------------------------------------------------------------
// Helper: call list_tool and decode the JSON payload from the text content
// ---------------------------------------------------------------------------

fn call_list(db: &TestDb, input: ListInput, session: &SessionState) -> serde_json::Value {
    let mgr = db.state_manager(); // SharedStateManager
    let result = list_tool(input, &mgr, session);
    // Content = Annotated<RawContent>; .raw is the RawContent enum; .as_text() gives &RawTextContent.
    let text = result
        .content
        .first()
        .expect("content must not be empty")
        .raw
        .as_text()
        .expect("first content item must be text")
        .text
        .clone();
    serde_json::from_str(&text).expect("response must be valid JSON")
}

// ---------------------------------------------------------------------------
// § 1  No project → error response
// ---------------------------------------------------------------------------

#[test]
fn no_project_id_returns_error() {
    let db = TestDb::new();
    let session = SessionState::new(); // no project_id
    let v = call_list(&db, ListInput::default(), &session);
    assert_eq!(v["success"], false);
    assert!(v["message"]
        .as_str()
        .unwrap()
        .contains("Could not detect project"));
}

#[test]
fn project_not_in_db_returns_error() {
    let db = TestDb::new();
    let session = session_with_project("missing-tenant");
    let v = call_list(&db, ListInput::default(), &session);
    assert_eq!(v["success"], false);
    assert!(v["message"]
        .as_str()
        .unwrap()
        .contains("not found in database"));
}

// ---------------------------------------------------------------------------
// § 2  Empty result
// ---------------------------------------------------------------------------

#[test]
fn empty_project_returns_success_with_zero_stats() {
    let db = TestDb::new();
    db.insert_project("wid-empty", "tenant-empty", "/proj/empty");
    let session = session_with_project("tenant-empty");
    let v = call_list(&db, ListInput::default(), &session);

    assert_eq!(v["success"], true);
    assert_eq!(v["stats"]["files"], 0);
    assert_eq!(v["stats"]["folders"], 0);
    assert_eq!(v["stats"]["truncated"], false);
    assert_eq!(v["stats"]["totalMatching"], 0);
    assert!(v["listing"].as_str().unwrap().is_empty());
    // next_token absent
    assert!(v.get("next_token").is_none() || v["next_token"].is_null());
}

// ---------------------------------------------------------------------------
// § 3  Tree format — byte-for-byte rendering
// ---------------------------------------------------------------------------

#[test]
fn tree_format_basic_rendering() {
    let db = TestDb::new();
    db.insert_project("wid1", "tenant1", "/proj");
    db.insert_file(
        "f1",
        "wid1",
        "src/main.rs",
        Some("rust"),
        Some("rs"),
        0,
        "[]",
        Some("code"),
    );
    db.insert_file(
        "f2",
        "wid1",
        "src/lib.rs",
        Some("rust"),
        Some("rs"),
        0,
        "[]",
        Some("code"),
    );
    db.insert_file("f3", "wid1", "README.md", None, Some("md"), 0, "[]", None);

    let session = session_with_project("tenant1");
    let v = call_list(
        &db,
        ListInput {
            format: Some("tree".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["success"], true);
    assert_eq!(v["format"], "tree");
    let listing = v["listing"].as_str().unwrap();
    // src/ folder should appear before README.md at root level (BTreeMap sorted)
    assert!(
        listing.contains("src/"),
        "expected 'src/' in listing: {listing:?}"
    );
    // Files inside src/ are indented
    assert!(
        listing.contains("  main.rs [rs]"),
        "expected '  main.rs [rs]': {listing:?}"
    );
    assert!(
        listing.contains("  lib.rs [rs]"),
        "expected '  lib.rs [rs]': {listing:?}"
    );
    // README at root level (no indent)
    assert!(
        listing.contains("README.md [md]"),
        "expected 'README.md [md]': {listing:?}"
    );
    // lib.rs sorts before main.rs
    let lib_pos = listing.find("lib.rs").unwrap();
    let main_pos = listing.find("main.rs").unwrap();
    assert!(lib_pos < main_pos, "lib.rs should appear before main.rs");
}

#[test]
fn tree_format_sample_listing_exact() {
    // Exact byte-for-byte assertion for a simple 2-file tree.
    let db = TestDb::new();
    db.insert_project("wid-exact", "t-exact", "/p");
    db.insert_file(
        "f1",
        "wid-exact",
        "src/main.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-exact");
    let v = call_list(
        &db,
        ListInput {
            format: Some("tree".to_string()),
            ..Default::default()
        },
        &session,
    );

    let listing = v["listing"].as_str().unwrap();
    // Expected: "src/\n  main.rs [rs]"
    assert_eq!(
        listing, "src/\n  main.rs [rs]",
        "exact tree listing mismatch: {listing:?}"
    );
}

// ---------------------------------------------------------------------------
// § 4  Summary format
// ---------------------------------------------------------------------------

#[test]
fn summary_format_shows_extension_counts() {
    let db = TestDb::new();
    db.insert_project("wid-sum", "t-sum", "/p");
    db.insert_file("f1", "wid-sum", "src/a.rs", None, Some("rs"), 0, "[]", None);
    db.insert_file("f2", "wid-sum", "src/b.rs", None, Some("rs"), 0, "[]", None);
    db.insert_file("f3", "wid-sum", "src/c.ts", None, Some("ts"), 0, "[]", None);

    let session = session_with_project("t-sum");
    let v = call_list(
        &db,
        ListInput {
            format: Some("summary".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["success"], true);
    assert_eq!(v["format"], "summary");
    let listing = v["listing"].as_str().unwrap();
    // Summary shows folder with counts
    assert!(
        listing.contains("src/"),
        "expected 'src/' in summary: {listing:?}"
    );
    assert!(
        listing.contains("3 files"),
        "expected '3 files': {listing:?}"
    );
    assert!(
        listing.contains("2 rs"),
        "expected '2 rs' in summary: {listing:?}"
    );
    assert!(
        listing.contains("1 ts"),
        "expected '1 ts' in summary: {listing:?}"
    );
}

// ---------------------------------------------------------------------------
// § 5  Flat format
// ---------------------------------------------------------------------------

#[test]
fn flat_format_one_path_per_line() {
    let db = TestDb::new();
    db.insert_project("wid-flat", "t-flat", "/p");
    db.insert_file("f1", "wid-flat", "a.rs", None, Some("rs"), 0, "[]", None);
    db.insert_file("f2", "wid-flat", "b.rs", None, Some("rs"), 0, "[]", None);
    db.insert_file("f3", "wid-flat", "c.rs", None, Some("rs"), 0, "[]", None);

    let session = session_with_project("t-flat");
    let v = call_list(
        &db,
        ListInput {
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["success"], true);
    assert_eq!(v["format"], "flat");
    let listing = v["listing"].as_str().unwrap();
    let lines: Vec<&str> = listing.lines().collect();
    assert_eq!(lines.len(), 3);
    // ORDER BY relative_path ASC
    assert_eq!(lines, vec!["a.rs", "b.rs", "c.rs"]);
}

// ---------------------------------------------------------------------------
// § 6  Filter: language
// ---------------------------------------------------------------------------

#[test]
fn filter_language() {
    let db = TestDb::new();
    db.insert_project("wid-lang", "t-lang", "/p");
    db.insert_file(
        "f1",
        "wid-lang",
        "a.rs",
        Some("rust"),
        Some("rs"),
        0,
        "[]",
        None,
    );
    db.insert_file(
        "f2",
        "wid-lang",
        "b.py",
        Some("python"),
        Some("py"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-lang");
    let v = call_list(
        &db,
        ListInput {
            language: Some("rust".to_string()),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    assert_eq!(v["stats"]["files"], 1);
    assert_eq!(v["listing"].as_str().unwrap(), "a.rs");
}

// ---------------------------------------------------------------------------
// § 7  Filter: extension — moved to list_tests_part5.rs
// § 8–§11 and § 12–§23: split into sibling files to keep each under 500 lines
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "list_tests_part2.rs"]
mod part2;

#[cfg(test)]
#[path = "list_tests_part3.rs"]
mod part3;

#[cfg(test)]
#[path = "list_tests_part4.rs"]
mod part4;

#[cfg(test)]
#[path = "list_tests_part5.rs"]
mod part5;
