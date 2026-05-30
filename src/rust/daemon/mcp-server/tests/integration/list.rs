// Integration tests: list tool → real SQLite tracked_files queries.
//
// it_list_tracked_files_nonexistent_wfid — empty result for unknown watch_folder_id
// it_count_tracked_files_nonexistent_wfid — zero count for unknown watch_folder_id
// it_list_tracked_files_first_real        — structural validation against first wfid
// it_list_project_components_real_db      — components for first real watch_folder_id

use super::helpers;
use mcp_server::sqlite::tracked_files::{
    count_tracked_files, list_project_components, list_tracked_files, ListTrackedFilesOptions,
};

// ---------------------------------------------------------------------------
// Live state.db: non-existent watch_folder_id returns empty (not a panic)
// ---------------------------------------------------------------------------

#[test]
fn it_list_tracked_files_nonexistent_wfid() {
    let mgr = match helpers::open_state_manager() {
        Some(m) => m,
        None => return,
    };

    let opts = ListTrackedFilesOptions {
        watch_folder_id: "__integration_test_nonexistent__".to_string(),
        limit: Some(10),
        ..Default::default()
    };

    let rows = list_tracked_files(mgr.connection(), &opts);
    assert!(
        rows.is_empty(),
        "non-existent watch_folder_id must yield zero rows; got: {}",
        rows.len()
    );
}

// ---------------------------------------------------------------------------
// Live state.db: count returns 0 for non-existent watch_folder_id
// ---------------------------------------------------------------------------

#[test]
fn it_count_tracked_files_nonexistent_wfid() {
    let mgr = match helpers::open_state_manager() {
        Some(m) => m,
        None => return,
    };

    let opts = ListTrackedFilesOptions {
        watch_folder_id: "__integration_test_nonexistent__".to_string(),
        ..Default::default()
    };

    let count = count_tracked_files(mgr.connection(), &opts);
    assert_eq!(
        count, 0,
        "non-existent watch_folder_id must count zero; got: {count}"
    );
}

// ---------------------------------------------------------------------------
// Live state.db: structural validation against the first real watch_folder_id
// ---------------------------------------------------------------------------

#[test]
fn it_list_tracked_files_first_real() {
    let mgr = match helpers::open_state_manager() {
        Some(m) => m,
        None => return,
    };

    let id = match first_watch_folder_id(&mgr) {
        Some(id) => id,
        None => return,
    };

    let opts = ListTrackedFilesOptions {
        watch_folder_id: id.clone(),
        limit: Some(5),
        ..Default::default()
    };

    let rows = list_tracked_files(mgr.connection(), &opts);

    for r in &rows {
        assert!(
            !r.relative_path.is_empty(),
            "relative_path must not be empty for watch_folder_id={id}"
        );
    }
    assert!(
        rows.len() <= 5,
        "list_tracked_files exceeded limit=5; got: {}",
        rows.len()
    );
}

// ---------------------------------------------------------------------------
// Live state.db: list_project_components for first watch_folder_id
// ---------------------------------------------------------------------------

#[test]
fn it_list_project_components_real_db() {
    let mgr = match helpers::open_state_manager() {
        Some(m) => m,
        None => return,
    };

    let id = match first_watch_folder_id(&mgr) {
        Some(id) => id,
        None => return,
    };

    let components = list_project_components(mgr.connection(), &id);

    for c in &components {
        assert!(
            !c.component_name.is_empty(),
            "component_name must not be empty for watch_folder_id={id}"
        );
    }
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

fn first_watch_folder_id(mgr: &mcp_server::sqlite::StateManager) -> Option<String> {
    let conn = mgr.connection()?;
    let id: Option<String> = conn
        .query_row(
            "SELECT watch_id FROM watch_folders ORDER BY watch_id ASC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .ok();
    if id.is_none() {
        eprintln!("SKIP: no watch_folders rows in state.db");
    }
    id
}
