//! List all known branches for a project
//!
//! Queries `tracked_files.branches` JSON arrays in state.db to enumerate
//! unique branch names, their file counts, and marks the currently checked-out
//! branch. No gRPC required — reads directly from the local SQLite state.db.

use anyhow::Result;
use tabled::builder::Builder;
use tabled::settings::object::{Columns, Rows};
use tabled::settings::style::{HorizontalLine, Style};
use tabled::settings::{Alignment, Color, Modify, Width};

use crate::data::db::connect_readonly;
use crate::output::canvas;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::render::print_table_summary;
use crate::output::table::terminal_width;

use super::resolver::resolve_project_id_or_cwd;

/// A single branch row for display.
struct BranchRow {
    name: String,
    file_count: usize,
    is_current: bool,
}

/// List branches for a project, queried from `tracked_files.branches`.
pub(super) async fn list_branches(project: Option<&str>) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    canvas::print_title("Project Branches");
    canvas::print_blank();

    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(e) => {
            crate::output::error(format!("Database not available: {}", e));
            return Ok(());
        }
    };

    let rows = query_branches(&conn, &project_id)?;

    if rows.is_empty() {
        crate::output::info(
            "No branches found for this project. \
             Ensure the project is registered and files have been indexed.",
        );
        return Ok(());
    }

    let locale = NumberLocale::default();
    let width = terminal_width();
    render_branches_table(&rows, width, &locale);

    let total = rows.len();
    let summary = format!("{} branch{}", total, if total == 1 { "" } else { "es" });
    print_table_summary(&summary);

    Ok(())
}

/// Query unique branches and per-branch file counts for a tenant.
///
/// Uses `json_each` to expand the JSON array stored in `tracked_files.branches`,
/// then counts distinct files per branch value. The `watch_folders` join scopes
/// the query to the given tenant_id.
fn query_branches(conn: &rusqlite::Connection, tenant_id: &str) -> Result<Vec<BranchRow>> {
    // Detect the current git branch from the project's watch path (best-effort).
    let current_branch = detect_current_branch(conn, tenant_id);

    let mut stmt = conn.prepare(
        "SELECT b.value AS branch_name, COUNT(DISTINCT tf.file_id) AS file_count
         FROM tracked_files tf
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id,
              json_each(tf.branches) b
         WHERE wf.tenant_id = ?1
           AND wf.collection = 'projects'
         GROUP BY b.value
         ORDER BY file_count DESC, b.value ASC",
    )?;

    let rows = stmt
        .query_map(rusqlite::params![tenant_id], |row| {
            let name: String = row.get(0)?;
            let file_count: i64 = row.get(1)?;
            Ok((name, file_count as usize))
        })?
        .filter_map(|r| r.ok())
        .map(|(name, file_count)| {
            let is_current = current_branch
                .as_deref()
                .map(|cb| cb == name)
                .unwrap_or(false);
            BranchRow {
                name,
                file_count,
                is_current,
            }
        })
        .collect();

    Ok(rows)
}

/// Detect the current git branch for the project's root path.
///
/// Reads the project's `path` from `watch_folders`, then asks git.
/// Returns `None` when the path cannot be determined or git is unavailable.
fn detect_current_branch(conn: &rusqlite::Connection, tenant_id: &str) -> Option<String> {
    let path: String = conn
        .query_row(
            "SELECT path FROM watch_folders \
             WHERE tenant_id = ?1 AND parent_watch_id IS NULL \
             AND collection = 'projects' LIMIT 1",
            rusqlite::params![tenant_id],
            |row| row.get(0),
        )
        .ok()?;

    git_current_branch(&path)
}

/// Ask git for the current branch at `path`. Returns `None` on any error
/// (git unavailable, detached HEAD, not a repo, etc.).
fn git_current_branch(path: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["-C", path, "rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if output.status.success() {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if branch.is_empty() || branch == "HEAD" {
            None
        } else {
            Some(branch)
        }
    } else {
        None
    }
}

/// Render the branches table with a header separator and closing line.
fn render_branches_table(rows: &[BranchRow], width: usize, locale: &NumberLocale) {
    if rows.is_empty() {
        return;
    }

    let table_width = width;

    let mut builder = Builder::default();
    builder.push_record(["Branch", "Files", "Current"]);

    for row in rows {
        let current_marker = if row.is_current { "yes" } else { "" };
        builder.push_record([
            row.name.clone(),
            format_usize(row.file_count, locale),
            current_marker.to_string(),
        ]);
    }

    let mut table = builder.build();

    let style = Style::blank().horizontals([(1, HorizontalLine::new('─').intersection('─'))]);
    table
        .with(style)
        .with(Modify::new(Rows::first()).with(Color::BOLD));

    // Right-align the Files column (index 1)
    table.with(Modify::new(Columns::single(1)).with(Alignment::right()));

    table.with(Width::wrap(table_width).keep_words());
    table.with(Width::increase(table_width));

    println!("{}", table);
    println!("{}", "─".repeat(width));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                tenant_id TEXT,
                path TEXT,
                collection TEXT,
                parent_watch_id TEXT,
                is_active INTEGER DEFAULT 1
            );
            CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY,
                watch_folder_id TEXT,
                relative_path TEXT,
                branches TEXT NOT NULL DEFAULT '[]',
                file_hash TEXT,
                collection TEXT,
                base_point TEXT,
                created_at TEXT,
                updated_at TEXT
            );",
        )
        .unwrap();
        conn
    }

    fn insert_watch(conn: &Connection, watch_id: &str, tenant_id: &str, path: &str) {
        conn.execute(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, parent_watch_id)
             VALUES (?1, ?2, ?3, 'projects', NULL)",
            rusqlite::params![watch_id, tenant_id, path],
        )
        .unwrap();
    }

    fn insert_file(conn: &Connection, file_id: i64, watch_id: &str, branches_json: &str) {
        conn.execute(
            "INSERT INTO tracked_files (file_id, watch_folder_id, relative_path, branches, \
             file_hash, collection, base_point, created_at, updated_at)
             VALUES (?1, ?2, 'path.rs', ?3, 'hash', 'projects', 'bp', \
             '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
            rusqlite::params![file_id, watch_id, branches_json],
        )
        .unwrap();
    }

    #[test]
    fn test_query_branches_empty_project() {
        let conn = setup_test_db();
        insert_watch(&conn, "w1", "tenant1", "/proj");

        let rows = query_branches(&conn, "tenant1").unwrap();
        assert!(rows.is_empty(), "No files → no branches");
    }

    #[test]
    fn test_query_branches_single_branch() {
        let conn = setup_test_db();
        insert_watch(&conn, "w1", "tenant1", "/proj");
        insert_file(&conn, 1, "w1", r#"["main"]"#);
        insert_file(&conn, 2, "w1", r#"["main"]"#);

        let rows = query_branches(&conn, "tenant1").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "main");
        assert_eq!(rows[0].file_count, 2);
    }

    #[test]
    fn test_query_branches_multiple_branches() {
        let conn = setup_test_db();
        insert_watch(&conn, "w1", "tenant1", "/proj");
        // File on both main and feature
        insert_file(&conn, 1, "w1", r#"["main","feature"]"#);
        // File on main only
        insert_file(&conn, 2, "w1", r#"["main"]"#);
        // File on feature only
        insert_file(&conn, 3, "w1", r#"["feature"]"#);

        let rows = query_branches(&conn, "tenant1").unwrap();
        assert_eq!(rows.len(), 2);

        // Sorted by file_count DESC: main=2, feature=2 (tie → alphabetical)
        let branch_names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(branch_names.contains(&"main"));
        assert!(branch_names.contains(&"feature"));

        // main has files 1 and 2 → 2 files
        let main = rows.iter().find(|r| r.name == "main").unwrap();
        assert_eq!(main.file_count, 2);

        // feature has files 1 and 3 → 2 files
        let feature = rows.iter().find(|r| r.name == "feature").unwrap();
        assert_eq!(feature.file_count, 2);
    }

    #[test]
    fn test_query_branches_tenant_isolation() {
        let conn = setup_test_db();
        insert_watch(&conn, "w1", "tenant1", "/proj1");
        insert_watch(&conn, "w2", "tenant2", "/proj2");
        insert_file(&conn, 1, "w1", r#"["main"]"#);
        insert_file(&conn, 2, "w2", r#"["dev"]"#);

        let rows_t1 = query_branches(&conn, "tenant1").unwrap();
        assert_eq!(rows_t1.len(), 1);
        assert_eq!(rows_t1[0].name, "main");

        let rows_t2 = query_branches(&conn, "tenant2").unwrap();
        assert_eq!(rows_t2.len(), 1);
        assert_eq!(rows_t2[0].name, "dev");
    }

    #[test]
    fn test_query_branches_empty_json_array() {
        let conn = setup_test_db();
        insert_watch(&conn, "w1", "tenant1", "/proj");
        // File with empty branches array produces no rows from json_each
        insert_file(&conn, 1, "w1", r#"[]"#);

        let rows = query_branches(&conn, "tenant1").unwrap();
        assert!(rows.is_empty(), "Empty branches array → no rows");
    }
}
