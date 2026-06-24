//! `wqm project branches` -- list branches known to the branch-storage index.
//!
//! File: `cli/src/commands/project/branches.rs`
//! Context: workspace-qdrant-mcp; AC-F10.8, AC-F10.9.
//!
//!   Source (AC-F10.8): `state.db.project_locations JOIN projects`. Each row
//!   represents one (project, branch, checkout) combination. Replaces the
//!   old `tracked_files.branches` json_each source, which used the pre-F0
//!   single-table schema. No daemon connection required.
//!
//!   Output modes: human-readable table (default), `--json`, `--script`
//!   (tab-separated), `--no-headers` (suppresses header row in script mode).
//!
//! Neighbors: `super::resolver::resolve_project_id_or_cwd` (tenant
//!   resolution), `crate::data::db::connect_readonly` (state.db connection).

use anyhow::Result;
use tabled::builder::Builder;
use tabled::settings::object::{Columns, Rows};
use tabled::settings::style::{HorizontalLine, Style};
use tabled::settings::{Alignment, Color, Modify, Width};

use crate::data::db::{connect_readonly, table_exists};
use crate::output::canvas;
use crate::output::render::print_table_summary;
use crate::output::table::terminal_width;

use super::resolver::resolve_project_id_or_cwd;

// ---------------------------------------------------------------------------
// Domain type
// ---------------------------------------------------------------------------

/// One branch row sourced from `project_locations JOIN projects`.
#[derive(Debug, Clone)]
struct BranchRow {
    branch_name: String,
    branch_id: String,
    location: String,
    sync_state: String,
    active: bool,
}

/// JSON-serializable form for `--json` output.
#[derive(serde::Serialize)]
struct BranchJsonRow {
    branch_name: String,
    branch_id: String,
    location: String,
    sync_state: String,
    active: bool,
}

impl From<&BranchRow> for BranchJsonRow {
    fn from(r: &BranchRow) -> Self {
        BranchJsonRow {
            branch_name: r.branch_name.clone(),
            branch_id: r.branch_id.clone(),
            location: r.location.clone(),
            sync_state: r.sync_state.clone(),
            active: r.active,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// List branches for a project (AC-F10.8).
///
/// Accepts optional format flags `json`, `script`, `no_headers` following
/// the same convention as `wqm project groups` (arch §6.2, AC-F10.8).
pub(super) async fn list_branches(
    project: Option<&str>,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
    let tenant_id = resolve_project_id_or_cwd(project)?;

    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(e) => {
            crate::output::error(format!("Database not available: {}", e));
            return Ok(());
        }
    };

    // Guard: new schema (project_locations) may not exist on older daemons.
    if !table_exists(&conn, "project_locations") {
        crate::output::info(
            "project_locations table not found. \
             Daemon schema F10+ required. Update the daemon and re-register the project.",
        );
        return Ok(());
    }

    let rows = query_branches(&conn, &tenant_id)?;

    if rows.is_empty() {
        crate::output::info(
            "No branches found for this project. \
             Ensure the project is registered and at least one branch has been indexed.",
        );
        return Ok(());
    }

    if json {
        render_json(&rows);
    } else if script {
        render_script(&rows, !no_headers);
    } else {
        canvas::print_title("Project Branches");
        canvas::print_blank();
        render_table(&rows);
        let total = rows.len();
        let summary = format!("{} branch{}", total, if total == 1 { "" } else { "es" });
        print_table_summary(&summary);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// Query branches from `project_locations JOIN projects` for `tenant_id`.
///
/// Returns one row per (branch, checkout location) combination, ordered by
/// branch name then location. All SQL values are bound parameters (arch §6.5
/// GP-8).
fn query_branches(conn: &rusqlite::Connection, tenant_id: &str) -> Result<Vec<BranchRow>> {
    let mut stmt = conn.prepare(
        "SELECT
             pl.branch_name,
             pl.branch_id,
             pl.location,
             pl.sync_state,
             pl.active
         FROM project_locations pl
         JOIN projects p ON p.project_id = pl.project_id
         WHERE p.tenant_id = ?1
         ORDER BY pl.branch_name ASC, pl.location ASC",
    )?;

    let rows = stmt
        .query_map(rusqlite::params![tenant_id], |row| {
            Ok(BranchRow {
                branch_name: row.get(0)?,
                branch_id: row.get(1)?,
                location: row.get(2)?,
                sync_state: row.get(3)?,
                active: row.get::<_, i64>(4)? != 0,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

fn render_json(rows: &[BranchRow]) {
    let json_rows: Vec<BranchJsonRow> = rows.iter().map(BranchJsonRow::from).collect();
    match serde_json::to_string_pretty(&json_rows) {
        Ok(s) => println!("{s}"),
        Err(e) => crate::output::error(format!("JSON serialization failed: {e}")),
    }
}

fn render_script(rows: &[BranchRow], with_headers: bool) {
    if with_headers {
        println!("branch_name\tbranch_id\tlocation\tsync_state\tactive");
    }
    for r in rows {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            r.branch_name, r.branch_id, r.location, r.sync_state, r.active
        );
    }
}

fn render_table(rows: &[BranchRow]) {
    let width = terminal_width();

    let mut builder = Builder::default();
    builder.push_record(["Branch", "Sync State", "Location", "Active"]);

    for r in rows {
        let active_marker = if r.active { "yes" } else { "" };
        builder.push_record([
            r.branch_name.clone(),
            r.sync_state.clone(),
            r.location.clone(),
            active_marker.to_string(),
        ]);
    }

    let mut table = builder.build();
    let style = Style::blank().horizontals([(1, HorizontalLine::new('-').intersection('-'))]);
    table
        .with(style)
        .with(Modify::new(Rows::first()).with(Color::BOLD))
        .with(Modify::new(Columns::single(3)).with(Alignment::right()))
        .with(Width::wrap(width).keep_words())
        .with(Width::increase(width));

    println!("{table}");
    println!("{}", "-".repeat(width));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE projects (
                project_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL,
                tenant_id    TEXT NOT NULL UNIQUE,
                db_path      TEXT NOT NULL,
                content_key_version INTEGER NOT NULL DEFAULT 3,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE TABLE project_locations (
                location_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   INTEGER NOT NULL REFERENCES projects(project_id),
                location     TEXT NOT NULL,
                branch_name  TEXT NOT NULL,
                branch_id    TEXT NOT NULL UNIQUE,
                active       INTEGER NOT NULL DEFAULT 1,
                sync_state   TEXT NOT NULL DEFAULT 'current'
                                 CHECK (sync_state IN ('pending','indexing','current','error')),
                last_synced  TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );",
        )
        .unwrap();
        conn
    }

    fn insert_project(conn: &Connection, tenant_id: &str) -> i64 {
        conn.execute(
            "INSERT INTO projects (name, tenant_id, db_path, created_at, updated_at)
             VALUES (?1, ?1, '/data/store.db', '2026-01-01', '2026-01-01')",
            rusqlite::params![tenant_id],
        )
        .unwrap();
        conn.last_insert_rowid()
    }

    fn insert_location(conn: &Connection, project_id: i64, branch_name: &str, sync_state: &str) {
        let branch_id = format!("bid-{project_id}-{branch_name}");
        conn.execute(
            "INSERT INTO project_locations
             (project_id, location, branch_name, branch_id, active, sync_state,
              created_at, updated_at)
             VALUES (?1, '/repo', ?2, ?3, 1, ?4, '2026-01-01', '2026-01-01')",
            rusqlite::params![project_id, branch_name, branch_id, sync_state],
        )
        .unwrap();
    }

    // AC-F10.8: empty project has no branches.
    #[test]
    fn t_f10_8_empty_project_returns_no_branches() {
        let conn = setup_db();
        insert_project(&conn, "tenant-1");
        let rows = query_branches(&conn, "tenant-1").unwrap();
        assert!(rows.is_empty());
    }

    // AC-F10.8: rows carry branch_name, sync_state, location, active.
    #[test]
    fn t_f10_8_rows_carry_expected_fields() {
        let conn = setup_db();
        let pid = insert_project(&conn, "tenant-2");
        insert_location(&conn, pid, "main", "current");
        insert_location(&conn, pid, "feat/x", "indexing");

        let rows = query_branches(&conn, "tenant-2").unwrap();
        assert_eq!(rows.len(), 2);

        // Ordered by branch_name ASC.
        assert_eq!(rows[0].branch_name, "feat/x");
        assert_eq!(rows[0].sync_state, "indexing");
        assert!(rows[0].active);

        assert_eq!(rows[1].branch_name, "main");
        assert_eq!(rows[1].sync_state, "current");
    }

    // AC-F10.8: tenant isolation.
    #[test]
    fn t_f10_8_tenant_isolation() {
        let conn = setup_db();
        let p1 = insert_project(&conn, "tenant-a");
        let p2 = insert_project(&conn, "tenant-b");
        insert_location(&conn, p1, "main", "current");
        insert_location(&conn, p2, "dev", "pending");

        let rows_a = query_branches(&conn, "tenant-a").unwrap();
        assert_eq!(rows_a.len(), 1);
        assert_eq!(rows_a[0].branch_name, "main");

        let rows_b = query_branches(&conn, "tenant-b").unwrap();
        assert_eq!(rows_b.len(), 1);
        assert_eq!(rows_b[0].branch_name, "dev");
    }
}
