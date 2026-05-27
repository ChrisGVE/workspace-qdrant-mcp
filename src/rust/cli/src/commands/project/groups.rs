//! `wqm project groups` -- show group memberships for a project.
//!
//! Queries `project_groups` in state.db to find all groups the current
//! project belongs to, and lists their members. No gRPC required.

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

/// A group the current project belongs to, with all its member tenant IDs.
struct GroupInfo {
    group_id: String,
    group_type: String,
    confidence: f64,
    members: Vec<String>,
}

/// JSON-serializable row for `--json` output.
#[derive(serde::Serialize)]
struct GroupJsonRow {
    group_id: String,
    group_type: String,
    confidence: f64,
    members: Vec<String>,
}

/// List groups the current project belongs to.
pub(super) async fn project_groups(
    project: Option<&str>,
    strategy: Option<&str>,
    json: bool,
    script: bool,
    no_headers: bool,
) -> Result<()> {
    let project_id = resolve_project_id_or_cwd(project)?;

    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(e) => {
            crate::output::error(format!("Database not available: {}", e));
            return Ok(());
        }
    };

    if !table_exists(&conn, "project_groups") {
        crate::output::info(
            "No project groups table found. \
             Ensure daemon schema v24+ is applied.",
        );
        return Ok(());
    }

    let groups = query_project_groups(&conn, &project_id, strategy)?;

    if groups.is_empty() {
        let qualifier = strategy
            .map(|s| format!(" with strategy '{s}'"))
            .unwrap_or_default();
        crate::output::info(format!(
            "No group memberships found for project {}{qualifier}.\n\
             Groups are computed automatically by the daemon scheduler.",
            &project_id,
        ));
        return Ok(());
    }

    if json {
        render_json(&groups);
    } else if script {
        render_script(&groups, !no_headers);
    } else {
        canvas::print_title("Project Groups");
        canvas::print_blank();
        render_table(&groups);
    }

    Ok(())
}

/// Query all groups a project belongs to, including each group's member list.
fn query_project_groups(
    conn: &rusqlite::Connection,
    tenant_id: &str,
    strategy: Option<&str>,
) -> Result<Vec<GroupInfo>> {
    // Step 1: find groups this tenant belongs to
    let mut sql = String::from(
        "SELECT group_id, group_type, confidence \
         FROM project_groups WHERE tenant_id = ?1",
    );
    if strategy.is_some() {
        sql.push_str(" AND group_type = ?2");
    }
    sql.push_str(" ORDER BY group_type, group_id");

    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<(String, String, f64)> = if let Some(gt) = strategy {
        stmt.query_map(rusqlite::params![tenant_id, gt], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect()
    } else {
        stmt.query_map(rusqlite::params![tenant_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect()
    };

    // Step 2: for each group, fetch all members
    let mut member_stmt = conn.prepare(
        "SELECT tenant_id FROM project_groups \
         WHERE group_id = ?1 ORDER BY tenant_id",
    )?;

    let mut groups = Vec::with_capacity(rows.len());
    for (group_id, group_type, confidence) in rows {
        let members: Vec<String> = member_stmt
            .query_map(rusqlite::params![&group_id], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        groups.push(GroupInfo {
            group_id,
            group_type,
            confidence,
            members,
        });
    }

    Ok(groups)
}

/// Render groups as JSON array.
fn render_json(groups: &[GroupInfo]) {
    let rows: Vec<GroupJsonRow> = groups
        .iter()
        .map(|g| GroupJsonRow {
            group_id: g.group_id.clone(),
            group_type: g.group_type.clone(),
            confidence: g.confidence,
            members: g.members.clone(),
        })
        .collect();

    match serde_json::to_string_pretty(&rows) {
        Ok(json) => println!("{json}"),
        Err(e) => crate::output::error(format!("JSON serialization failed: {e}")),
    }
}

/// Render groups as tab-separated flat rows (one line per member).
fn render_script(groups: &[GroupInfo], headers: bool) {
    if headers {
        println!("group_id\tgroup_type\tconfidence\tmember");
    }
    for g in groups {
        let conf = format!("{:.2}", g.confidence);
        for member in &g.members {
            println!("{}\t{}\t{conf}\t{member}", g.group_id, g.group_type);
        }
    }
}

/// Render groups as a human-readable table.
fn render_table(groups: &[GroupInfo]) {
    let width = terminal_width();

    let mut builder = Builder::default();
    builder.push_record(["Group ID", "Type", "Members", "Confidence"]);

    for g in groups {
        let members_display = g.members.join(", ");
        builder.push_record([
            g.group_id.clone(),
            g.group_type.clone(),
            members_display,
            format!("{:.2}", g.confidence),
        ]);
    }

    let mut table = builder.build();

    let style = Style::blank().horizontals([(1, HorizontalLine::new('─').intersection('─'))]);
    table
        .with(style)
        .with(Modify::new(Rows::first()).with(Color::BOLD));

    // Right-align confidence column (index 3)
    table.with(Modify::new(Columns::single(3)).with(Alignment::right()));

    table.with(Width::wrap(width).keep_words());
    table.with(Width::increase(width));

    println!("{table}");
    println!("{}", "─".repeat(width));

    let count = groups.len();
    let summary = format!(
        "{count} group{} ({} total members)",
        if count == 1 { "" } else { "s" },
        groups.iter().map(|g| g.members.len()).sum::<usize>()
    );
    print_table_summary(&summary);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE project_groups (
                group_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                group_type TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (group_id, tenant_id)
            );",
        )
        .unwrap();
        conn
    }

    fn insert_membership(
        conn: &Connection,
        group_id: &str,
        tenant_id: &str,
        group_type: &str,
        confidence: f64,
    ) {
        conn.execute(
            "INSERT INTO project_groups (group_id, tenant_id, group_type, confidence, created_at)
             VALUES (?1, ?2, ?3, ?4, '2024-01-01T00:00:00Z')",
            rusqlite::params![group_id, tenant_id, group_type, confidence],
        )
        .unwrap();
    }

    #[test]
    fn test_query_empty_table() {
        let conn = setup_test_db();
        let groups = query_project_groups(&conn, "proj-a", None).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn test_query_single_group() {
        let conn = setup_test_db();
        insert_membership(&conn, "grp-1", "proj-a", "workspace", 1.0);
        insert_membership(&conn, "grp-1", "proj-b", "workspace", 1.0);
        insert_membership(&conn, "grp-1", "proj-c", "workspace", 0.8);

        let groups = query_project_groups(&conn, "proj-a", None).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group_id, "grp-1");
        assert_eq!(groups[0].group_type, "workspace");
        assert_eq!(groups[0].members.len(), 3);
        assert!(groups[0].members.contains(&"proj-a".to_string()));
        assert!(groups[0].members.contains(&"proj-b".to_string()));
        assert!(groups[0].members.contains(&"proj-c".to_string()));
    }

    #[test]
    fn test_query_multiple_groups() {
        let conn = setup_test_db();
        insert_membership(&conn, "grp-1", "proj-a", "workspace", 1.0);
        insert_membership(&conn, "grp-1", "proj-b", "workspace", 1.0);
        insert_membership(&conn, "grp-2", "proj-a", "git_org", 0.9);
        insert_membership(&conn, "grp-2", "proj-c", "git_org", 0.9);

        let groups = query_project_groups(&conn, "proj-a", None).unwrap();
        assert_eq!(groups.len(), 2);

        let git_org = groups.iter().find(|g| g.group_type == "git_org").unwrap();
        assert_eq!(git_org.members.len(), 2);
        assert!(git_org.members.contains(&"proj-a".to_string()));
        assert!(git_org.members.contains(&"proj-c".to_string()));
    }

    #[test]
    fn test_query_strategy_filter() {
        let conn = setup_test_db();
        insert_membership(&conn, "grp-1", "proj-a", "workspace", 1.0);
        insert_membership(&conn, "grp-2", "proj-a", "git_org", 0.9);

        let groups = query_project_groups(&conn, "proj-a", Some("workspace")).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group_type, "workspace");
    }

    #[test]
    fn test_query_strategy_filter_no_match() {
        let conn = setup_test_db();
        insert_membership(&conn, "grp-1", "proj-a", "workspace", 1.0);

        let groups = query_project_groups(&conn, "proj-a", Some("affinity")).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn test_query_tenant_isolation() {
        let conn = setup_test_db();
        insert_membership(&conn, "grp-1", "proj-a", "workspace", 1.0);
        insert_membership(&conn, "grp-1", "proj-b", "workspace", 1.0);
        insert_membership(&conn, "grp-2", "proj-c", "dependency", 0.85);

        // proj-c should only see grp-2
        let groups = query_project_groups(&conn, "proj-c", None).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group_id, "grp-2");
    }

    #[test]
    fn test_query_members_sorted() {
        let conn = setup_test_db();
        insert_membership(&conn, "grp-1", "proj-c", "workspace", 1.0);
        insert_membership(&conn, "grp-1", "proj-a", "workspace", 1.0);
        insert_membership(&conn, "grp-1", "proj-b", "workspace", 1.0);

        let groups = query_project_groups(&conn, "proj-a", None).unwrap();
        assert_eq!(groups[0].members, vec!["proj-a", "proj-b", "proj-c"]);
    }
}
