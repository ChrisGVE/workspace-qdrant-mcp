//! Bulk-cancel pending queue items for a project.

use anyhow::Result;

use crate::commands::project::resolver::resolve_tenant_by_hint;
use crate::output;

use super::db::connect_readwrite;

/// Execute `wqm queue cancel`.
pub async fn execute(
    project: &str,
    statuses: &[&str],
    dry_run: bool,
    yes: bool,
) -> Result<()> {
    let conn = connect_readwrite()?;

    let (tenant_id, project_path) = resolve_tenant_by_hint(&conn, project)?;

    // Build IN-clause; in_progress is always excluded for safety
    let safe_statuses: Vec<&str> = statuses
        .iter()
        .copied()
        .filter(|&s| s != "in_progress")
        .collect();

    if safe_statuses.is_empty() {
        anyhow::bail!("No cancellable statuses specified (in_progress cannot be cancelled)");
    }

    let placeholders = safe_statuses
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 2))
        .collect::<Vec<_>>()
        .join(", ");

    let count_sql = format!(
        "SELECT COUNT(*) FROM unified_queue \
         WHERE tenant_id = ?1 AND status IN ({placeholders})"
    );

    // Build params for rusqlite (tenant_id first, then statuses)
    let count: i64 = {
        let mut stmt = conn.prepare(&count_sql)?;
        let all_params: Vec<Box<dyn rusqlite::ToSql>> = std::iter::once(
            Box::new(tenant_id.clone()) as Box<dyn rusqlite::ToSql>,
        )
        .chain(
            safe_statuses
                .iter()
                .map(|s| Box::new(s.to_string()) as Box<dyn rusqlite::ToSql>),
        )
        .collect();
        let refs: Vec<&dyn rusqlite::ToSql> = all_params.iter().map(|b| b.as_ref()).collect();
        stmt.query_row(refs.as_slice(), |row| row.get(0))?
    };

    if count == 0 {
        output::success(format!(
            "No {} items to cancel for project {} ({})",
            safe_statuses.join("/"),
            project_path,
            &tenant_id[..tenant_id.len().min(12)],
        ));
        return Ok(());
    }

    let status_label = safe_statuses.join("/");

    if dry_run {
        output::info(format!(
            "[dry-run] Would cancel {} {} items for project {} ({})",
            count,
            status_label,
            project_path,
            &tenant_id[..tenant_id.len().min(12)],
        ));
        return Ok(());
    }

    if !yes {
        output::warning(format!(
            "Will cancel {} {} items for project {} ({}).\n\
             In-progress items are never touched. Use -y to confirm or --dry-run to preview.",
            count,
            status_label,
            project_path,
            &tenant_id[..tenant_id.len().min(12)],
        ));
        return Ok(());
    }

    let delete_sql = format!(
        "DELETE FROM unified_queue \
         WHERE tenant_id = ?1 AND status IN ({placeholders})"
    );

    let deleted = {
        let all_params: Vec<Box<dyn rusqlite::ToSql>> = std::iter::once(
            Box::new(tenant_id.clone()) as Box<dyn rusqlite::ToSql>,
        )
        .chain(
            safe_statuses
                .iter()
                .map(|s| Box::new(s.to_string()) as Box<dyn rusqlite::ToSql>),
        )
        .collect();
        let refs: Vec<&dyn rusqlite::ToSql> = all_params.iter().map(|b| b.as_ref()).collect();
        conn.execute(&delete_sql, refs.as_slice())?
    };

    output::success(format!(
        "Cancelled {} {} items for project {} ({})",
        deleted,
        status_label,
        project_path,
        &tenant_id[..tenant_id.len().min(12)],
    ));

    Ok(())
}

#[cfg(test)]
mod tests {
    use rusqlite::Connection;

    /// Set up an in-memory DB with the minimal schema for cancel tests.
    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                path TEXT NOT NULL,
                collection TEXT NOT NULL,
                parent_watch_id TEXT
            );
            CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                status TEXT NOT NULL,
                item_type TEXT NOT NULL DEFAULT 'file',
                op TEXT NOT NULL DEFAULT 'add',
                collection TEXT NOT NULL DEFAULT 'projects',
                idempotency_key TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO watch_folders VALUES
                ('wf1', 'tenant-abc', '/projects/myproject', 'projects', NULL);
            INSERT INTO unified_queue VALUES
                ('q1', 'tenant-abc', 'pending',     'file', 'add', 'projects', 'key1', '{}'),
                ('q2', 'tenant-abc', 'pending',     'file', 'add', 'projects', 'key2', '{}'),
                ('q3', 'tenant-abc', 'in_progress', 'file', 'add', 'projects', 'key3', '{}'),
                ('q4', 'tenant-abc', 'failed',      'file', 'add', 'projects', 'key4', '{}'),
                ('q5', 'tenant-xyz', 'pending',     'file', 'add', 'projects', 'key5', '{}');
            "#,
        )
        .unwrap();
        conn
    }

    #[test]
    fn cancel_pending_only_deletes_pending() {
        let conn = setup_db();
        let deleted = conn
            .execute(
                "DELETE FROM unified_queue \
                 WHERE tenant_id = ?1 AND status IN ('pending')",
                rusqlite::params!["tenant-abc"],
            )
            .unwrap();
        assert_eq!(deleted, 2, "should delete exactly 2 pending items");

        let remaining: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-abc'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 2, "in_progress and failed should remain");
    }

    #[test]
    fn cancel_does_not_touch_in_progress() {
        let conn = setup_db();
        conn.execute(
            "DELETE FROM unified_queue WHERE tenant_id = ?1 AND status IN ('pending', 'failed')",
            rusqlite::params!["tenant-abc"],
        )
        .unwrap();

        let in_progress: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE status = 'in_progress'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(in_progress, 1, "in_progress item must not be cancelled");
    }

    #[test]
    fn cancel_does_not_touch_other_tenants() {
        let conn = setup_db();
        conn.execute(
            "DELETE FROM unified_queue WHERE tenant_id = ?1 AND status IN ('pending')",
            rusqlite::params!["tenant-abc"],
        )
        .unwrap();

        let other: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-xyz'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(other, 1, "other tenant's items must be untouched");
    }

    #[test]
    fn cancel_unknown_project_returns_zero() {
        let conn = setup_db();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue \
                 WHERE tenant_id = 'nonexistent' AND status IN ('pending')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn resolve_tenant_by_exact_id() {
        let conn = setup_db();
        let result = super::super::super::project::resolver::resolve_tenant_by_hint(
            &conn,
            "tenant-abc",
        );
        assert!(result.is_ok());
        let (tid, _) = result.unwrap();
        assert_eq!(tid, "tenant-abc");
    }

    #[test]
    fn resolve_tenant_by_name_substring() {
        let conn = setup_db();
        let result = super::super::super::project::resolver::resolve_tenant_by_hint(
            &conn,
            "myproject",
        );
        assert!(result.is_ok());
        let (tid, _) = result.unwrap();
        assert_eq!(tid, "tenant-abc");
    }

    #[test]
    fn resolve_tenant_unknown_fails() {
        let conn = setup_db();
        let result = super::super::super::project::resolver::resolve_tenant_by_hint(
            &conn,
            "definitely-not-a-project",
        );
        assert!(result.is_err());
    }
}
