//! SQLite fetch functions for dashboard data.

use std::collections::HashMap;

use rusqlite::Connection;
use wqm_common::constants::TENANT_GLOBAL;

use crate::data::db::connect_readonly;

use super::dashboard_data::{
    ActiveProjectRow, DashboardData, ErrorRow, LibrarySummaryRow, ProjectSummaryRow,
    RulesSummaryRow, ScratchpadSummaryRow,
};

/// Fetch all dashboard data from SQLite (synchronous).
pub fn fetch_dashboard_data() -> DashboardData {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return DashboardData::default(),
    };

    let mut data = DashboardData::default();

    fetch_queue_stats(&conn, &mut data);
    fetch_projects(&conn, &mut data);
    fetch_libraries(&conn, &mut data);
    fetch_scratchpad_queue(&conn, &mut data);
    fetch_rules_queue(&conn, &mut data);
    fetch_active_projects(&conn, &mut data);
    fetch_errors(&conn, &mut data);

    data
}

fn fetch_queue_stats(conn: &Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare("SELECT status, COUNT(*) FROM unified_queue GROUP BY status")
    else {
        return;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }) else {
        return;
    };
    for row in rows.flatten() {
        match row.0.as_str() {
            "pending" => data.queue_pending = row.1,
            "in_progress" => data.queue_in_progress = row.1,
            "failed" => data.queue_failed = row.1,
            _ => {}
        }
    }
}

fn fetch_projects(conn: &Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects' \
         ORDER BY path",
    ) else {
        return;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return;
    };
    let projects: Vec<(String, String)> = rows.flatten().collect();

    let branch_info = branch_info_by_tenant(conn, "projects");
    let tf_counts = tracked_file_counts(conn, "projects");
    let q_counts = queue_counts_by_tenant(conn, "projects");

    data.projects = projects
        .into_iter()
        .map(|(tid, path)| {
            let name = path_last_component(&path);
            let q = q_counts.get(&tid);
            let bi = branch_info.get(&tid);
            ProjectSummaryRow {
                branch_count: bi.map_or(0, |b| b.count),
                qdrant_points: 0,
                tracked_files: *tf_counts.get(&tid).unwrap_or(&0),
                queue_pending: q_val(q, "pending"),
                queue_in_progress: q_val(q, "in_progress"),
                queue_failed: q_val(q, "failed"),
                name,
                tenant_id: tid,
            }
        })
        .collect();
    data.projects.sort_by_key(|p| p.name.to_lowercase());
}

fn fetch_libraries(conn: &Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare(
        "SELECT wf.tenant_id, wf.path, COALESCE(wf.library_mode, 'incremental') \
         FROM watch_folders wf WHERE wf.collection = 'libraries' ORDER BY wf.tenant_id",
    ) else {
        return;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    }) else {
        return;
    };
    let libs: Vec<(String, String, String)> = rows.flatten().collect();

    let tf_counts = tracked_file_counts(conn, "libraries");
    let q_counts = queue_counts_by_tenant(conn, "libraries");

    data.libraries = libs
        .into_iter()
        .map(|(tid, _path, mode)| {
            let q = q_counts.get(&tid);
            let mode_short = match mode.as_str() {
                "sync" => "sync".to_string(),
                "incremental" => "inc".to_string(),
                _ => mode.clone(),
            };
            LibrarySummaryRow {
                name: tid.clone(),
                qdrant_points: 0,
                tracked_files: *tf_counts.get(&tid).unwrap_or(&0),
                queue_pending: q_val(q, "pending"),
                queue_in_progress: q_val(q, "in_progress"),
                queue_failed: q_val(q, "failed"),
                sync_mode: mode_short,
                tenant_id: tid,
            }
        })
        .collect();
    data.libraries.sort_by_key(|lib| lib.name.to_lowercase());
}

fn fetch_scratchpad_queue(conn: &Connection, data: &mut DashboardData) {
    let q_counts = queue_counts_by_tenant(conn, "scratchpad");
    let tenant_names = resolve_tenant_names(conn);

    let mut tenants: Vec<String> = q_counts.keys().cloned().collect();
    tenants.sort();

    data.scratchpad = tenants
        .into_iter()
        .map(|tid| {
            let name = tenant_display_name(&tid, &tenant_names);
            let q = q_counts.get(&tid);
            ScratchpadSummaryRow {
                note_count: 0,
                queue_pending: q_val(q, "pending"),
                queue_in_progress: q_val(q, "in_progress"),
                queue_failed: q_val(q, "failed"),
                name,
                tenant_id: tid,
            }
        })
        .collect();
}

fn fetch_rules_queue(conn: &Connection, data: &mut DashboardData) {
    let q_counts = queue_counts_by_tenant(conn, "rules");
    let tenant_names = resolve_tenant_names(conn);

    let mut tenants: Vec<String> = q_counts.keys().cloned().collect();
    tenants.sort();

    data.rules = tenants
        .into_iter()
        .map(|tid| {
            let name = tenant_display_name(&tid, &tenant_names);
            let q = q_counts.get(&tid);
            RulesSummaryRow {
                rule_count: 0,
                queue_pending: q_val(q, "pending"),
                queue_in_progress: q_val(q, "in_progress"),
                queue_failed: q_val(q, "failed"),
                name,
                tenant_id: tid,
            }
        })
        .collect();
}

fn fetch_active_projects(conn: &Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE is_active > 0 AND collection = 'projects' AND parent_watch_id IS NULL \
         ORDER BY path",
    ) else {
        return;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return;
    };
    let actives: Vec<(String, String)> = rows.flatten().collect();

    let tf_counts = tracked_file_counts(conn, "projects");
    let q_counts = queue_counts_by_tenant(conn, "projects");
    let branch_info = branch_info_by_tenant(conn, "projects");

    data.active_projects = actives
        .into_iter()
        .map(|(tid, path)| {
            let name = path_last_component(&path);
            let q = q_counts.get(&tid);
            let branch = branch_info
                .get(&tid)
                .map_or_else(|| "—".to_string(), |b| b.primary.clone());
            ActiveProjectRow {
                tracked_files: *tf_counts.get(&tid).unwrap_or(&0),
                queue_pending: q_val(q, "pending"),
                queue_in_progress: q_val(q, "in_progress"),
                queue_failed: q_val(q, "failed"),
                name,
                branch,
                tenant_id: tid,
            }
        })
        .collect();
    data.active_projects.sort_by_key(|p| p.name.to_lowercase());
}

fn fetch_errors(conn: &Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare(
        "SELECT queue_id, collection, tenant_id, error_message \
         FROM unified_queue \
         WHERE status = 'failed' AND error_message IS NOT NULL \
         ORDER BY updated_at DESC LIMIT 50",
    ) else {
        return;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    }) else {
        return;
    };

    let tenant_names = resolve_tenant_names(conn);

    data.errors = rows
        .flatten()
        .map(|(qid, coll, tid, msg)| {
            let letter = collection_letter(&coll);
            let tname = tenant_names.get(&tid).cloned().unwrap_or(tid.clone());
            ErrorRow {
                queue_id: qid,
                collection_letter: letter,
                tenant_name: tname,
                error_message: msg,
            }
        })
        .collect();
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn count_by_tenant(conn: &Connection, sql: &str) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    let Ok(mut stmt) = conn.prepare(sql) else {
        return map;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }) else {
        return map;
    };
    for row in rows.flatten() {
        map.insert(row.0, row.1);
    }
    map
}

/// Per-tenant branch summary derived from `tracked_files`.
pub struct BranchInfo {
    /// Number of distinct branches indexed.
    pub count: i64,
    /// The branch with the most indexed files (the "current" branch proxy).
    pub primary: String,
}

/// Build a tenant → branch summary from `tracked_files.primary_branch`. The
/// branch count comes from the actual indexed branches (not the transient
/// queue, which is usually empty), and the primary branch is the one with the
/// most indexed files.
fn branch_info_by_tenant(conn: &Connection, collection: &str) -> HashMap<String, BranchInfo> {
    let mut per_tenant: HashMap<String, HashMap<String, i64>> = HashMap::new();
    let sql = format!(
        "SELECT wf.tenant_id, tf.primary_branch, COUNT(*) \
         FROM tracked_files tf JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.collection = '{}' AND tf.primary_branch IS NOT NULL AND tf.primary_branch <> '' \
         GROUP BY wf.tenant_id, tf.primary_branch",
        collection
    );
    if let Ok(mut stmt) = conn.prepare(&sql) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        }) {
            for (tid, branch, n) in rows.flatten() {
                per_tenant.entry(tid).or_default().insert(branch, n);
            }
        }
    }

    per_tenant
        .into_iter()
        .map(|(tid, branches)| {
            let count = branches.len() as i64;
            let primary = branches
                .into_iter()
                .max_by_key(|(_, n)| *n)
                .map(|(b, _)| b)
                .unwrap_or_default();
            (tid, BranchInfo { count, primary })
        })
        .collect()
}

fn tracked_file_counts(conn: &Connection, collection: &str) -> HashMap<String, i64> {
    count_by_tenant(
        conn,
        &format!(
            "SELECT wf.tenant_id, COUNT(tf.file_id) FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.collection = '{}' GROUP BY wf.tenant_id",
            collection
        ),
    )
}

fn queue_counts_by_tenant(
    conn: &Connection,
    collection: &str,
) -> HashMap<String, HashMap<String, i64>> {
    let mut map: HashMap<String, HashMap<String, i64>> = HashMap::new();
    let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, status, COUNT(*) FROM unified_queue \
         WHERE collection = ?1 AND status IN ('pending','in_progress','failed') \
         GROUP BY tenant_id, status",
    ) else {
        return map;
    };
    let Ok(rows) = stmt.query_map(rusqlite::params![collection], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    }) else {
        return map;
    };
    for row in rows.flatten() {
        map.entry(row.0).or_default().insert(row.1, row.2);
    }
    map
}

fn resolve_tenant_names(conn: &Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let Ok(mut stmt) =
        conn.prepare("SELECT tenant_id, path FROM watch_folders WHERE parent_watch_id IS NULL")
    else {
        return map;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return map;
    };
    for row in rows.flatten() {
        let name = path_last_component(&row.1);
        map.insert(row.0, name);
    }
    map
}

fn q_val(q: Option<&HashMap<String, i64>>, status: &str) -> i64 {
    q.map_or(0, |m| *m.get(status).unwrap_or(&0))
}

fn tenant_display_name(tid: &str, names: &HashMap<String, String>) -> String {
    if tid == TENANT_GLOBAL {
        TENANT_GLOBAL.to_string()
    } else {
        names.get(tid).cloned().unwrap_or(tid.to_string())
    }
}

pub fn path_last_component(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

pub fn collection_letter(collection: &str) -> char {
    match collection {
        "projects" => 'P',
        "libraries" => 'L',
        "scratchpad" => 'S',
        "rules" => 'R',
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_last_component_extracts_name() {
        assert_eq!(path_last_component("/foo/bar/baz"), "baz");
        assert_eq!(path_last_component("single"), "single");
    }

    #[test]
    fn collection_letter_mapping() {
        assert_eq!(collection_letter("projects"), 'P');
        assert_eq!(collection_letter("libraries"), 'L');
        assert_eq!(collection_letter("scratchpad"), 'S');
        assert_eq!(collection_letter("rules"), 'R');
        assert_eq!(collection_letter("unknown"), '?');
    }

    #[test]
    fn q_val_extracts_count() {
        let mut m = HashMap::new();
        m.insert("pending".to_string(), 5i64);
        assert_eq!(q_val(Some(&m), "pending"), 5);
        assert_eq!(q_val(Some(&m), "failed"), 0);
        assert_eq!(q_val(None, "pending"), 0);
    }

    #[test]
    fn tenant_display_global() {
        let names = HashMap::new();
        assert_eq!(tenant_display_name("global", &names), "global");
    }
}
