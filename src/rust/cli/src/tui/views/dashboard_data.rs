//! Dashboard data structures and fetch logic.
//!
//! Provides synchronous SQLite queries for the dashboard grid cells
//! and an async background thread for Qdrant health + point counts.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use rusqlite::Connection;

use crate::commands::queue::db::connect_readonly;

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Service health state for the Services cell.
#[derive(Debug, Clone)]
pub struct ServiceHealth {
    /// `None` = not yet checked, `Some(true)` = healthy.
    pub qdrant_healthy: Option<bool>,
    /// `None` = not yet checked, `Some(true)` = healthy.
    pub daemon_healthy: Option<bool>,
}

impl Default for ServiceHealth {
    fn default() -> Self {
        Self {
            qdrant_healthy: None,
            daemon_healthy: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Row types for each grid cell
// ---------------------------------------------------------------------------

/// Projects cell (1,2).
#[derive(Debug, Clone)]
pub struct ProjectSummaryRow {
    pub tenant_id: String,
    pub name: String,
    pub workspace_count: i64,
    pub branch_count: i64,
    pub qdrant_points: usize,
    pub tracked_files: i64,
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_failed: i64,
}

/// Libraries cell (2,2).
#[derive(Debug, Clone)]
pub struct LibrarySummaryRow {
    pub tenant_id: String,
    pub name: String,
    pub qdrant_points: usize,
    pub tracked_files: i64,
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_failed: i64,
    pub sync_mode: String,
}

/// Scratchpad cell (1,3).
#[derive(Debug, Clone)]
pub struct ScratchpadSummaryRow {
    pub tenant_id: String,
    pub name: String,
    pub note_count: usize,
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_failed: i64,
}

/// Rules cell (2,3).
#[derive(Debug, Clone)]
pub struct RulesSummaryRow {
    pub tenant_id: String,
    pub name: String,
    pub rule_count: usize,
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_failed: i64,
}

/// Active projects cell (1,4).
#[derive(Debug, Clone)]
pub struct ActiveProjectRow {
    pub tenant_id: String,
    pub name: String,
    pub workspace: String,
    pub tracked_files: i64,
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_failed: i64,
}

/// Last errors cell (2,4).
#[derive(Debug, Clone)]
pub struct ErrorRow {
    pub queue_id: String,
    pub collection_letter: char,
    pub tenant_name: String,
    pub error_message: String,
    pub tenant_id: String,
    pub collection: String,
}

// ---------------------------------------------------------------------------
// Async data from background thread
// ---------------------------------------------------------------------------

/// Data fetched asynchronously (Qdrant health + point counts).
#[derive(Debug, Clone, Default)]
pub struct AsyncDashboardData {
    pub health: ServiceHealth,
    pub projects_points: HashMap<String, usize>,
    pub libraries_points: HashMap<String, usize>,
    pub scratchpad_points: HashMap<String, usize>,
    pub rules_points: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Full dashboard snapshot
// ---------------------------------------------------------------------------

/// Complete data snapshot rendered by the dashboard.
#[derive(Debug, Clone)]
pub struct DashboardData {
    pub db_connected: bool,
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_failed: i64,
    pub projects: Vec<ProjectSummaryRow>,
    pub libraries: Vec<LibrarySummaryRow>,
    pub scratchpad: Vec<ScratchpadSummaryRow>,
    pub rules: Vec<RulesSummaryRow>,
    pub active_projects: Vec<ActiveProjectRow>,
    pub errors: Vec<ErrorRow>,
}

impl Default for DashboardData {
    fn default() -> Self {
        Self {
            db_connected: false,
            queue_pending: 0,
            queue_in_progress: 0,
            queue_failed: 0,
            projects: Vec::new(),
            libraries: Vec::new(),
            scratchpad: Vec::new(),
            rules: Vec::new(),
            active_projects: Vec::new(),
            errors: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Background async fetcher
// ---------------------------------------------------------------------------

/// Spawns a background thread that periodically fetches Qdrant health
/// and point counts, storing results in shared state.
pub fn spawn_async_fetcher() -> Arc<Mutex<AsyncDashboardData>> {
    let shared = Arc::new(Mutex::new(AsyncDashboardData::default()));
    let shared_clone = Arc::clone(&shared);

    thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(_) => return,
        };

        rt.block_on(async move {
            loop {
                let data = fetch_async_data().await;
                if let Ok(mut guard) = shared_clone.lock() {
                    *guard = data;
                }
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });
    });

    shared
}

async fn fetch_async_data() -> AsyncDashboardData {
    let mut data = AsyncDashboardData::default();

    // Health checks
    data.health = check_services_health().await;

    // Qdrant point counts per collection per tenant
    if let Ok(client) = crate::commands::qdrant_helpers::build_qdrant_http_client() {
        let base_url = crate::commands::qdrant_helpers::qdrant_base_url();

        if let Ok(counts) = crate::commands::qdrant_helpers::scroll_tenant_point_counts(
            &client,
            &base_url,
            "projects",
            "tenant_id",
        )
        .await
        {
            data.projects_points = counts;
        }
        if let Ok(counts) = crate::commands::qdrant_helpers::scroll_tenant_point_counts(
            &client,
            &base_url,
            "libraries",
            "library_name",
        )
        .await
        {
            data.libraries_points = counts;
        }
        if let Ok(counts) = crate::commands::qdrant_helpers::scroll_tenant_point_counts(
            &client,
            &base_url,
            "scratchpad",
            "tenant_id",
        )
        .await
        {
            data.scratchpad_points = counts;
        }
        if let Ok(counts) = crate::commands::qdrant_helpers::scroll_tenant_point_counts(
            &client,
            &base_url,
            "rules",
            "tenant_id",
        )
        .await
        {
            data.rules_points = counts;
        }
    }

    data
}

async fn check_services_health() -> ServiceHealth {
    let mut health = ServiceHealth::default();

    // Qdrant health: try GET /collections (lightweight)
    if let Ok(client) = crate::commands::qdrant_helpers::build_qdrant_http_client() {
        let url = format!(
            "{}/collections",
            crate::commands::qdrant_helpers::qdrant_base_url()
        );
        health.qdrant_healthy = Some(
            client
                .get(&url)
                .timeout(std::time::Duration::from_secs(3))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false),
        );
    }

    // Daemon health: try gRPC connect + health RPC
    match crate::grpc::client::DaemonClient::connect_default().await {
        Ok(mut client) => {
            health.daemon_healthy = Some(client.system().health(()).await.is_ok());
        }
        Err(_) => {
            health.daemon_healthy = Some(false);
        }
    }

    health
}

// ---------------------------------------------------------------------------
// SQLite fetch: full dashboard snapshot
// ---------------------------------------------------------------------------

/// Fetch all dashboard data from SQLite (synchronous).
pub fn fetch_dashboard_data() -> DashboardData {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return DashboardData::default(),
    };

    let mut data = DashboardData {
        db_connected: true,
        ..DashboardData::default()
    };

    fetch_queue_stats(&conn, &mut data);
    fetch_projects(&conn, &mut data);
    fetch_libraries(&conn, &mut data);
    fetch_scratchpad_queue(&conn, &mut data);
    fetch_rules_queue(&conn, &mut data);
    fetch_active_projects(&conn, &mut data);
    fetch_errors(&conn, &mut data);

    data
}

// ---------------------------------------------------------------------------
// Individual fetch helpers
// ---------------------------------------------------------------------------

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
    // Top-level projects
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

    // Workspace counts
    let ws_counts = count_by_tenant(
        conn,
        "SELECT tenant_id, COUNT(*) FROM watch_folders \
         WHERE collection = 'projects' GROUP BY tenant_id",
    );

    // Branch counts
    let br_counts = count_by_tenant(
        conn,
        "SELECT tenant_id, COUNT(DISTINCT branch) FROM unified_queue \
         WHERE collection = 'projects' GROUP BY tenant_id",
    );

    // Tracked file counts
    let tf_counts = count_by_tenant(
        conn,
        "SELECT wf.tenant_id, COUNT(tf.file_id) FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.collection = 'projects' GROUP BY wf.tenant_id",
    );

    // Queue counts per tenant+status
    let q_counts = queue_counts_by_tenant(conn, "projects");

    data.projects = projects
        .into_iter()
        .map(|(tid, path)| {
            let name = path_last_component(&path);
            let q = q_counts.get(&tid);
            ProjectSummaryRow {
                workspace_count: *ws_counts.get(&tid).unwrap_or(&0),
                branch_count: *br_counts.get(&tid).unwrap_or(&0),
                qdrant_points: 0, // filled from async data
                tracked_files: *tf_counts.get(&tid).unwrap_or(&0),
                queue_pending: q.map_or(0, |m| *m.get("pending").unwrap_or(&0)),
                queue_in_progress: q.map_or(0, |m| *m.get("in_progress").unwrap_or(&0)),
                queue_failed: q.map_or(0, |m| *m.get("failed").unwrap_or(&0)),
                name,
                tenant_id: tid,
            }
        })
        .collect();
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

    let tf_counts = count_by_tenant(
        conn,
        "SELECT wf.tenant_id, COUNT(tf.file_id) FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.collection = 'libraries' GROUP BY wf.tenant_id",
    );

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
                queue_pending: q.map_or(0, |m| *m.get("pending").unwrap_or(&0)),
                queue_in_progress: q.map_or(0, |m| *m.get("in_progress").unwrap_or(&0)),
                queue_failed: q.map_or(0, |m| *m.get("failed").unwrap_or(&0)),
                sync_mode: mode_short,
                tenant_id: tid,
            }
        })
        .collect();
}

fn fetch_scratchpad_queue(conn: &Connection, data: &mut DashboardData) {
    let q_counts = queue_counts_by_tenant(conn, "scratchpad");
    let tenant_names = resolve_tenant_names(conn);

    // Collect all tenants that have scratchpad queue items
    let mut tenants: Vec<String> = q_counts.keys().cloned().collect();
    tenants.sort();

    data.scratchpad = tenants
        .into_iter()
        .map(|tid| {
            let name = if tid == "global" {
                "global".to_string()
            } else {
                tenant_names.get(&tid).cloned().unwrap_or(tid.clone())
            };
            let q = q_counts.get(&tid);
            ScratchpadSummaryRow {
                note_count: 0, // filled from async
                queue_pending: q.map_or(0, |m| *m.get("pending").unwrap_or(&0)),
                queue_in_progress: q.map_or(0, |m| *m.get("in_progress").unwrap_or(&0)),
                queue_failed: q.map_or(0, |m| *m.get("failed").unwrap_or(&0)),
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
            let name = if tid == "global" {
                "global".to_string()
            } else {
                tenant_names.get(&tid).cloned().unwrap_or(tid.clone())
            };
            let q = q_counts.get(&tid);
            RulesSummaryRow {
                rule_count: 0,
                queue_pending: q.map_or(0, |m| *m.get("pending").unwrap_or(&0)),
                queue_in_progress: q.map_or(0, |m| *m.get("in_progress").unwrap_or(&0)),
                queue_failed: q.map_or(0, |m| *m.get("failed").unwrap_or(&0)),
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

    let tf_counts = count_by_tenant(
        conn,
        "SELECT wf.tenant_id, COUNT(tf.file_id) FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.collection = 'projects' GROUP BY wf.tenant_id",
    );

    let q_counts = queue_counts_by_tenant(conn, "projects");

    data.active_projects = actives
        .into_iter()
        .map(|(tid, path)| {
            let name = path_last_component(&path);
            let display = crate::output::style::home_to_tilde(&path);
            let q = q_counts.get(&tid);
            ActiveProjectRow {
                tracked_files: *tf_counts.get(&tid).unwrap_or(&0),
                queue_pending: q.map_or(0, |m| *m.get("pending").unwrap_or(&0)),
                queue_in_progress: q.map_or(0, |m| *m.get("in_progress").unwrap_or(&0)),
                queue_failed: q.map_or(0, |m| *m.get("failed").unwrap_or(&0)),
                name,
                workspace: display,
                tenant_id: tid,
            }
        })
        .collect();
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
                tenant_id: tid,
                collection: coll,
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
    let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE parent_watch_id IS NULL",
    ) else {
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

fn path_last_component(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn collection_letter(collection: &str) -> char {
    match collection {
        "projects" => 'P',
        "libraries" => 'L',
        "scratchpad" => 'S',
        "rules" => 'R',
        _ => '?',
    }
}

// ---------------------------------------------------------------------------
// Merge async data into sync data
// ---------------------------------------------------------------------------

/// Merge Qdrant point counts from async data into the dashboard snapshot.
pub fn merge_async_data(data: &mut DashboardData, async_data: &AsyncDashboardData) {
    for p in &mut data.projects {
        if let Some(&count) = async_data.projects_points.get(&p.tenant_id) {
            p.qdrant_points = count;
        }
    }
    for l in &mut data.libraries {
        if let Some(&count) = async_data.libraries_points.get(&l.tenant_id) {
            l.qdrant_points = count;
        }
    }
    for s in &mut data.scratchpad {
        if let Some(&count) = async_data.scratchpad_points.get(&s.tenant_id) {
            s.note_count = count;
        }
    }
    for r in &mut data.rules {
        if let Some(&count) = async_data.rules_points.get(&r.tenant_id) {
            r.rule_count = count;
        }
    }

    // Add scratchpad/rules tenants that only exist in Qdrant (no queue items)
    add_qdrant_only_tenants(
        &async_data.scratchpad_points,
        &mut data.scratchpad,
        |tid, name, count| ScratchpadSummaryRow {
            tenant_id: tid,
            name,
            note_count: count,
            queue_pending: 0,
            queue_in_progress: 0,
            queue_failed: 0,
        },
    );
    add_qdrant_only_tenants(
        &async_data.rules_points,
        &mut data.rules,
        |tid, name, count| RulesSummaryRow {
            tenant_id: tid,
            name,
            rule_count: count,
            queue_pending: 0,
            queue_in_progress: 0,
            queue_failed: 0,
        },
    );
}

fn add_qdrant_only_tenants<T, F>(
    qdrant_counts: &HashMap<String, usize>,
    existing: &mut Vec<T>,
    make_row: F,
) where
    T: HasTenantId,
    F: Fn(String, String, usize) -> T,
{
    let existing_tids: std::collections::HashSet<&str> =
        existing.iter().map(|r| r.tenant_id()).collect();

    let mut new_rows: Vec<T> = qdrant_counts
        .iter()
        .filter(|(tid, _)| !existing_tids.contains(tid.as_str()))
        .map(|(tid, &count)| {
            let name = if tid == "global" {
                "global".to_string()
            } else {
                tid.clone()
            };
            make_row(tid.clone(), name, count)
        })
        .collect();

    existing.append(&mut new_rows);
    existing.sort_by(|a, b| a.tenant_id().cmp(b.tenant_id()));
}

trait HasTenantId {
    fn tenant_id(&self) -> &str;
}

impl HasTenantId for ScratchpadSummaryRow {
    fn tenant_id(&self) -> &str {
        &self.tenant_id
    }
}

impl HasTenantId for RulesSummaryRow {
    fn tenant_id(&self) -> &str {
        &self.tenant_id
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
    fn default_dashboard_data() {
        let d = DashboardData::default();
        assert!(!d.db_connected);
        assert!(d.projects.is_empty());
        assert!(d.errors.is_empty());
    }

    #[test]
    fn merge_async_updates_counts() {
        let mut data = DashboardData {
            db_connected: true,
            projects: vec![ProjectSummaryRow {
                tenant_id: "t1".into(),
                name: "proj".into(),
                workspace_count: 1,
                branch_count: 1,
                qdrant_points: 0,
                tracked_files: 5,
                queue_pending: 0,
                queue_in_progress: 0,
                queue_failed: 0,
            }],
            ..DashboardData::default()
        };
        let async_data = AsyncDashboardData {
            projects_points: [("t1".into(), 42)].into(),
            ..Default::default()
        };
        merge_async_data(&mut data, &async_data);
        assert_eq!(data.projects[0].qdrant_points, 42);
    }
}
