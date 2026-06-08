//! Dashboard data structures and async background fetcher.
//!
//! SQLite fetch functions are in `dashboard_fetch.rs`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use wqm_common::constants::TENANT_GLOBAL;

// Re-export the main fetch entry point from the fetch module.
pub use super::dashboard_fetch::fetch_dashboard_data;

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Service health state for the Services cell.
#[derive(Debug, Clone)]
pub struct ServiceHealth {
    pub qdrant_healthy: Option<bool>,
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

impl DashboardData {
    /// Sort every cell's rows by tenant display name (natural, case-insensitive)
    /// so the dashboard ordering matches the full list views. Errors sort by
    /// their tenant name. The sort is stable, so equal-named rows keep their
    /// relative order across refreshes.
    pub fn sort_by_tenant(&mut self) {
        use crate::tui::util::natural_cmp;
        self.projects.sort_by(|a, b| natural_cmp(&a.name, &b.name));
        self.libraries.sort_by(|a, b| natural_cmp(&a.name, &b.name));
        self.scratchpad
            .sort_by(|a, b| natural_cmp(&a.name, &b.name));
        self.rules.sort_by(|a, b| natural_cmp(&a.name, &b.name));
        self.active_projects
            .sort_by(|a, b| natural_cmp(&a.name, &b.name));
        self.errors
            .sort_by(|a, b| natural_cmp(&a.tenant_name, &b.tenant_name));
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
    data.health = check_services_health().await;

    if let Ok(reader) = crate::commands::qdrant_helpers::QdrantReader::from_config() {
        if let Ok(c) = reader.tenant_point_counts("projects", "tenant_id").await {
            data.projects_points = c;
        }
        if let Ok(c) = reader
            .tenant_point_counts("libraries", "library_name")
            .await
        {
            data.libraries_points = c;
        }
        if let Ok(c) = reader.tenant_point_counts("scratchpad", "tenant_id").await {
            data.scratchpad_points = c;
        }
        if let Ok(c) = reader.tenant_point_counts("rules", "tenant_id").await {
            data.rules_points = c;
        }
    }

    data
}

async fn check_services_health() -> ServiceHealth {
    let mut health = ServiceHealth::default();

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

    match crate::grpc::connect_default().await {
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
// Merge async data into sync data
// ---------------------------------------------------------------------------

/// Merge Qdrant point counts from async data into the dashboard snapshot.
pub fn merge_async_data(data: &mut DashboardData, async_data: &AsyncDashboardData) {
    let names = crate::data::tenants::name_map();
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
        &names,
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
        &names,
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
    names: &HashMap<String, String>,
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
            let name = if tid == TENANT_GLOBAL {
                TENANT_GLOBAL.to_string()
            } else {
                crate::data::tenants::display_name(names, tid)
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
    fn default_dashboard_data() {
        let d = DashboardData::default();
        assert!(d.projects.is_empty());
        assert!(d.errors.is_empty());
    }

    #[test]
    fn sort_by_tenant_orders_cells_naturally() {
        let mk = |name: &str| ProjectSummaryRow {
            tenant_id: name.into(),
            name: name.into(),
            workspace_count: 0,
            branch_count: 0,
            qdrant_points: 0,
            tracked_files: 0,
            queue_pending: 0,
            queue_in_progress: 0,
            queue_failed: 0,
        };
        let mut data = DashboardData {
            projects: vec![mk("item10"), mk("Beta"), mk("item2"), mk("alpha")],
            errors: vec![
                ErrorRow {
                    queue_id: "q2".into(),
                    collection_letter: 'P',
                    tenant_name: "zeta".into(),
                    error_message: "e".into(),
                },
                ErrorRow {
                    queue_id: "q1".into(),
                    collection_letter: 'P',
                    tenant_name: "alpha".into(),
                    error_message: "e".into(),
                },
            ],
            ..DashboardData::default()
        };
        data.sort_by_tenant();
        let names: Vec<&str> = data.projects.iter().map(|p| p.name.as_str()).collect();
        // Case-insensitive, numeric-aware: alpha, Beta, item2, item10.
        assert_eq!(names, vec!["alpha", "Beta", "item2", "item10"]);
        assert_eq!(data.errors[0].tenant_name, "alpha");
    }

    #[test]
    fn merge_async_updates_counts() {
        let mut data = DashboardData {
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
