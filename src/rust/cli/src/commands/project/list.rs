//! List all registered projects

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use tabled::Tabled;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::ListProjectsRequest;
use crate::output::style::home_to_tilde;
use crate::output::{self, ColumnHints};

use super::super::qdrant_helpers;

/// Project table row
#[derive(Tabled)]
struct ProjectRow {
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "Path")]
    path: String,
    #[tabled(rename = "Status")]
    status: String,
    #[tabled(rename = "Documents")]
    documents: String,
}

impl ColumnHints for ProjectRow {
    fn content_columns() -> &'static [usize] {
        &[1] // Path is the content column
    }
}

/// Get document counts per tenant from SQLite tracked_files table.
///
/// This is O(n_tenants) via a GROUP BY query instead of scrolling all Qdrant
/// points (which was O(total_points) and took 20+ seconds).
fn get_document_counts_from_db() -> HashMap<String, usize> {
    let conn = match qdrant_helpers::open_state_db() {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };

    let mut counts = HashMap::new();

    // Count tracked files per tenant (join through watch_folders for tenant_id)
    let mut stmt = match conn.prepare(
        "SELECT wf.tenant_id, COUNT(tf.file_id) \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE tf.collection = 'projects' \
         GROUP BY wf.tenant_id",
    ) {
        Ok(s) => s,
        Err(_) => return counts,
    };

    if let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
    }) {
        for row in rows.flatten() {
            counts.insert(row.0, row.1);
        }
    }

    counts
}

pub(super) async fn list_projects(active_only: bool, priority: Option<String>) -> Result<()> {
    output::section("Registered Projects");

    // Get document counts from SQLite (instant) instead of scrolling Qdrant (slow)
    let doc_counts = get_document_counts_from_db();

    // Also get Qdrant tenant IDs for orphan detection (fast: just unique field values)
    let qdrant_tenant_ids = match qdrant_helpers::build_qdrant_http_client() {
        Ok(client) => {
            let base_url = qdrant_helpers::qdrant_base_url();
            // Use the state DB to get known tenants instead of scrolling Qdrant
            match qdrant_helpers::open_state_db() {
                Ok(conn) => qdrant_helpers::get_known_tenants_for_collection(
                    &conn,
                    wqm_common::constants::COLLECTION_PROJECTS,
                )
                .unwrap_or_default(),
                Err(_) => {
                    // Fall back to Qdrant scroll for orphan detection only
                    qdrant_helpers::scroll_unique_field_values(
                        &client,
                        &base_url,
                        wqm_common::constants::COLLECTION_PROJECTS,
                        "tenant_id",
                    )
                    .await
                    .unwrap_or_default()
                }
            }
        }
        Err(_) => HashSet::new(),
    };

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = ListProjectsRequest {
                priority_filter: priority,
                active_only,
            };

            match client.project().list_projects(request).await {
                Ok(response) => {
                    let list = response.into_inner();
                    print_project_list(&list, &doc_counts, &qdrant_tenant_ids);
                }
                Err(e) => {
                    output::error(format!("Failed to list projects: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

fn print_project_list(
    list: &crate::grpc::proto::ListProjectsResponse,
    doc_counts: &HashMap<String, usize>,
    qdrant_tenant_ids: &HashSet<String>,
) {
    if list.projects.is_empty() && qdrant_tenant_ids.is_empty() {
        output::info("No projects registered. Use `wqm project register` to add one.");
        return;
    }

    let mut known_ids = HashSet::new();

    // Sort: active first, then by name (case-insensitive)
    let mut projects: Vec<_> = list.projects.iter().collect();
    projects.sort_by(|a, b| {
        b.is_active.cmp(&a.is_active).then_with(|| {
            a.project_name
                .to_lowercase()
                .cmp(&b.project_name.to_lowercase())
        })
    });

    let mut rows: Vec<ProjectRow> = projects
        .iter()
        .map(|proj| {
            known_ids.insert(proj.project_id.clone());
            let name = if proj.is_worktree {
                format!("{} [worktree]", proj.project_name)
            } else {
                proj.project_name.clone()
            };
            let status = if proj.is_active {
                "Active".to_string()
            } else {
                "Inactive".to_string()
            };
            let documents = doc_counts
                .get(&proj.project_id)
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".to_string());
            ProjectRow {
                name,
                path: home_to_tilde(&proj.project_root),
                status,
                documents,
            }
        })
        .collect();

    // Add orphaned projects (in Qdrant but not tracked by daemon)
    let mut orphan_ids: Vec<&String> = qdrant_tenant_ids
        .iter()
        .filter(|id| !known_ids.contains(*id))
        .collect();
    orphan_ids.sort();

    for id in &orphan_ids {
        let documents = doc_counts
            .get(*id)
            .map(|c| c.to_string())
            .unwrap_or_else(|| "-".to_string());
        rows.push(ProjectRow {
            name: format!("{} [orphan]", &id[..id.len().min(12)]),
            path: "-".to_string(),
            status: "Orphan".to_string(),
            documents,
        });
    }

    if !rows.is_empty() {
        output::print_table_auto(&rows);
    }

    output::separator();
    let total = list.total_count as usize + orphan_ids.len();
    output::summary(output::summary_line(total, total, "projects"));
}
