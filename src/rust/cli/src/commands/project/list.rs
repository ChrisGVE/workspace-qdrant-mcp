//! List all registered projects

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use wqm_common::constants::COLLECTION_PROJECTS;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::ListProjectsRequest;
use crate::output::{self, ServiceStatus};

use super::super::qdrant_helpers;

pub(super) async fn list_projects(active_only: bool, priority: Option<String>) -> Result<()> {
    output::section("Registered Projects");

    // Try to get Qdrant point counts per tenant (non-fatal if Qdrant is down)
    let qdrant_counts = match qdrant_helpers::build_qdrant_http_client() {
        Ok(client) => {
            let base_url = qdrant_helpers::qdrant_base_url();
            qdrant_helpers::scroll_tenant_point_counts(
                &client,
                &base_url,
                COLLECTION_PROJECTS,
                "tenant_id",
            )
            .await
            .unwrap_or_default()
        }
        Err(_) => HashMap::new(),
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
                    print_project_list(&list, &qdrant_counts);
                }
                Err(e) => {
                    output::error(format!("Failed to list projects: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
            print_orphans_without_daemon(&qdrant_counts);
        }
    }

    Ok(())
}

fn print_project_list(
    list: &crate::grpc::proto::ListProjectsResponse,
    qdrant_counts: &HashMap<String, usize>,
) {
    if list.projects.is_empty() && qdrant_counts.is_empty() {
        output::info("No projects registered");
        output::info("Register a project with: wqm project register [path]");
        return;
    }

    let mut known_ids = HashSet::new();

    for proj in &list.projects {
        known_ids.insert(proj.project_id.clone());
        let status = if proj.is_active {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Unknown
        };
        output::status_line(&proj.project_name, status);
        output::kv("  ID", &proj.project_id);
        output::kv("  Path", &proj.project_root);
        output::kv("  Priority", &proj.priority);
        output::kv("  Active", if proj.is_active { "Yes" } else { "No" });
        if let Some(count) = qdrant_counts.get(&proj.project_id) {
            output::kv("  Points", &count.to_string());
        }
    }

    // Show orphaned projects (in Qdrant but not in daemon)
    let mut orphan_ids: Vec<(&String, &usize)> = qdrant_counts
        .iter()
        .filter(|(id, _)| !known_ids.contains(*id))
        .collect();
    orphan_ids.sort_by_key(|(id, _)| (*id).clone());

    if !orphan_ids.is_empty() {
        output::separator();
        output::warning(format!("Orphaned projects ({}):", orphan_ids.len()));
        for (id, count) in &orphan_ids {
            output::kv(&format!("  {} (ORPHAN)", id), &format!("{} points", count));
        }
        output::info("Run: wqm admin cleanup-orphans");
    }

    output::separator();
    output::info(format!("Total: {} projects", list.total_count));
}

fn print_orphans_without_daemon(qdrant_counts: &HashMap<String, usize>) {
    if !qdrant_counts.is_empty() {
        output::separator();
        output::warning(format!(
            "Qdrant has {} tenant(s) with no daemon running:",
            qdrant_counts.len()
        ));
        let mut sorted: Vec<_> = qdrant_counts.iter().collect();
        sorted.sort_by_key(|(id, _)| (*id).clone());
        for (id, count) in sorted {
            output::kv(&format!("  {}", id), &format!("{} points", count));
        }
    }
}
