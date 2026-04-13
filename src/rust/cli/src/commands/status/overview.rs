//! Default status overview subcommand.
//!
//! Uses SQLite canonical queries for aggregate metrics (collections,
//! documents, projects, queue). Only uses gRPC for daemon health and
//! resource mode info.

use anyhow::Result;

use crate::data::db::connect_readonly;
use crate::data::health;
use crate::data::queries;
use crate::grpc::client::{workspace_daemon::SystemStatusResponse, DaemonClient};
use crate::output::{self, ServiceStatus};

use super::types::{status_label, SystemStatusJson};

/// Show the default system status overview.
pub async fn default_status(
    show_queue: bool,
    show_watch: bool,
    show_performance: bool,
    json: bool,
) -> Result<()> {
    if !json {
        output::section("System Status");
    }

    // Check daemon connectivity (gRPC) — only for health + resource mode
    let (daemon_up, daemon_status) = match DaemonClient::connect_default().await {
        Ok(mut client) => {
            if !json {
                output::status_line("Daemon", ServiceStatus::Healthy);
            }
            let status = client
                .system()
                .get_status(())
                .await
                .ok()
                .map(|r| r.into_inner());
            (true, status)
        }
        Err(_) => {
            if !json {
                output::status_line("Daemon", ServiceStatus::Unhealthy);
                output::warning("Daemon not running. Start with: wqm service start");
            }
            (false, None)
        }
    };

    // Check Qdrant connectivity
    let qdrant_health = health::check_qdrant().await;
    if !json && !qdrant_health.reachable {
        output::status_line("Qdrant", ServiceStatus::Unhealthy);
        if let Some(ref err) = qdrant_health.error {
            output::warning(err);
        }
    } else if !json {
        output::status_line("Qdrant", ServiceStatus::Healthy);
    }

    // Get metrics from SQLite (canonical source of truth)
    let (collection_count, document_count, active_project_count, project_names, queue_stats) =
        match connect_readonly() {
            Ok(conn) => {
                let collections = queries::get_active_collection_count(&conn).unwrap_or(0);
                let documents = queries::get_total_document_count(&conn, "projects").unwrap_or(0);
                let active = queries::get_active_project_count(&conn).unwrap_or(0);
                let projects = queries::get_projects(&conn).unwrap_or_default();
                let names: Vec<String> = projects
                    .iter()
                    .filter(|p| p.is_active)
                    .map(|p| {
                        p.path
                            .rsplit('/')
                            .find(|s| !s.is_empty())
                            .unwrap_or(&p.tenant_id)
                            .to_string()
                    })
                    .collect();
                let q = queries::get_queue_stats(&conn).unwrap_or_default();
                (collections, documents, active, names, q)
            }
            Err(_) => {
                if !json {
                    output::warning("Database not available — metrics unavailable");
                }
                (0, 0, 0, Vec::new(), queries::QueueStats::default())
            }
        };

    // Also add Qdrant collection count if available and higher
    let total_collections = if qdrant_health.reachable {
        qdrant_health.collection_count.max(collection_count)
    } else {
        collection_count
    };

    if json {
        let json_out = SystemStatusJson {
            connected: daemon_up,
            status: if daemon_up {
                daemon_status
                    .as_ref()
                    .map(|s| status_label(ServiceStatus::from_proto(s.status)).to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            } else {
                "unhealthy".to_string()
            },
            collections: total_collections as i32,
            documents: document_count as i32,
            active_projects: project_names,
            pending_operations: Some(queue_stats.pending as i32),
            resource_mode: daemon_status.as_ref().and_then(|s| s.resource_mode.clone()),
            idle_seconds: daemon_status.as_ref().and_then(|s| s.idle_seconds),
            current_max_embeddings: daemon_status
                .as_ref()
                .and_then(|s| s.current_max_embeddings),
            current_inter_item_delay_ms: daemon_status
                .as_ref()
                .and_then(|s| s.current_inter_item_delay_ms),
        };
        output::print_json(&json_out);
        return Ok(());
    }

    // Display metrics from SQLite
    output::separator();
    output::kv("Collections", total_collections.to_string());
    output::kv("Documents", document_count.to_string());
    output::kv("Active Projects", active_project_count.to_string());
    output::kv("Pending Operations", queue_stats.pending.to_string());

    if queue_stats.failed > 0 {
        output::kv("Failed Operations", queue_stats.failed.to_string());
    }

    // Resource mode from daemon (only available via gRPC)
    if let Some(ref status) = daemon_status {
        render_resource_mode(status);
    }

    if show_queue {
        output::separator();
        super::queue::queue(false).await?;
    }
    if show_watch {
        output::separator();
        super::watch::watch().await?;
    }
    if show_performance {
        output::separator();
        super::performance::performance().await?;
    }

    Ok(())
}

fn render_resource_mode(status: &SystemStatusResponse) {
    if let Some(ref mode) = status.resource_mode {
        output::separator();
        output::kv("Resource Mode", mode);
        if let Some(idle) = status.idle_seconds {
            output::kv(
                "Idle Time",
                wqm_common::duration_fmt::format_duration(idle, 0),
            );
        }
        if let Some(max_emb) = status.current_max_embeddings {
            output::kv("Max Embeddings", max_emb.to_string());
        }
        if let Some(delay) = status.current_inter_item_delay_ms {
            output::kv("Inter-item Delay", format!("{}ms", delay));
        }
    }
}
