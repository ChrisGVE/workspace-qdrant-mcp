//! Default status overview subcommand.
//!
//! Columnar template per cli-feedback.md. Uses SQLite for metrics,
//! gRPC only for daemon health and resource mode.

use anyhow::Result;
use colored::Colorize;

use crate::data::db::connect_readonly;
use crate::data::health;
use crate::data::queries::{self, HealthLevel};
use crate::grpc::client::{workspace_daemon::SystemStatusResponse, DaemonClient};
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::number::{format_usize, NumberLocale};
use crate::output::{self, ServiceStatus};

use super::types::{status_label, SystemStatusJson};

/// Show the default system status overview.
pub async fn default_status(
    show_queue: bool,
    show_watch: bool,
    show_performance: bool,
    json: bool,
) -> Result<()> {
    // Check daemon connectivity (gRPC)
    let (daemon_up, daemon_status) = match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let status = client
                .system()
                .get_status(())
                .await
                .ok()
                .map(|r| r.into_inner());
            (true, status)
        }
        Err(_) => (false, None),
    };

    // Check Qdrant connectivity
    let qdrant_health = health::check_qdrant().await;

    // Get metrics from SQLite
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
            Err(_) => (0, 0, 0, Vec::new(), queries::QueueStats::default()),
        };

    let total_collections = if qdrant_health.reachable {
        qdrant_health.collection_count.max(collection_count)
    } else {
        collection_count
    };

    // Compute health levels
    let worker_health = if daemon_up {
        HealthLevel::Healthy
    } else {
        HealthLevel::Unhealthy
    };
    let qdrant_level = if qdrant_health.reachable {
        HealthLevel::Healthy
    } else {
        HealthLevel::Unhealthy
    };
    let queue_health = queue_stats.health();
    let overall = worker_health.worst(qdrant_level).worst(queue_health);

    // JSON mode
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

    canvas::print_title("System Status");
    canvas::print_blank();

    let locale = NumberLocale::default();

    // Build columnar display
    let total_queue = queue_stats.pending + queue_stats.in_progress + queue_stats.failed;

    let mut builder = ColumnarBuilder::new()
        .kv("Overall", format_health(overall))
        .section(Some("Services"))
        .kv("Workspace-Qdrant Worker", format_health(worker_health))
        .kv("Workspace-Qdrant Queue", format_health(queue_health))
        .kv("Qdrant Server", format_health(qdrant_level))
        .section(Some("Metrics"))
        .kv("Collections", format_usize(total_collections, &locale))
        .kv("Documents", format_usize(document_count, &locale))
        .kv(
            "Active Projects",
            format_usize(active_project_count, &locale),
        )
        .section(Some("Queue"))
        .kv("Total", format_usize(total_queue, &locale));

    // Queue decomposition — right-aligned values as nested group
    {
        let mut decomp: Vec<(&str, String, Gutter)> = Vec::new();
        decomp.push((
            "Pending",
            format_usize(queue_stats.pending, &locale),
            Gutter::Add,
        ));
        decomp.push((
            "In Progress",
            format_usize(queue_stats.in_progress, &locale),
            Gutter::Update,
        ));
        decomp.push((
            "Failed",
            format_usize(queue_stats.failed, &locale),
            Gutter::Remove,
        ));

        let inner = ColumnarBuilder::new().aligned_group(decomp);
        builder = builder.nested("", inner);
    }

    // Resource mode from daemon (only via gRPC)
    if let Some(ref status) = daemon_status {
        if status.resource_mode.is_some() {
            builder = add_resource_mode(builder, status);
        }
    }

    builder.render();

    // Warnings after columnar block
    if !daemon_up {
        output::warning("Worker not running. Start with: wqm service start");
    }
    if !qdrant_health.reachable {
        if let Some(ref err) = qdrant_health.error {
            output::warning(err);
        }
    }

    // Optional additional sections
    if show_queue {
        println!();
        super::queue::queue(false).await?;
    }
    if show_watch {
        println!();
        super::watch::watch().await?;
    }
    if show_performance {
        println!();
        super::performance::performance().await?;
    }

    Ok(())
}

/// Format a health level with color (no gutter — inline colored text).
fn format_health(level: HealthLevel) -> String {
    match level {
        HealthLevel::Healthy => "healthy".green().to_string(),
        HealthLevel::Degraded => "degraded".yellow().to_string(),
        HealthLevel::Unhealthy => "unhealthy".red().to_string(),
    }
}

fn add_resource_mode(
    mut builder: ColumnarBuilder,
    status: &SystemStatusResponse,
) -> ColumnarBuilder {
    builder = builder.section(Some("Resource Mode"));

    if let Some(ref mode) = status.resource_mode {
        builder = builder.kv("Mode", mode);
    }
    if let Some(idle) = status.idle_seconds {
        builder = builder.kv(
            "Idle Time",
            wqm_common::duration_fmt::format_duration(idle, 0),
        );
    }
    if let Some(max_emb) = status.current_max_embeddings {
        builder = builder.kv("Max Embeddings", max_emb.to_string());
    }
    if let Some(delay) = status.current_inter_item_delay_ms {
        builder = builder.kv("Inter-Item Delay", format!("{}ms", delay));
    }
    builder
}
