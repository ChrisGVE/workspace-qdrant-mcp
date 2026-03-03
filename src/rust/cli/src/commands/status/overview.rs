//! Default status overview subcommand.

use anyhow::Result;

use crate::grpc::client::{workspace_daemon::SystemStatusResponse, DaemonClient};
use crate::output::{self, ServiceStatus};

use super::types::{status_label, SystemStatusJson};

/// Show the default system status overview.
///
/// When `show_queue`, `show_watch`, or `show_performance` flags are set,
/// appends additional sections after the overview.
pub async fn default_status(
    show_queue: bool,
    show_watch: bool,
    show_performance: bool,
    json: bool,
) -> Result<()> {
    if !json {
        output::section("System Status");
    }

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            if !json {
                output::status_line("Daemon", ServiceStatus::Healthy);
            }
            let done = render_daemon_status(&mut client, json).await?;
            if done {
                return Ok(());
            }
        }
        Err(_) => {
            if json {
                print_disconnected_json(false);
                return Ok(());
            }
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::error("Daemon not running. Start with: wqm service start");
        }
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

/// Returns `true` if output is complete (JSON mode early exit).
async fn render_daemon_status(client: &mut DaemonClient, json: bool) -> Result<bool> {
    match client.system().get_status(()).await {
        Ok(response) => {
            let status = response.into_inner();
            let overall = ServiceStatus::from_proto(status.status);

            if json {
                let pending = status.metrics.as_ref().map(|m| m.pending_operations);
                let json_out = SystemStatusJson {
                    connected: true,
                    status: status_label(overall).to_string(),
                    collections: status.total_collections,
                    documents: status.total_documents,
                    active_projects: status.active_projects.clone(),
                    pending_operations: pending,
                    resource_mode: status.resource_mode.clone(),
                    idle_seconds: status.idle_seconds,
                    current_max_embeddings: status.current_max_embeddings,
                    current_inter_item_delay_ms: status.current_inter_item_delay_ms,
                };
                output::print_json(&json_out);
                return Ok(true);
            }

            output::status_line("Overall", overall);
            output::separator();
            output::kv("Collections", &status.total_collections.to_string());
            output::kv("Documents", &status.total_documents.to_string());
            output::kv("Active Projects", &status.active_projects.len().to_string());

            if let Some(metrics) = &status.metrics {
                output::kv("Pending Operations", &metrics.pending_operations.to_string());
            }

            render_resource_mode(&status);
        }
        Err(e) => {
            if json {
                print_disconnected_json(true);
                return Ok(true);
            }
            output::warning(format!("Could not get status: {}", e));
        }
    }
    Ok(false)
}

fn render_resource_mode(status: &SystemStatusResponse) {
    if let Some(ref mode) = status.resource_mode {
        output::separator();
        output::kv("Resource Mode", mode);
        if let Some(idle) = status.idle_seconds {
            output::kv("Idle Time", &wqm_common::duration_fmt::format_duration(idle, 0));
        }
        if let Some(max_emb) = status.current_max_embeddings {
            output::kv("Max Embeddings", &max_emb.to_string());
        }
        if let Some(delay) = status.current_inter_item_delay_ms {
            output::kv("Inter-item Delay", &format!("{}ms", delay));
        }
    }
}

fn print_disconnected_json(connected: bool) {
    let status_str = if connected { "unknown" } else { "unhealthy" };
    let json_out = SystemStatusJson {
        connected,
        status: status_str.to_string(),
        collections: 0,
        documents: 0,
        active_projects: Vec::new(),
        pending_operations: None,
        resource_mode: None,
        idle_seconds: None,
        current_max_embeddings: None,
        current_inter_item_delay_ms: None,
    };
    output::print_json(&json_out);
}
