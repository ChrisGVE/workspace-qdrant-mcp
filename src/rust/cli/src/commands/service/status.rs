//! Service status subcommand
//!
//! Columnar template per cli-feedback.md.

use anyhow::Result;
use colored::Colorize;
use serde::Serialize;

use crate::grpc::client::DaemonClient;
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::{self, ServiceStatus};

/// JSON-serializable service status
#[derive(Serialize)]
struct ServiceStatusJson {
    connected: bool,
    health: String,
    components: Vec<ComponentStatusJson>,
}

/// JSON-serializable component health
#[derive(Serialize)]
struct ComponentStatusJson {
    name: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

fn status_name(s: ServiceStatus) -> &'static str {
    match s {
        ServiceStatus::Healthy => "healthy",
        ServiceStatus::Degraded => "degraded",
        ServiceStatus::Unhealthy => "unhealthy",
        ServiceStatus::Active => "active",
        ServiceStatus::Inactive => "inactive",
        ServiceStatus::Unknown => "unknown",
    }
}

fn format_status(status: ServiceStatus) -> String {
    match status {
        ServiceStatus::Healthy | ServiceStatus::Active => status_name(status).green().to_string(),
        ServiceStatus::Degraded => status_name(status).yellow().to_string(),
        ServiceStatus::Unhealthy => status_name(status).red().to_string(),
        ServiceStatus::Inactive | ServiceStatus::Unknown => {
            status_name(status).dimmed().to_string()
        }
    }
}

fn status_gutter(status: ServiceStatus) -> Gutter {
    match status {
        ServiceStatus::Healthy | ServiceStatus::Active => Gutter::Sync,
        ServiceStatus::Degraded => Gutter::Warning,
        ServiceStatus::Unhealthy => Gutter::Error,
        _ => Gutter::None,
    }
}

/// Display or serialize the health response when connected to the daemon.
async fn handle_connected(mut client: crate::grpc::client::DaemonClient, json: bool) -> Result<()> {
    match client.system().health(()).await {
        Ok(response) => {
            let health = response.into_inner();
            let overall = ServiceStatus::from_proto(health.status);

            if json {
                let components: Vec<ComponentStatusJson> = health
                    .components
                    .iter()
                    .map(|c| ComponentStatusJson {
                        name: c.component_name.clone(),
                        status: status_name(ServiceStatus::from_proto(c.status)).to_string(),
                        message: if c.message.is_empty() {
                            None
                        } else {
                            Some(c.message.clone())
                        },
                    })
                    .collect();
                output::print_json(&ServiceStatusJson {
                    connected: true,
                    health: status_name(overall).to_string(),
                    components,
                });
            } else {
                let overall_gutter = status_gutter(overall);
                let mut builder = ColumnarBuilder::new()
                    .kv_gutter(
                        "Connection",
                        format_status(ServiceStatus::Healthy),
                        Gutter::Sync,
                    )
                    .kv_gutter("Health", format_status(overall), overall_gutter);

                if !health.components.is_empty() {
                    builder = builder.section(Some("Components"));
                    for comp in &health.components {
                        let comp_status = ServiceStatus::from_proto(comp.status);
                        let gutter = status_gutter(comp_status);
                        builder = builder.kv_gutter(
                            &comp.component_name,
                            format_status(comp_status),
                            gutter,
                        );
                        if !comp.message.is_empty() {
                            builder =
                                builder.raw(&format!("  {}", comp.message.dimmed()), Gutter::None);
                        }
                    }
                }

                builder.render();
            }
        }
        Err(e) => {
            if json {
                output::print_json(&ServiceStatusJson {
                    connected: true,
                    health: "unknown".to_string(),
                    components: Vec::new(),
                });
            } else {
                ColumnarBuilder::new()
                    .kv_gutter(
                        "Connection",
                        format_status(ServiceStatus::Healthy),
                        Gutter::Sync,
                    )
                    .kv("Health", format_status(ServiceStatus::Unknown))
                    .render();
                output::warning(format!("Could not get health: {}", e));
            }
        }
    }
    Ok(())
}

/// Show daemon status, optionally as JSON
pub async fn execute(json: bool) -> Result<()> {
    if !json {
        canvas::print_title("Daemon Status");
        canvas::print_blank();
    }

    match DaemonClient::connect_default().await {
        Ok(client) => {
            handle_connected(client, json).await?;
        }
        Err(_) => {
            if json {
                output::print_json(&ServiceStatusJson {
                    connected: false,
                    health: "unhealthy".to_string(),
                    components: Vec::new(),
                });
            } else {
                ColumnarBuilder::new()
                    .kv_gutter(
                        "Connection",
                        format_status(ServiceStatus::Unhealthy),
                        Gutter::Error,
                    )
                    .render();
                output::error("Daemon not running or not reachable");
                output::info("Start with: wqm service start");
            }
        }
    }

    Ok(())
}
