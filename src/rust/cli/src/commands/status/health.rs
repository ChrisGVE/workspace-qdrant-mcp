//! Health check subcommand.
//!
//! Columnar template per cli-feedback.md.

use anyhow::Result;
use colored::Colorize;

use crate::grpc::client::workspace_daemon::ComponentHealth;
use crate::grpc::client::DaemonClient;
use crate::output::canvas;
use crate::output::columnar::ColumnarBuilder;
use crate::output::gutter::Gutter;
use crate::output::{self, ServiceStatus};

use super::types::{status_label, HealthComponentJson, HealthStatusJson};

/// Show system health, optionally as JSON.
pub async fn health(json: bool) -> Result<()> {
    match DaemonClient::connect_default().await {
        Ok(mut client) => match client.system().health(()).await {
            Ok(response) => {
                let health = response.into_inner();
                let overall = ServiceStatus::from_proto(health.status);

                if json {
                    print_health_json(true, overall, &health.components);
                } else {
                    canvas::print_title("System Health");
                    canvas::print_blank();
                    print_health_columnar(true, overall, &health.components);
                }
            }
            Err(e) => {
                if json {
                    print_health_json_disconnected(true);
                } else {
                    canvas::print_title("System Health");
                    canvas::print_blank();
                    ColumnarBuilder::new()
                        .kv("Connection", format_status(ServiceStatus::Healthy))
                        .kv("Health Check", format_status(ServiceStatus::Unknown))
                        .render();
                    output::warning(format!("Could not get health: {}", e));
                }
            }
        },
        Err(_) => {
            if json {
                print_health_json_disconnected(false);
            } else {
                canvas::print_title("System Health");
                canvas::print_blank();
                ColumnarBuilder::new()
                    .kv_gutter(
                        "Connection",
                        format_status(ServiceStatus::Unhealthy),
                        Gutter::Error,
                    )
                    .render();
                output::error("Daemon not running. Start with: wqm service start");
            }
        }
    }

    Ok(())
}

fn print_health_columnar(_connected: bool, overall: ServiceStatus, components: &[ComponentHealth]) {
    let overall_gutter = status_gutter(overall);

    let mut builder = ColumnarBuilder::new()
        .kv_gutter(
            "Connection",
            format_status(ServiceStatus::Healthy),
            Gutter::Sync,
        )
        .kv_gutter("Overall", format_status(overall), overall_gutter);

    if !components.is_empty() {
        builder = builder.section(Some("Components"));
        for comp in components {
            let comp_status = ServiceStatus::from_proto(comp.status);
            let gutter = status_gutter(comp_status);
            builder = builder.kv_gutter(&comp.component_name, format_status(comp_status), gutter);
            if !comp.message.is_empty() {
                builder = builder.raw(&format!("  {}", comp.message.dimmed()), Gutter::None);
            }
        }
    }

    builder.render();
}

fn format_status(status: ServiceStatus) -> String {
    match status {
        ServiceStatus::Healthy | ServiceStatus::Active => status_label(status).green().to_string(),
        ServiceStatus::Degraded => status_label(status).yellow().to_string(),
        ServiceStatus::Unhealthy => status_label(status).red().to_string(),
        ServiceStatus::Inactive => status_label(status).dimmed().to_string(),
        ServiceStatus::Unknown => status_label(status).dimmed().to_string(),
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

fn print_health_json(connected: bool, overall: ServiceStatus, components: &[ComponentHealth]) {
    let components: Vec<HealthComponentJson> = components
        .iter()
        .map(|c| HealthComponentJson {
            name: c.component_name.clone(),
            status: status_label(ServiceStatus::from_proto(c.status)).to_string(),
            message: if c.message.is_empty() {
                None
            } else {
                Some(c.message.clone())
            },
        })
        .collect();
    let json_out = HealthStatusJson {
        connected,
        health: status_label(overall).to_string(),
        components,
    };
    output::print_json(&json_out);
}

fn print_health_json_disconnected(connected: bool) {
    let json_out = HealthStatusJson {
        connected,
        health: if connected {
            "unknown".to_string()
        } else {
            "unhealthy".to_string()
        },
        components: Vec::new(),
    };
    output::print_json(&json_out);
}
