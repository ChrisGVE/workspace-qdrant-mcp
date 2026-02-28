//! Health check subcommand.

use anyhow::Result;

use crate::grpc::client::workspace_daemon::ComponentHealth;
use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

use super::types::{status_label, HealthComponentJson, HealthStatusJson};

/// Show system health, optionally as JSON.
pub async fn health(json: bool) -> Result<()> {
    if !json {
        output::section("System Health");
    }

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            if !json {
                output::status_line("Daemon Connection", ServiceStatus::Healthy);
            }

            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    let overall = ServiceStatus::from_proto(health.status);

                    if json {
                        print_health_json(true, overall, &health.components);
                    } else {
                        print_health_text(overall, &health.components);
                    }
                }
                Err(e) => {
                    if json {
                        print_health_json_disconnected(true);
                    } else {
                        output::status_line("Health Check", ServiceStatus::Unknown);
                        output::warning(format!("Could not get health: {}", e));
                    }
                }
            }
        }
        Err(_) => {
            if json {
                print_health_json_disconnected(false);
            } else {
                output::status_line("Daemon Connection", ServiceStatus::Unhealthy);
                output::error("Daemon not running");
                output::info("Start with: wqm service start");
            }
        }
    }

    Ok(())
}

fn print_health_json(
    connected: bool,
    overall: ServiceStatus,
    components: &[ComponentHealth],
) {
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

fn print_health_text(overall: ServiceStatus, components: &[ComponentHealth]) {
    output::status_line("Overall Health", overall);

    if !components.is_empty() {
        output::separator();
        for comp in components {
            let comp_status = ServiceStatus::from_proto(comp.status);
            output::status_line(&comp.component_name, comp_status);
            if !comp.message.is_empty() {
                output::kv("  Message", &comp.message);
            }
        }
    }
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
