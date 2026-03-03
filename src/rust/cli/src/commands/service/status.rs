//! Service status subcommand

use anyhow::Result;
use serde::Serialize;

use crate::grpc::client::DaemonClient;
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
        ServiceStatus::Unknown => "unknown",
    }
}

/// Display or serialize the health response when connected to the daemon.
async fn handle_connected(
    mut client: crate::grpc::client::DaemonClient,
    json: bool,
) -> Result<()> {
    if !json {
        output::status_line("Connection", ServiceStatus::Healthy);
    }

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
                        message: if c.message.is_empty() { None } else { Some(c.message.clone()) },
                    })
                    .collect();
                output::print_json(&ServiceStatusJson {
                    connected: true,
                    health: status_name(overall).to_string(),
                    components,
                });
            } else {
                output::status_line("Health", overall);
                if !health.components.is_empty() {
                    output::separator();
                    for comp in health.components {
                        let comp_status = ServiceStatus::from_proto(comp.status);
                        output::status_line(&comp.component_name, comp_status);
                        if !comp.message.is_empty() {
                            output::kv("  Message", &comp.message);
                        }
                    }
                }
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
                output::status_line("Health", ServiceStatus::Unknown);
                output::warning(format!("Could not get health: {}", e));
            }
        }
    }
    Ok(())
}

/// Show daemon status, optionally as JSON
pub async fn execute(json: bool) -> Result<()> {
    if !json {
        output::section("Daemon Status");
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
                output::status_line("Connection", ServiceStatus::Unhealthy);
                output::error("Daemon not running or not reachable");
                output::info("Start with: wqm service start");
            }
        }
    }

    Ok(())
}
