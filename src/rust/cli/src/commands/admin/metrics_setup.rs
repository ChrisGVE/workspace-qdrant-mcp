//! `wqm admin metrics enable/disable/status` — manage the Prometheus metrics endpoint
//!
//! Modifies the daemon's launchd plist (macOS) or systemd unit (Linux) to
//! add/remove the `--metrics-port` argument, then restarts the daemon.

use anyhow::{Context, Result};
use std::process::Command;

use crate::commands::service::platform::{
    generate_launchd_plist_with_options, get_service_manager, launchd_plist_path,
    parse_binary_from_plist, parse_metrics_port_from_plist, ServiceManager,
};
use crate::output;

const DEFAULT_METRICS_PORT: u16 = 9090;

/// Enable the metrics endpoint by adding --metrics-port to daemon launch args.
pub async fn enable(port: Option<u16>) -> Result<()> {
    let port = port.unwrap_or(DEFAULT_METRICS_PORT);

    match get_service_manager() {
        ServiceManager::Launchctl => enable_launchctl(port).await,
        ServiceManager::Systemd => {
            output::warning("Systemd metrics setup not yet implemented");
            output::info(format!(
                "Manually add '--metrics-port {}' to the ExecStart line in the systemd unit",
                port
            ));
            Ok(())
        }
        _ => {
            output::warning("Service management not supported on this platform");
            Ok(())
        }
    }
}

/// Disable the metrics endpoint by removing --metrics-port from daemon launch args.
pub async fn disable() -> Result<()> {
    match get_service_manager() {
        ServiceManager::Launchctl => disable_launchctl().await,
        ServiceManager::Systemd => {
            output::warning("Systemd metrics setup not yet implemented");
            output::info(
                "Manually remove '--metrics-port' from the ExecStart line in the systemd unit",
            );
            Ok(())
        }
        _ => {
            output::warning("Service management not supported on this platform");
            Ok(())
        }
    }
}

/// Check if the metrics endpoint is configured and responding.
pub async fn status(port: u16) -> Result<()> {
    output::section("Metrics Endpoint Status");

    let configured_port = parse_metrics_port_from_plist();

    match configured_port {
        Some(p) => {
            output::kv("Configured", format!("enabled (port {})", p));
        }
        None => {
            output::kv("Configured", "disabled");
            output::info(format!(
                "Enable with: wqm admin metrics enable [--port {}]",
                DEFAULT_METRICS_PORT
            ));
            return Ok(());
        }
    }

    let check_port = configured_port.unwrap_or(port);
    let url = format!("http://127.0.0.1:{}/metrics", check_port);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .context("Failed to build HTTP client")?;

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            output::kv("Responding", format!("yes ({})", url));
        }
        Ok(resp) => {
            output::kv("Responding", format!("error (HTTP {})", resp.status()));
        }
        Err(_) => {
            output::kv("Responding", "no (connection refused)");
            output::info("The daemon may need a restart: wqm service restart");
        }
    }

    Ok(())
}

async fn enable_launchctl(port: u16) -> Result<()> {
    let plist_path = launchd_plist_path()?;

    if !plist_path.exists() {
        anyhow::bail!(
            "Daemon plist not found at {}.\n\
             Install the daemon first with: wqm service install",
            plist_path.display()
        );
    }

    let binary = parse_binary_from_plist().context(
        "Could not determine daemon binary path from plist.\n\
         Reinstall the daemon with: wqm service install",
    )?;

    if let Some(existing) = parse_metrics_port_from_plist() {
        if existing == port {
            output::info(format!(
                "Metrics endpoint already enabled on port {}",
                port
            ));
            return Ok(());
        }
        output::info(format!(
            "Changing metrics port from {} to {}",
            existing, port
        ));
    }

    // Regenerate plist with metrics port
    let content = generate_launchd_plist_with_options(&binary, Some(port));
    std::fs::write(&plist_path, &content)
        .with_context(|| format!("Failed to write plist: {}", plist_path.display()))?;

    // Restart daemon
    restart_launchctl(&plist_path)?;

    output::success(format!(
        "Metrics endpoint enabled on port {} — daemon restarting",
        port
    ));
    output::info(format!("View metrics: wqm admin metrics show --port {}", port));

    Ok(())
}

async fn disable_launchctl() -> Result<()> {
    let plist_path = launchd_plist_path()?;

    if !plist_path.exists() {
        anyhow::bail!("Daemon plist not found at {}", plist_path.display());
    }

    if parse_metrics_port_from_plist().is_none() {
        output::info("Metrics endpoint is already disabled");
        return Ok(());
    }

    let binary = parse_binary_from_plist().context("Could not determine daemon binary path")?;

    // Regenerate plist without metrics port
    let content = generate_launchd_plist_with_options(&binary, None);
    std::fs::write(&plist_path, &content)
        .with_context(|| format!("Failed to write plist: {}", plist_path.display()))?;

    restart_launchctl(&plist_path)?;

    output::success("Metrics endpoint disabled — daemon restarting");

    Ok(())
}

fn restart_launchctl(plist_path: &std::path::Path) -> Result<()> {
    let _ = Command::new("launchctl")
        .args(["unload", "-w"])
        .arg(plist_path)
        .status();

    let status = Command::new("launchctl")
        .args(["load", "-w"])
        .arg(plist_path)
        .status()
        .context("Failed to reload launchd service")?;

    if !status.success() {
        output::warning("Daemon may need manual restart");
        output::info(format!(
            "launchctl load -w {}",
            plist_path.display()
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_metrics_port() {
        assert_eq!(DEFAULT_METRICS_PORT, 9090);
    }
}
