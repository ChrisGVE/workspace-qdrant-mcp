//! Service uninstall subcommand

use anyhow::{Context, Result};
use std::process::Command;

use crate::output;

use super::platform::{
    get_service_manager, install_dir, is_daemon_running, log_directory, ServiceManager,
    DAEMON_BINARY, SERVICE_NAME,
};

/// Uninstall the daemon system service
pub async fn execute(remove_data: bool) -> Result<()> {
    output::section("Uninstalling daemon service");

    // 1. Stop service if running
    if is_daemon_running().await {
        output::info("Stopping running daemon...");
        super::stop::execute().await?;
    }

    // 2. Remove platform-specific service config
    match get_service_manager() {
        ServiceManager::Launchctl => uninstall_launchctl()?,
        ServiceManager::Systemd => uninstall_systemd()?,
        ServiceManager::WindowsService => {
            output::warning("Windows service uninstallation is not yet supported");
        }
        ServiceManager::Unknown => {
            output::info("No platform service config to remove");
        }
    }

    // 3. Remove binary
    let bin_path = install_dir()?.join(DAEMON_BINARY);
    if bin_path.exists() {
        std::fs::remove_file(&bin_path)
            .with_context(|| format!("Failed to remove binary: {}", bin_path.display()))?;
        output::success(format!("Removed binary: {}", bin_path.display()));
    }

    // 4. Optionally remove data
    if remove_data {
        remove_data_directories()?;
    } else {
        output::info("Data directory preserved (use --remove-data to delete)");
    }

    output::success("Daemon service uninstalled");
    Ok(())
}

fn uninstall_launchctl() -> Result<()> {
    let plist_path = dirs::home_dir()
        .context("Could not find home directory")?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", SERVICE_NAME));

    if plist_path.exists() {
        // Unload first (ignore errors - may already be unloaded)
        let _ = Command::new("launchctl")
            .args(["unload"])
            .arg(&plist_path)
            .status();

        std::fs::remove_file(&plist_path)
            .with_context(|| format!("Failed to remove plist: {}", plist_path.display()))?;
        output::success(format!("Removed service config: {}", plist_path.display()));
    } else {
        output::info("Service config already removed");
    }

    Ok(())
}

fn uninstall_systemd() -> Result<()> {
    // Disable the service
    let _ = Command::new("systemctl")
        .args(["--user", "disable", "--now", "memexd"])
        .status();

    let unit_path = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".config/systemd/user/memexd.service");

    if unit_path.exists() {
        std::fs::remove_file(&unit_path)
            .with_context(|| format!("Failed to remove unit file: {}", unit_path.display()))?;
        output::success(format!("Removed service config: {}", unit_path.display()));

        let _ = Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status();
    } else {
        output::info("Service config already removed");
    }

    Ok(())
}

fn remove_data_directories() -> Result<()> {
    let data_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".workspace-qdrant");
    if data_dir.exists() {
        std::fs::remove_dir_all(&data_dir)
            .with_context(|| format!("Failed to remove data: {}", data_dir.display()))?;
        output::success(format!("Removed data directory: {}", data_dir.display()));
    }

    let log_dir = log_directory()?;
    if log_dir.exists() {
        std::fs::remove_dir_all(&log_dir)
            .with_context(|| format!("Failed to remove logs: {}", log_dir.display()))?;
        output::success(format!("Removed log directory: {}", log_dir.display()));
    }

    Ok(())
}
