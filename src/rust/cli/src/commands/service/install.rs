//! Service install subcommand

use anyhow::{Context, Result};
use std::process::Command;

use crate::output;

use super::platform::{
    find_daemon_binary, generate_launchd_plist, generate_systemd_unit, get_service_manager,
    install_dir, is_daemon_running, log_directory, ServiceManager, DAEMON_BINARY, SERVICE_NAME,
};

/// Install the daemon as a system service
pub async fn execute(binary: Option<String>) -> Result<()> {
    output::section("Installing daemon service");

    // 1. Find the binary
    let source = find_daemon_binary(binary)?;
    output::kv("Source binary", source.display());

    // 2. Copy binary to install dir
    let dest_dir = install_dir()?;
    std::fs::create_dir_all(&dest_dir)
        .with_context(|| format!("Failed to create install directory: {}", dest_dir.display()))?;

    let dest = dest_dir.join(DAEMON_BINARY);

    // Only copy if source != destination
    if source != dest {
        std::fs::copy(&source, &dest)
            .with_context(|| format!("Failed to copy binary to {}", dest.display()))?;

        // Ensure binary is executable on unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&dest)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&dest, perms)?;
        }
        output::kv("Installed binary", dest.display());
    } else {
        output::info("Binary already at install location");
    }

    // 3. Create log directory
    let log_dir = log_directory()?;
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("Failed to create log directory: {}", log_dir.display()))?;

    // 4. Create platform-specific service config
    match get_service_manager() {
        ServiceManager::Launchctl => install_launchctl(&dest).await,
        ServiceManager::Systemd => install_systemd(&dest).await,
        ServiceManager::WindowsService | ServiceManager::Unknown => {
            output::warning("Service management not supported on this platform");
            output::info(format!("Binary installed at: {}", dest.display()));
            output::info("To run manually: memexd --foreground");
            Ok(())
        }
    }
}

async fn install_launchctl(dest: &std::path::Path) -> Result<()> {
    let plist_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join("Library/LaunchAgents");
    std::fs::create_dir_all(&plist_dir)?;

    let plist_path = plist_dir.join(format!("{}.plist", SERVICE_NAME));
    let content = generate_launchd_plist(dest);
    std::fs::write(&plist_path, &content)
        .with_context(|| format!("Failed to write plist: {}", plist_path.display()))?;
    output::kv("Service config", plist_path.display());

    // Load the service
    let load_status = Command::new("launchctl")
        .args(["load", "-w"])
        .arg(&plist_path)
        .status()?;

    if load_status.success() {
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        if is_daemon_running().await {
            output::success("Daemon installed and running");
        } else {
            output::warning("Service loaded but daemon not responding yet");
            output::info("Check status with: wqm service status");
        }
    } else {
        output::warning("Service config installed but failed to load");
        output::info("Try: launchctl load -w <plist_path>");
    }

    Ok(())
}

async fn install_systemd(dest: &std::path::Path) -> Result<()> {
    let unit_dir = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".config/systemd/user");
    std::fs::create_dir_all(&unit_dir)?;

    let unit_path = unit_dir.join("memexd.service");
    let content = generate_systemd_unit(dest);
    std::fs::write(&unit_path, &content)
        .with_context(|| format!("Failed to write unit file: {}", unit_path.display()))?;
    output::kv("Service config", unit_path.display());

    // Reload systemd and enable
    let _ = Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .status();

    let enable_status = Command::new("systemctl")
        .args(["--user", "enable", "--now", "memexd"])
        .status()?;

    if enable_status.success() {
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        if is_daemon_running().await {
            output::success("Daemon installed and running");
        } else {
            output::warning("Service enabled but daemon not responding yet");
            output::info("Check status with: wqm service status");
        }
    } else {
        output::warning("Service config installed but failed to enable");
        output::info("Try: systemctl --user enable --now memexd");
    }

    Ok(())
}
