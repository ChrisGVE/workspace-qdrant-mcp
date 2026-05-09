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
        ServiceManager::WindowsService => install_windows_service(&dest).await,
        ServiceManager::Unknown => {
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

/// Register `memexd.exe` with the Windows Service Control Manager.
///
/// `dest` is the absolute path to the freshly-copied daemon binary;
/// the SCM stores this as the `BinaryPathName` so `sc.exe start
/// memexd` can locate it later.
#[cfg(windows)]
async fn install_windows_service(dest: &std::path::Path) -> Result<()> {
    use windows_service::service::{
        ServiceAccess, ServiceErrorControl, ServiceInfo, ServiceStartType, ServiceType,
    };
    use windows_service::service_manager::{ServiceManager as WinScm, ServiceManagerAccess};

    let manager_access = ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE;
    let scm = WinScm::local_computer(None::<&str>, manager_access)
        .context("Failed to connect to the Windows Service Control Manager")?;

    let service_info = ServiceInfo {
        name: SERVICE_NAME.into(),
        display_name: "Workspace Qdrant MCP Daemon".into(),
        service_type: ServiceType::OWN_PROCESS,
        start_type: ServiceStartType::AutoStart,
        error_control: ServiceErrorControl::Normal,
        executable_path: dest.to_path_buf(),
        launch_arguments: Vec::new(),
        dependencies: Vec::new(),
        // Default to LocalSystem so the daemon can write to its
        // per-machine data directory. Operators who want a more
        // restricted account can re-create the service manually with
        // `sc.exe config memexd obj= ".\\<account>" password= <pwd>`.
        account_name: None,
        account_password: None,
    };

    let service_access = ServiceAccess::CHANGE_CONFIG | ServiceAccess::START;
    let service = scm
        .create_service(&service_info, service_access)
        .context("Failed to register memexd with the Windows SCM")?;

    output::kv("Service registered", SERVICE_NAME);

    if let Err(e) = service.start::<&str>(&[]) {
        output::warning(format!("Service registered but failed to start: {}", e));
        output::info("Try: sc.exe start memexd");
        return Ok(());
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    if is_daemon_running().await {
        output::success("Daemon installed and running");
    } else {
        output::warning("Service started but daemon not responding yet");
        output::info("Check status with: wqm service status");
    }

    Ok(())
}

#[cfg(not(windows))]
async fn install_windows_service(_dest: &std::path::Path) -> Result<()> {
    // unreachable: get_service_manager() never returns WindowsService
    // outside of cfg(windows). Kept as a stub so the dispatch table
    // typechecks cross-platform.
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
