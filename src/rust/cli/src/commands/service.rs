//! Service command - daemon lifecycle management
//!
//! Manages the memexd daemon runtime.
//! Subcommands: start, stop, restart, status, install, uninstall, logs

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use std::path::PathBuf;
use std::process::Command;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Service command arguments
#[derive(Args)]
pub struct ServiceArgs {
    #[command(subcommand)]
    command: ServiceCommand,
}

/// Service subcommands
#[derive(Subcommand)]
enum ServiceCommand {
    /// Start the daemon
    Start,

    /// Stop the daemon
    Stop,

    /// Restart the daemon
    Restart,

    /// Show daemon status
    Status,

    /// Install the daemon as a system service
    Install {
        /// Path to memexd binary (auto-detected if not specified)
        #[arg(long)]
        binary: Option<String>,
    },

    /// Uninstall the daemon system service
    Uninstall {
        /// Also remove data directory (~/.workspace-qdrant/)
        #[arg(long)]
        remove_data: bool,
    },

    /// View daemon logs (convenience shortcut for `wqm debug logs`)
    Logs {
        /// Number of lines to show (default: 50)
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,

        /// Follow log output (like tail -f)
        #[arg(short, long)]
        follow: bool,

        /// Show only ERROR and WARN level entries
        #[arg(short, long)]
        errors_only: bool,
    },
}

/// Execute service command
pub async fn execute(args: ServiceArgs) -> Result<()> {
    match args.command {
        ServiceCommand::Start => start().await,
        ServiceCommand::Stop => stop().await,
        ServiceCommand::Restart => restart().await,
        ServiceCommand::Status => status().await,
        ServiceCommand::Install { binary } => install(binary).await,
        ServiceCommand::Uninstall { remove_data } => uninstall(remove_data).await,
        ServiceCommand::Logs { lines, follow, errors_only } => logs(lines, follow, errors_only).await,
    }
}

/// Check if daemon is running by attempting gRPC connection
async fn is_daemon_running() -> bool {
    DaemonClient::connect_default().await.is_ok()
}

/// Get platform-specific service manager
fn get_service_manager() -> ServiceManager {
    #[cfg(target_os = "macos")]
    {
        ServiceManager::Launchctl
    }
    #[cfg(target_os = "linux")]
    {
        ServiceManager::Systemd
    }
    #[cfg(target_os = "windows")]
    {
        ServiceManager::WindowsService
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        ServiceManager::Unknown
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Variants used based on target platform
enum ServiceManager {
    Launchctl,
    Systemd,
    WindowsService,
    Unknown,
}

const SERVICE_NAME: &str = "com.workspace-qdrant.memexd";
const DAEMON_BINARY: &str = "memexd";

async fn start() -> Result<()> {
    output::info("Starting daemon...");

    if is_daemon_running().await {
        output::info("Daemon is already running");
        return Ok(());
    }

    match get_service_manager() {
        ServiceManager::Launchctl => {
            let plist_path = dirs::home_dir()
                .context("Could not find home directory")?
                .join("Library/LaunchAgents")
                .join(format!("{}.plist", SERVICE_NAME));

            if plist_path.exists() {
                let status = Command::new("launchctl")
                    .args(["load", "-w"])
                    .arg(&plist_path)
                    .status()?;

                if status.success() {
                    // Wait a moment for daemon to start
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    if is_daemon_running().await {
                        output::success("Daemon started");
                    } else {
                        output::warning("Service loaded but daemon not responding");
                    }
                } else {
                    output::error("Failed to start daemon");
                }
            } else {
                output::error("Service not installed. Run: wqm service install");
            }
        }
        ServiceManager::Systemd => {
            let status = Command::new("systemctl")
                .args(["--user", "start", "memexd"])
                .status()?;

            if status.success() {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                if is_daemon_running().await {
                    output::success("Daemon started");
                } else {
                    output::warning("Service started but daemon not responding");
                }
            } else {
                output::error("Failed to start daemon");
            }
        }
        ServiceManager::WindowsService => {
            #[cfg(windows)]
            {
                let status = Command::new("sc.exe")
                    .args(["start", "memexd"])
                    .status()?;

                if status.success() {
                    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                    if is_daemon_running().await {
                        output::success("Daemon started");
                    } else {
                        output::warning("Service started but daemon not responding yet");
                        output::info("Try: wqm service status");
                    }
                } else {
                    output::error("Failed to start daemon");
                    output::info("Check if service is installed: wqm service status");
                }
            }
            #[cfg(not(windows))]
            {
                output::error("Windows service start only available on Windows");
            }
        }
        _ => {
            output::error("Service management not supported on this platform");
        }
    }

    Ok(())
}

async fn stop() -> Result<()> {
    output::info("Stopping daemon...");

    match get_service_manager() {
        ServiceManager::Launchctl => {
            let plist_path = dirs::home_dir()
                .context("Could not find home directory")?
                .join("Library/LaunchAgents")
                .join(format!("{}.plist", SERVICE_NAME));

            if plist_path.exists() {
                let _ = Command::new("launchctl")
                    .args(["unload"])
                    .arg(&plist_path)
                    .status();
            }

            // Also try killing by name
            let _ = Command::new("pkill").args(["-f", DAEMON_BINARY]).status();

            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

            if !is_daemon_running().await {
                output::success("Daemon stopped");
            } else {
                output::warning("Daemon may still be running");
            }
        }
        ServiceManager::Systemd => {
            let status = Command::new("systemctl")
                .args(["--user", "stop", "memexd"])
                .status()?;

            if status.success() {
                output::success("Daemon stopped");
            } else {
                output::warning("Failed to stop daemon via systemd");
            }
        }
        ServiceManager::WindowsService => {
            #[cfg(windows)]
            {
                let status = Command::new("sc.exe")
                    .args(["stop", "memexd"])
                    .status()?;

                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                if !is_daemon_running().await {
                    output::success("Daemon stopped");
                } else {
                    // Try force kill as fallback
                    let _ = Command::new("taskkill")
                        .args(["/F", "/IM", "memexd.exe"])
                        .status();

                    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

                    if !is_daemon_running().await {
                        output::success("Daemon stopped (force kill)");
                    } else {
                        output::warning("Daemon may still be running");
                    }
                }
            }
            #[cfg(not(windows))]
            {
                output::error("Windows service stop only available on Windows");
            }
        }
        _ => {
            // Fallback: try to kill by process name
            #[cfg(unix)]
            let _ = Command::new("pkill").args(["-f", DAEMON_BINARY]).status();
            #[cfg(windows)]
            let _ = Command::new("taskkill").args(["/F", "/IM", "memexd.exe"]).status();
            output::info("Attempted to stop daemon");
        }
    }

    Ok(())
}

async fn restart() -> Result<()> {
    output::info("Restarting daemon...");
    stop().await?;
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    start().await?;
    Ok(())
}

async fn status() -> Result<()> {
    output::section("Daemon Status");

    // Try to connect and get health
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Connection", ServiceStatus::Healthy);

            // Get health check
            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    let status = ServiceStatus::from_proto(health.status);
                    output::status_line("Health", status);

                    // Show component health
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
                Err(e) => {
                    output::status_line("Health", ServiceStatus::Unknown);
                    output::warning(format!("Could not get health: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Connection", ServiceStatus::Unhealthy);
            output::error("Daemon not running or not reachable");
            output::info("Start with: wqm service start");
        }
    }

    Ok(())
}

// =========================================================================
// Install / Uninstall / Logs
// =========================================================================

/// Resolve the path to the memexd binary
fn find_daemon_binary(explicit: Option<String>) -> Result<PathBuf> {
    // 1. Explicit path from --binary flag
    if let Some(path) = explicit {
        let p = PathBuf::from(&path);
        if p.is_file() {
            return Ok(p);
        }
        bail!("Specified binary not found: {}", path);
    }

    // 2. Check well-known install locations
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let candidates = [
        home.join(".local/bin/memexd"),
        home.join(".cargo/bin/memexd"),
    ];
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }

    // 3. Fall back to `which`
    if let Ok(output) = Command::new("which").arg(DAEMON_BINARY).output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let p = PathBuf::from(&path_str);
            if p.is_file() {
                return Ok(p);
            }
        }
    }

    bail!(
        "Could not find '{}' binary. Build it first or specify --binary <path>",
        DAEMON_BINARY
    );
}

/// Get the canonical install directory for the daemon binary
fn install_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".local/bin"))
}

/// Get the canonical log directory for the daemon
fn log_directory() -> Result<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        Ok(home.join("Library/Logs/workspace-qdrant"))
    }
    #[cfg(target_os = "linux")]
    {
        let base = std::env::var("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir().unwrap_or_default().join(".local/state")
            });
        Ok(base.join("workspace-qdrant/logs"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let home = dirs::home_dir().context("Could not determine home directory")?;
        Ok(home.join(".workspace-qdrant/logs"))
    }
}

/// Generate macOS launchd plist content
fn generate_launchd_plist(binary_path: &std::path::Path) -> String {
    let home = dirs::home_dir().unwrap_or_default();
    let log_dir = home.join("Library/Logs/workspace-qdrant");
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{service}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/daemon.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/daemon.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>RUST_LOG</key>
        <string>info</string>
        <key>HOME</key>
        <string>{home}</string>
    </dict>
</dict>
</plist>
"#,
        service = SERVICE_NAME,
        binary = binary_path.display(),
        log_dir = log_dir.display(),
        home = home.display(),
    )
}

/// Generate Linux systemd user unit content
fn generate_systemd_unit(binary_path: &std::path::Path) -> String {
    format!(
        r#"[Unit]
Description=Workspace Qdrant MCP Daemon
After=network.target

[Service]
Type=simple
ExecStart={binary}
Restart=on-failure
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=default.target
"#,
        binary = binary_path.display(),
    )
}

async fn install(binary: Option<String>) -> Result<()> {
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
        ServiceManager::Launchctl => {
            let plist_dir = dirs::home_dir()
                .context("Could not find home directory")?
                .join("Library/LaunchAgents");
            std::fs::create_dir_all(&plist_dir)?;

            let plist_path = plist_dir.join(format!("{}.plist", SERVICE_NAME));
            let content = generate_launchd_plist(&dest);
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
        }
        ServiceManager::Systemd => {
            let unit_dir = dirs::home_dir()
                .context("Could not find home directory")?
                .join(".config/systemd/user");
            std::fs::create_dir_all(&unit_dir)?;

            let unit_path = unit_dir.join("memexd.service");
            let content = generate_systemd_unit(&dest);
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
        }
        ServiceManager::WindowsService => {
            output::warning("Windows service installation is not yet supported");
            output::info(format!("Binary installed at: {}", dest.display()));
            output::info("To run manually: memexd --foreground");
        }
        ServiceManager::Unknown => {
            output::warning("Service management not supported on this platform");
            output::info(format!("Binary installed at: {}", dest.display()));
            output::info("To run manually: memexd --foreground");
        }
    }

    Ok(())
}

async fn uninstall(remove_data: bool) -> Result<()> {
    output::section("Uninstalling daemon service");

    // 1. Stop service if running
    if is_daemon_running().await {
        output::info("Stopping running daemon...");
        stop().await?;
    }

    // 2. Remove platform-specific service config
    match get_service_manager() {
        ServiceManager::Launchctl => {
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
        }
        ServiceManager::Systemd => {
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
        }
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
    } else {
        output::info("Data directory preserved (use --remove-data to delete)");
    }

    output::success("Daemon service uninstalled");
    Ok(())
}

async fn logs(lines: usize, follow: bool, errors_only: bool) -> Result<()> {
    let log_dir = log_directory()?;
    let log_file = log_dir.join("daemon.jsonl");

    // Fall back to daemon.log if .jsonl doesn't exist
    let log_file = if log_file.exists() {
        log_file
    } else {
        let alt = log_dir.join("daemon.log");
        if alt.exists() {
            alt
        } else {
            // Check launchd stdout/stderr log locations as last resort
            #[cfg(target_os = "macos")]
            {
                let home = dirs::home_dir().context("Could not determine home directory")?;
                let launchd_log = home.join("Library/Logs/workspace-qdrant/daemon.out.log");
                if launchd_log.exists() {
                    launchd_log
                } else {
                    bail!(
                        "No log files found in {}\nHint: use 'wqm debug logs' for comprehensive log discovery",
                        log_dir.display()
                    );
                }
            }
            #[cfg(not(target_os = "macos"))]
            bail!(
                "No log files found in {}\nHint: use 'wqm debug logs' for comprehensive log discovery",
                log_dir.display()
            );
        }
    };

    output::info(format!("Log file: {}", log_file.display()));

    if follow {
        // Use tail -f for follow mode
        let mut args = vec!["-f".to_string()];
        if errors_only {
            // Pipe through grep for errors-only in follow mode
            let mut child = Command::new("tail")
                .args(["-f", log_file.to_str().unwrap_or("")])
                .stdout(std::process::Stdio::piped())
                .spawn()
                .context("Failed to start tail command")?;

            let stdout = child.stdout.take().context("Failed to capture tail output")?;
            let reader = std::io::BufReader::new(stdout);
            use std::io::BufRead;
            for line in reader.lines() {
                let line = line?;
                let upper = line.to_uppercase();
                if upper.contains("ERROR") || upper.contains("WARN") {
                    println!("{}", line);
                }
            }
            child.wait()?;
        } else {
            args.push("-n".to_string());
            args.push(lines.to_string());
            let status = Command::new("tail")
                .args(&args)
                .arg(&log_file)
                .status()
                .context("Failed to run tail command")?;
            if !status.success() {
                bail!("tail command failed");
            }
        }
    } else {
        // Read last N lines
        let content = std::fs::read_to_string(&log_file)
            .with_context(|| format!("Failed to read log file: {}", log_file.display()))?;

        let all_lines: Vec<&str> = content.lines().collect();
        let start = all_lines.len().saturating_sub(lines);
        let tail = &all_lines[start..];

        if errors_only {
            for line in tail {
                let upper = line.to_uppercase();
                if upper.contains("ERROR") || upper.contains("WARN") {
                    println!("{}", line);
                }
            }
        } else {
            for line in tail {
                println!("{}", line);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_daemon_binary_explicit_missing() {
        let result = find_daemon_binary(Some("/nonexistent/path/memexd".to_string()));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_install_dir() {
        let dir = install_dir().unwrap();
        assert!(dir.ends_with(".local/bin"));
    }

    #[test]
    fn test_log_directory() {
        let dir = log_directory().unwrap();
        // Platform-specific check
        #[cfg(target_os = "macos")]
        assert!(dir.to_str().unwrap().contains("Library/Logs/workspace-qdrant"));
        #[cfg(target_os = "linux")]
        assert!(dir.to_str().unwrap().contains("workspace-qdrant/logs"));
    }

    #[test]
    fn test_generate_launchd_plist() {
        let binary = PathBuf::from("/Users/test/.local/bin/memexd");
        let plist = generate_launchd_plist(&binary);
        assert!(plist.contains(SERVICE_NAME));
        assert!(plist.contains("/Users/test/.local/bin/memexd"));
        assert!(plist.contains("<key>RunAtLoad</key>"));
        assert!(plist.contains("<key>KeepAlive</key>"));
        assert!(plist.contains("RUST_LOG"));
    }

    #[test]
    fn test_generate_systemd_unit() {
        let binary = PathBuf::from("/home/test/.local/bin/memexd");
        let unit = generate_systemd_unit(&binary);
        assert!(unit.contains("ExecStart=/home/test/.local/bin/memexd"));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("RUST_LOG=info"));
        assert!(unit.contains("[Install]"));
    }

    #[test]
    fn test_get_service_manager() {
        let manager = get_service_manager();
        #[cfg(target_os = "macos")]
        assert!(matches!(manager, ServiceManager::Launchctl));
        #[cfg(target_os = "linux")]
        assert!(matches!(manager, ServiceManager::Systemd));
    }
}
