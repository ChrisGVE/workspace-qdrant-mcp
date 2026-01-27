//! Service command - daemon lifecycle management
//!
//! Phase 1 HIGH priority command for managing memexd daemon.
//! Subcommands: install, uninstall, start, stop, restart, status, logs

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
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
    /// Install the daemon as a user service
    Install,

    /// Uninstall the daemon user service
    Uninstall,

    /// Start the daemon
    Start,

    /// Stop the daemon
    Stop,

    /// Restart the daemon
    Restart,

    /// Show daemon status
    Status,

    /// Show daemon logs
    Logs {
        /// Number of lines to show
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,

        /// Follow log output (like tail -f)
        #[arg(short, long)]
        follow: bool,
    },
}

/// Execute service command
pub async fn execute(args: ServiceArgs) -> Result<()> {
    match args.command {
        ServiceCommand::Install => install().await,
        ServiceCommand::Uninstall => uninstall().await,
        ServiceCommand::Start => start().await,
        ServiceCommand::Stop => stop().await,
        ServiceCommand::Restart => restart().await,
        ServiceCommand::Status => status().await,
        ServiceCommand::Logs { lines, follow } => logs(lines, follow).await,
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

/// Find ONNX Runtime library path on the system
fn find_onnx_runtime_path() -> String {
    // Check environment variable first
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        return path;
    }

    // Common locations for ONNX Runtime on macOS
    let candidates = [
        "/usr/local/lib/libonnxruntime.dylib",           // Homebrew Intel
        "/opt/homebrew/lib/libonnxruntime.dylib",        // Homebrew Apple Silicon
        "/usr/lib/libonnxruntime.dylib",                 // System
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }

    // Fallback - user will need to set ORT_DYLIB_PATH
    "/usr/local/lib/libonnxruntime.dylib".to_string()
}

async fn install() -> Result<()> {
    output::info("Installing daemon service...");

    match get_service_manager() {
        ServiceManager::Launchctl => install_launchctl().await,
        ServiceManager::Systemd => install_systemd().await,
        _ => {
            output::error("Service installation not supported on this platform");
            Ok(())
        }
    }
}

async fn install_launchctl() -> Result<()> {
    let plist_path = dirs::home_dir()
        .context("Could not find home directory")?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", SERVICE_NAME));

    // Find daemon binary
    let daemon_path = which::which(DAEMON_BINARY)
        .unwrap_or_else(|_| std::path::PathBuf::from("/usr/local/bin/memexd"));

    // Find ONNX Runtime library path
    let ort_dylib_path = find_onnx_runtime_path();

    let plist_content = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{daemon}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ORT_DYLIB_PATH</key>
        <string>{ort_path}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/memexd.out.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/memexd.err.log</string>
</dict>
</plist>"#,
        label = SERVICE_NAME,
        daemon = daemon_path.display(),
        ort_path = ort_dylib_path
    );

    // Create LaunchAgents directory if needed
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&plist_path, plist_content)?;
    output::success(format!("Created {}", plist_path.display()));

    // Unload existing service first (ignore errors if not loaded)
    let _ = Command::new("launchctl")
        .args(["unload"])
        .arg(&plist_path)
        .status();

    // Load the service
    let status = Command::new("launchctl")
        .args(["load", "-w"])
        .arg(&plist_path)
        .status()?;

    if status.success() {
        output::success("Daemon service installed and started");
    } else {
        output::warning("Service file created but failed to load. Try: launchctl load -w <path>");
    }

    Ok(())
}

async fn install_systemd() -> Result<()> {
    let service_path = dirs::home_dir()
        .context("Could not find home directory")?
        .join(".config/systemd/user")
        .join("memexd.service");

    let daemon_path = which::which(DAEMON_BINARY)
        .unwrap_or_else(|_| std::path::PathBuf::from("/usr/local/bin/memexd"));

    let service_content = format!(
        r#"[Unit]
Description=Workspace Qdrant MCP Daemon
After=network.target

[Service]
Type=simple
ExecStart={}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"#,
        daemon_path.display()
    );

    // Create systemd user directory if needed
    if let Some(parent) = service_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&service_path, service_content)?;
    output::success(format!("Created {}", service_path.display()));

    // Reload systemd and enable
    let _ = Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .status();

    let status = Command::new("systemctl")
        .args(["--user", "enable", "memexd"])
        .status()?;

    if status.success() {
        output::success("Daemon service installed and enabled");
        output::info("Start with: wqm service start");
    } else {
        output::warning("Service file created but failed to enable");
    }

    Ok(())
}

async fn uninstall() -> Result<()> {
    output::info("Uninstalling daemon service...");

    // Stop first
    let _ = stop().await;

    match get_service_manager() {
        ServiceManager::Launchctl => {
            let plist_path = dirs::home_dir()
                .context("Could not find home directory")?
                .join("Library/LaunchAgents")
                .join(format!("{}.plist", SERVICE_NAME));

            if plist_path.exists() {
                let _ = Command::new("launchctl")
                    .args(["unload", "-w"])
                    .arg(&plist_path)
                    .status();

                std::fs::remove_file(&plist_path)?;
                output::success("Daemon service uninstalled");
            } else {
                output::info("Service not installed");
            }
        }
        ServiceManager::Systemd => {
            let _ = Command::new("systemctl")
                .args(["--user", "disable", "memexd"])
                .status();

            let service_path = dirs::home_dir()
                .context("Could not find home directory")?
                .join(".config/systemd/user/memexd.service");

            if service_path.exists() {
                std::fs::remove_file(&service_path)?;
                output::success("Daemon service uninstalled");
            } else {
                output::info("Service not installed");
            }
        }
        _ => {
            output::error("Service uninstallation not supported on this platform");
        }
    }

    Ok(())
}

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
        _ => {
            // Fallback: try to kill by process name
            let _ = Command::new("pkill").args(["-f", DAEMON_BINARY]).status();
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
            match client.system().health_check(()).await {
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

async fn logs(lines: usize, follow: bool) -> Result<()> {
    // Determine log file location based on platform
    let log_paths = vec![
        "/tmp/memexd.out.log",
        "/tmp/memexd.err.log",
        "/var/log/memexd.log",
    ];

    let mut found_log = false;

    for log_path in &log_paths {
        let path = std::path::Path::new(log_path);
        if path.exists() {
            found_log = true;
            output::info(format!("Reading from {}", log_path));

            if follow {
                // Use tail -f for following
                let mut cmd = Command::new("tail");
                cmd.args(["-f", "-n", &lines.to_string(), log_path]);

                output::info("Press Ctrl+C to stop following...");
                let _ = cmd.status();
            } else {
                // Read last N lines
                let output_result = Command::new("tail")
                    .args(["-n", &lines.to_string(), log_path])
                    .output()?;

                if output_result.status.success() {
                    let content = String::from_utf8_lossy(&output_result.stdout);
                    println!("{}", content);
                }
            }
            break;
        }
    }

    if !found_log {
        // Try journalctl on Linux
        #[cfg(target_os = "linux")]
        {
            output::info("Checking journalctl...");
            let mut cmd = Command::new("journalctl");
            cmd.args(["--user", "-u", "memexd", "-n", &lines.to_string()]);

            if follow {
                cmd.arg("-f");
            }

            let _ = cmd.status();
            return Ok(());
        }

        output::warning("No log files found");
        output::info("Expected locations: /tmp/memexd.out.log, /tmp/memexd.err.log");
    }

    Ok(())
}
