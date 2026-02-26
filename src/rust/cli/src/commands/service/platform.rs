//! Platform-specific service management logic
//!
//! Contains service manager detection, binary resolution, directory helpers,
//! and service config generators (launchd plist, systemd unit).

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::process::Command;

pub const SERVICE_NAME: &str = "com.workspace-qdrant.memexd";
pub const DAEMON_BINARY: &str = "memexd";

/// Platform service manager variants
#[derive(Debug)]
#[allow(dead_code)] // Variants used based on target platform
pub enum ServiceManager {
    Launchctl,
    Systemd,
    WindowsService,
    Unknown,
}

/// Get platform-specific service manager
pub fn get_service_manager() -> ServiceManager {
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

/// Check if daemon is running by attempting gRPC connection
pub async fn is_daemon_running() -> bool {
    crate::grpc::client::DaemonClient::connect_default()
        .await
        .is_ok()
}

/// Resolve the path to the memexd binary
pub fn find_daemon_binary(explicit: Option<String>) -> Result<PathBuf> {
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
pub fn install_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".local/bin"))
}

/// Get the canonical log directory for the daemon
pub fn log_directory() -> Result<PathBuf> {
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
pub fn generate_launchd_plist(binary_path: &std::path::Path) -> String {
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
pub fn generate_systemd_unit(binary_path: &std::path::Path) -> String {
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
    fn test_generate_launchd_plist_uses_canonical_log_dir() {
        let binary = PathBuf::from("/Users/test/.local/bin/memexd");
        let plist = generate_launchd_plist(&binary);

        // Must NOT use /tmp for logs (Task 515)
        assert!(
            !plist.contains("/tmp/memexd"),
            "plist must not use /tmp for log paths"
        );

        // Must use canonical log directory
        assert!(
            plist.contains("Library/Logs/workspace-qdrant"),
            "plist must use canonical log directory"
        );
        assert!(
            plist.contains("daemon.out.log"),
            "plist must define stdout log"
        );
        assert!(
            plist.contains("daemon.err.log"),
            "plist must define stderr log"
        );
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
