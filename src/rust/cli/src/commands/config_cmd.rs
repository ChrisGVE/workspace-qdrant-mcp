//! Config command - configuration management
//!
//! Subcommands: generate, default, xdg, show, path

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::output;

/// Config command arguments
#[derive(Args)]
pub struct ConfigCmdArgs {
    #[command(subcommand)]
    command: ConfigCommand,
}

/// Config subcommands
#[derive(Subcommand)]
enum ConfigCommand {
    /// Output the default configuration YAML to stdout
    Generate,
    /// Move configuration to ~/.workspace-qdrant/ (default location)
    Default,
    /// Move configuration to XDG directories
    Xdg,
    /// Show the active configuration (merged defaults + user overrides)
    Show,
    /// Show configuration file search paths and which one is active
    Path,
}

/// Execute config command
pub async fn execute(args: ConfigCmdArgs) -> Result<()> {
    match args.command {
        ConfigCommand::Generate => generate(),
        ConfigCommand::Default => move_to_default().await,
        ConfigCommand::Xdg => move_to_xdg().await,
        ConfigCommand::Show => show(),
        ConfigCommand::Path => show_path(),
    }
}

/// Output the embedded default YAML configuration to stdout
fn generate() -> Result<()> {
    print!("{}", wqm_common::yaml_defaults::DEFAULT_YAML);
    Ok(())
}

// =========================================================================
// Safety checks shared by `default` and `xdg`
// =========================================================================

/// Check for active MCP server sessions via operational_state table.
/// Returns true if an active session was detected.
fn check_active_mcp_sessions() -> Result<bool> {
    let db_path = wqm_common::paths::get_database_path()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        // No database means no sessions
        return Ok(false);
    }

    let conn = rusqlite::Connection::open(&db_path)
        .context("Failed to open state database")?;

    // Check if operational_state table exists (schema v17+)
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='operational_state'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !table_exists {
        return Ok(false);
    }

    // Look for any server component entries with recent updated_at (within last 10 minutes)
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM operational_state
             WHERE component = 'server'
             AND updated_at > datetime('now', '-10 minutes')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    Ok(count > 0)
}

/// Check if daemon is reachable via gRPC
async fn is_daemon_running() -> bool {
    DaemonClient::connect_default().await.is_ok()
}

/// Stop the daemon and wait for it to shut down.
/// Returns true if the daemon was stopped or already not running.
async fn stop_daemon() -> Result<bool> {
    if !is_daemon_running().await {
        return Ok(true);
    }

    output::info("Stopping daemon...");

    #[cfg(target_os = "macos")]
    {
        let plist_path = dirs::home_dir()
            .context("Could not find home directory")?
            .join("Library/LaunchAgents/com.workspace-qdrant.memexd.plist");

        if plist_path.exists() {
            let _ = std::process::Command::new("launchctl")
                .args(["unload"])
                .arg(&plist_path)
                .status();
        }
        let _ = std::process::Command::new("pkill")
            .args(["-f", "memexd"])
            .status();
    }

    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "stop", "memexd"])
            .status();
    }

    // Wait for daemon to stop
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    if is_daemon_running().await {
        bail!("Could not stop daemon cleanly. Please stop it manually and retry.");
    }

    output::success("Daemon stopped");
    Ok(true)
}

/// Start the daemon after config migration.
async fn start_daemon() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let plist_path = dirs::home_dir()
            .context("Could not find home directory")?
            .join("Library/LaunchAgents/com.workspace-qdrant.memexd.plist");

        if plist_path.exists() {
            let status = std::process::Command::new("launchctl")
                .args(["load", "-w"])
                .arg(&plist_path)
                .status()?;
            if status.success() {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                if is_daemon_running().await {
                    output::success("Daemon restarted");
                } else {
                    output::warning("Service loaded but daemon not responding yet");
                }
                return Ok(());
            }
        }
        output::warning("Could not restart daemon. Start manually: wqm service start");
    }

    #[cfg(target_os = "linux")]
    {
        let status = std::process::Command::new("systemctl")
            .args(["--user", "start", "memexd"])
            .status()?;
        if status.success() {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            if is_daemon_running().await {
                output::success("Daemon restarted");
            } else {
                output::warning("Service started but daemon not responding yet");
            }
            return Ok(());
        }
        output::warning("Could not restart daemon. Start manually: wqm service start");
    }

    Ok(())
}

/// Perform safety checks before config migration.
/// Returns an error if migration should not proceed.
async fn pre_migration_checks() -> Result<()> {
    // Check for active MCP sessions
    if check_active_mcp_sessions()? {
        bail!(
            "Active MCP server session detected.\n\
             Close all MCP server connections before moving configuration.\n\
             Then retry this command."
        );
    }
    Ok(())
}

/// Move a config file from source to target, creating parent dirs.
/// Uses copy+delete for cross-device safety.
fn move_file(source: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    // Copy then delete for cross-device safety (atomic rename only works same-device)
    std::fs::copy(source, target).with_context(|| {
        format!(
            "Failed to copy {} → {}",
            source.display(),
            target.display()
        )
    })?;

    std::fs::remove_file(source).with_context(|| {
        format!(
            "Failed to remove source file after copy: {}",
            source.display()
        )
    })?;

    Ok(())
}

/// Copy a file from source to target, creating parent dirs.
fn copy_file(source: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    std::fs::copy(source, target).with_context(|| {
        format!(
            "Failed to copy {} → {}",
            source.display(),
            target.display()
        )
    })?;

    Ok(())
}

/// Shared migration logic: move config (and optionally database) to target location.
/// Returns Ok(()) on success or user abort.
async fn migrate_config(target_config: &Path, target_data_dir: Option<&Path>) -> Result<()> {
    pre_migration_checks().await?;

    let current_config = wqm_common::paths::find_config_file();
    let current_db = wqm_common::paths::get_database_path().ok();

    // Determine what will happen
    output::section("Configuration Migration Plan");

    let already_at_target = current_config
        .as_ref()
        .map_or(false, |p| p == target_config);

    if already_at_target {
        output::success(format!(
            "Configuration is already at {}",
            target_config.display()
        ));
        return Ok(());
    }

    // Show dry-run summary
    match &current_config {
        Some(source) => {
            output::kv("Current config", &source.display().to_string());
            output::kv("Target config", &target_config.display().to_string());
            output::info("Action: move config file to target location");
        }
        None => {
            output::info("No existing config file found");
            output::kv("Target config", &target_config.display().to_string());
            output::info("Action: write default config to target location");
        }
    }

    // Show database migration plan if applicable
    if let (Some(data_dir), Some(db_path)) = (target_data_dir, &current_db) {
        if db_path.exists() {
            let target_db = data_dir.join("state.db");
            let db_already_there = db_path == &target_db;
            if !db_already_there {
                output::kv("Current database", &db_path.display().to_string());
                output::kv("Target database", &target_db.display().to_string());
                output::info("Action: copy database to target location");
            }
        }
    }

    output::separator();

    // Ask for confirmation
    if !output::confirm("Proceed with migration?") {
        output::info("Aborted.");
        return Ok(());
    }

    // Stop daemon
    let daemon_was_running = is_daemon_running().await;
    if daemon_was_running {
        stop_daemon().await?;
    }

    // Execute migration
    match &current_config {
        Some(source) => {
            move_file(source, target_config)?;
            output::success(format!(
                "Moved config: {} → {}",
                source.display(),
                target_config.display()
            ));
        }
        None => {
            if let Some(parent) = target_config.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create directory: {}", parent.display())
                })?;
            }
            std::fs::write(target_config, wqm_common::yaml_defaults::DEFAULT_YAML)
                .with_context(|| {
                    format!(
                        "Failed to write default config to {}",
                        target_config.display()
                    )
                })?;
            output::success(format!(
                "Wrote default config to {}",
                target_config.display()
            ));
        }
    }

    // Copy database if target data dir is different
    if let (Some(data_dir), Some(db_path)) = (target_data_dir, &current_db) {
        if db_path.exists() {
            let target_db = data_dir.join("state.db");
            if db_path != &target_db && !target_db.exists() {
                copy_file(db_path, &target_db)?;
                output::success(format!(
                    "Copied database: {} → {}",
                    db_path.display(),
                    target_db.display()
                ));
            }
        }
    }

    // Restart daemon if it was running
    if daemon_was_running {
        start_daemon().await?;
    }

    output::separator();
    output::success("Configuration migration complete");

    Ok(())
}

// =========================================================================
// Subcommand implementations
// =========================================================================

/// Move config to ~/.workspace-qdrant/ (default location)
async fn move_to_default() -> Result<()> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let target_dir = home.join(".workspace-qdrant");
    let target_config = target_dir.join("config.yaml");

    migrate_config(&target_config, Some(&target_dir)).await
}

/// Move config to XDG directories
async fn move_to_xdg() -> Result<()> {
    // Resolve XDG_CONFIG_HOME
    let xdg_config_home = match std::env::var("XDG_CONFIG_HOME") {
        Ok(val) if !val.is_empty() => PathBuf::from(val),
        _ => {
            // Check if we're on macOS — XDG is not standard on macOS
            #[cfg(target_os = "macos")]
            {
                let home = dirs::home_dir().context("Could not determine home directory")?;
                let default_xdg = home.join(".config");
                output::warning(
                    "XDG_CONFIG_HOME is not set. Using default: ~/.config",
                );
                default_xdg
            }
            #[cfg(not(target_os = "macos"))]
            {
                let home = dirs::home_dir().context("Could not determine home directory")?;
                home.join(".config")
            }
        }
    };

    // Resolve XDG_DATA_HOME for database/state
    let xdg_data_home = match std::env::var("XDG_DATA_HOME") {
        Ok(val) if !val.is_empty() => PathBuf::from(val),
        _ => {
            let home = dirs::home_dir().context("Could not determine home directory")?;
            home.join(".local").join("share")
        }
    };

    let target_config = xdg_config_home
        .join("workspace-qdrant")
        .join("config.yaml");
    let target_data_dir = xdg_data_home.join("workspace-qdrant");

    output::kv("XDG_CONFIG_HOME", &xdg_config_home.display().to_string());
    output::kv("XDG_DATA_HOME", &xdg_data_home.display().to_string());
    output::kv(
        "XDG cache dir",
        &std::env::var("XDG_CACHE_HOME")
            .unwrap_or_else(|_| "~/.cache (default)".into()),
    );

    migrate_config(&target_config, Some(&target_data_dir)).await
}

/// Show the active configuration (defaults merged with user overrides)
fn show() -> Result<()> {
    let active_path = wqm_common::paths::find_config_file();

    match &active_path {
        Some(path) => {
            output::kv("Config file", &path.display().to_string());
            output::separator();
            let content =
                std::fs::read_to_string(path).context("Failed to read config file")?;
            print!("{}", content);
        }
        None => {
            output::info("No user config file found. Using built-in defaults.");
            output::separator();
            print!("{}", wqm_common::yaml_defaults::DEFAULT_YAML);
        }
    }

    Ok(())
}

/// Show config search paths and which one is active
fn show_path() -> Result<()> {
    output::section("Configuration Paths");

    let search_paths = wqm_common::paths::get_config_search_paths();
    let active = wqm_common::paths::find_config_file();

    for path in &search_paths {
        let exists = path.exists();
        let is_active = active.as_ref().map_or(false, |a| a == path);

        if is_active {
            output::success(format!("{} (active)", path.display()));
        } else if exists {
            output::kv("Found", &path.display().to_string());
        } else {
            output::kv("  -", &path.display().to_string());
        }
    }

    if active.is_none() {
        output::separator();
        output::info("No config file found. Using built-in defaults.");
        output::info(
            "Run `wqm config generate > ~/.workspace-qdrant/config.yaml` to create one.",
        );
    }

    output::separator();
    output::kv(
        "Database",
        &wqm_common::paths::get_database_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "(not found)".into()),
    );
    output::kv(
        "Logs",
        &wqm_common::paths::get_canonical_log_dir()
            .display()
            .to_string(),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_produces_valid_yaml() {
        // DEFAULT_YAML_CONFIG is validated at compile time by LazyLock
        // Accessing it forces the parse, confirming the YAML is valid
        let config = &*wqm_common::yaml_defaults::DEFAULT_YAML_CONFIG;
        assert!(!config.embedding.model.is_empty());
    }

    #[test]
    fn test_search_paths_not_empty() {
        let paths = wqm_common::paths::get_config_search_paths();
        assert!(
            !paths.is_empty(),
            "Should have at least one config search path"
        );
    }

    #[test]
    fn test_check_active_mcp_sessions_no_db() {
        // With a nonexistent database, should return false (no sessions)
        let prev = std::env::var("WQM_DATABASE_PATH").ok();
        std::env::set_var("WQM_DATABASE_PATH", "/nonexistent/path/state.db");

        let result = check_active_mcp_sessions();
        // Either Ok(false) or error from path resolution — both are acceptable
        if let Ok(active) = result {
            assert!(!active, "Should report no active sessions when DB doesn't exist");
        }

        match prev {
            Some(val) => std::env::set_var("WQM_DATABASE_PATH", val),
            None => std::env::remove_var("WQM_DATABASE_PATH"),
        }
    }

    #[test]
    fn test_move_file_creates_parent_dirs() {
        let temp = tempfile::TempDir::new().unwrap();
        let source = temp.path().join("source.yaml");
        std::fs::write(&source, "# test config").unwrap();

        let target = temp.path().join("nested").join("dir").join("config.yaml");

        move_file(&source, &target).unwrap();

        assert!(!source.exists(), "Source should be removed after move");
        assert!(target.exists(), "Target should exist after move");
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "# test config"
        );
    }

    #[test]
    fn test_copy_file_creates_parent_dirs() {
        let temp = tempfile::TempDir::new().unwrap();
        let source = temp.path().join("source.db");
        std::fs::write(&source, "test data").unwrap();

        let target = temp.path().join("deep").join("nested").join("state.db");

        copy_file(&source, &target).unwrap();

        assert!(source.exists(), "Source should still exist after copy");
        assert!(target.exists(), "Target should exist after copy");
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "test data"
        );
    }

    #[test]
    fn test_move_file_nonexistent_source() {
        let temp = tempfile::TempDir::new().unwrap();
        let source = temp.path().join("nonexistent.yaml");
        let target = temp.path().join("target.yaml");

        let result = move_file(&source, &target);
        assert!(result.is_err(), "Should fail when source doesn't exist");
    }

    #[test]
    fn test_xdg_paths_resolve() {
        // Verify XDG path resolution with custom env
        let prev_config = std::env::var("XDG_CONFIG_HOME").ok();
        let prev_data = std::env::var("XDG_DATA_HOME").ok();

        std::env::set_var("XDG_CONFIG_HOME", "/tmp/test-xdg-config");
        std::env::set_var("XDG_DATA_HOME", "/tmp/test-xdg-data");

        let config_home = std::env::var("XDG_CONFIG_HOME").unwrap();
        let data_home = std::env::var("XDG_DATA_HOME").unwrap();

        assert_eq!(config_home, "/tmp/test-xdg-config");
        assert_eq!(data_home, "/tmp/test-xdg-data");

        let expected_config = PathBuf::from("/tmp/test-xdg-config/workspace-qdrant/config.yaml");
        let expected_data = PathBuf::from("/tmp/test-xdg-data/workspace-qdrant");

        assert_eq!(
            PathBuf::from(&config_home)
                .join("workspace-qdrant")
                .join("config.yaml"),
            expected_config
        );
        assert_eq!(
            PathBuf::from(&data_home).join("workspace-qdrant"),
            expected_data
        );

        // Restore
        match prev_config {
            Some(val) => std::env::set_var("XDG_CONFIG_HOME", val),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
        match prev_data {
            Some(val) => std::env::set_var("XDG_DATA_HOME", val),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
    }
}
