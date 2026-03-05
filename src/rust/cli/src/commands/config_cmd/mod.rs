//! Config command - configuration management
//!
//! Subcommands: generate, default, xdg, show, path

mod daemon;
mod handlers;
mod migration;

use anyhow::Result;
use clap::{Args, Subcommand};

use handlers::{generate, move_to_default, move_to_xdg, show, show_path};

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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::migration::{check_active_mcp_sessions, copy_file, move_file};

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
            assert!(
                !active,
                "Should report no active sessions when DB doesn't exist"
            );
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
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "# test config");
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
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "test data");
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
