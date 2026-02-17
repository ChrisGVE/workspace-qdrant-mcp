//! Config command - configuration management
//!
//! Subcommands: generate, show, path

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

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
    /// Show the active configuration (merged defaults + user overrides)
    Show,
    /// Show configuration file search paths and which one is active
    Path,
}

/// Execute config command
pub async fn execute(args: ConfigCmdArgs) -> Result<()> {
    match args.command {
        ConfigCommand::Generate => generate(),
        ConfigCommand::Show => show(),
        ConfigCommand::Path => show_path(),
    }
}

/// Output the embedded default YAML configuration to stdout
fn generate() -> Result<()> {
    print!("{}", wqm_common::yaml_defaults::DEFAULT_YAML);
    Ok(())
}

/// Show the active configuration (defaults merged with user overrides)
fn show() -> Result<()> {
    let active_path = wqm_common::paths::find_config_file();

    match &active_path {
        Some(path) => {
            output::kv("Config file", &path.display().to_string());
            output::separator();
            let content = std::fs::read_to_string(path)
                .context("Failed to read config file")?;
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
        output::info("Run `wqm config generate > ~/.workspace-qdrant/config.yaml` to create one.");
    }

    output::separator();
    output::kv("Database", &wqm_common::paths::get_database_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "(not found)".into()));
    output::kv("Logs", &wqm_common::paths::get_canonical_log_dir().display().to_string());

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_generate_produces_valid_yaml() {
        // DEFAULT_YAML_CONFIG is validated at compile time by LazyLock
        // Accessing it forces the parse, confirming the YAML is valid
        let config = &*wqm_common::yaml_defaults::DEFAULT_YAML_CONFIG;
        assert!(config.embedding.model.len() > 0);
    }

    #[test]
    fn test_search_paths_not_empty() {
        let paths = wqm_common::paths::get_config_search_paths();
        assert!(!paths.is_empty(), "Should have at least one config search path");
    }
}
