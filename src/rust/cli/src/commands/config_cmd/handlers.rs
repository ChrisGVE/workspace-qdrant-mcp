//! Subcommand handler implementations for the config command.
//!
//! Each function here corresponds to one of the `ConfigCommand` variants:
//! `generate`, `default`, `xdg`, `show`, and `path`.

use anyhow::{Context, Result};
use colored::Colorize;

use super::migration::migrate_config;
use crate::output;
use crate::output::style::home_to_tilde;

/// Output the embedded default YAML configuration to stdout.
pub(super) fn generate() -> Result<()> {
    print!("{}", wqm_common::yaml_defaults::DEFAULT_YAML);
    Ok(())
}

/// Move config to ~/.config/workspace-qdrant/ (default location).
pub(super) async fn move_to_default() -> Result<()> {
    let target_dir =
        wqm_common::paths::get_config_dir().context("Could not determine config directory")?;
    let target_config = target_dir.join("config.yaml");

    migrate_config(&target_config, Some(&target_dir)).await
}

/// Move config to XDG directories.
pub(super) async fn move_to_xdg() -> Result<()> {
    // Resolve XDG_CONFIG_HOME
    let xdg_config_home = match std::env::var("XDG_CONFIG_HOME") {
        Ok(val) if !val.is_empty() => std::path::PathBuf::from(val),
        _ => {
            // Check if we're on macOS — XDG is not standard on macOS
            #[cfg(target_os = "macos")]
            {
                let home = dirs::home_dir().context("Could not determine home directory")?;
                let default_xdg = home.join(".config");
                output::warning("XDG_CONFIG_HOME is not set. Using default: ~/.config");
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
        Ok(val) if !val.is_empty() => std::path::PathBuf::from(val),
        _ => {
            let home = dirs::home_dir().context("Could not determine home directory")?;
            home.join(".local").join("share")
        }
    };

    let target_config = xdg_config_home.join("workspace-qdrant").join("config.yaml");
    let target_data_dir = xdg_data_home.join("workspace-qdrant");

    output::kv(
        "XDG_CONFIG_HOME",
        home_to_tilde(&xdg_config_home.display().to_string()),
    );
    output::kv(
        "XDG_DATA_HOME",
        home_to_tilde(&xdg_data_home.display().to_string()),
    );
    output::kv(
        "XDG cache dir",
        std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| "~/.cache (default)".into()),
    );

    migrate_config(&target_config, Some(&target_data_dir)).await
}

/// Show the active configuration (defaults merged with user overrides).
pub(super) fn show() -> Result<()> {
    let active_path = wqm_common::paths::find_config_file();

    match &active_path {
        Some(path) => {
            output::kv("Config file", home_to_tilde(&path.display().to_string()));
            output::separator();
            let content = std::fs::read_to_string(path).context("Failed to read config file")?;
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

/// Show config search paths and which one is active.
pub(super) fn show_path() -> Result<()> {
    output::section("Configuration Paths");

    let search_paths = wqm_common::paths::get_config_search_paths();
    let active = wqm_common::paths::find_config_file();

    for path in &search_paths {
        let display = home_to_tilde(&path.display().to_string());
        let is_active = active.as_ref() == Some(path);

        if is_active {
            println!("{} {} (active)", "●".green(), display);
        } else {
            println!("{} {}", "○".dimmed(), display.dimmed());
        }
    }

    if active.is_none() {
        output::separator();
        output::info("No config file found. Using built-in defaults.");
        let config_path = wqm_common::paths::get_config_dir()
            .map(|d| d.join("config.yaml").display().to_string())
            .map(|p| home_to_tilde(&p))
            .unwrap_or_else(|_| "<config_dir>/config.yaml".to_string());
        output::info(format!(
            "Run `wqm config generate > {}` to create one.",
            config_path
        ));
    }

    output::separator();
    output::kv(
        "Database",
        wqm_common::paths::get_database_path()
            .map(|p| home_to_tilde(&p.display().to_string()))
            .unwrap_or_else(|_| "(not found)".into()),
    );
    output::kv(
        "Logs",
        home_to_tilde(
            &wqm_common::paths::get_canonical_log_dir()
                .display()
                .to_string(),
        ),
    );

    Ok(())
}
