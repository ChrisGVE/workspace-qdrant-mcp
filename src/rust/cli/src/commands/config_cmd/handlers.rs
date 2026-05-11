//! Subcommand handler implementations for the config command.
//!
//! Each function here corresponds to one of the `ConfigCommand` variants:
//! `generate`, `show`, and `path`.

use anyhow::{Context, Result};
use colored::Colorize;

use crate::output;
use crate::output::style::home_to_tilde;

/// Output the embedded default YAML configuration to stdout.
pub(super) fn generate() -> Result<()> {
    print!("{}", wqm_common::yaml_defaults::DEFAULT_YAML);
    Ok(())
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
