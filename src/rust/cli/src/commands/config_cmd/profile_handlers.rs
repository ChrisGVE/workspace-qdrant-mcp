//! Handlers for profile-related `wqm config` subcommands.
//!
//! Profiles describe CLI connection targets (daemon gRPC endpoint, Qdrant
//! base URL, optional API-key env var, SQLite state path). The TOML file is
//! authoritative; everything else (DaemonClient, qdrant helpers) consults the
//! active profile through `crate::config`.

use anyhow::{Context, Result};
use colored::Colorize;

use wqm_common::cli_profiles::{ensure_cli_config, save_cli_config, CliConfigFile, Profile};

use crate::output;
use crate::output::style::home_to_tilde;

/// `wqm config list` — show all known profiles and mark the active one.
pub fn list_profiles() -> Result<()> {
    let (cfg, path) = ensure_cli_config().context("Failed to load CLI config")?;
    output::section("CLI Profiles");
    output::kv("Config file", home_to_tilde(&path.display().to_string()));
    output::separator();

    for profile in &cfg.profiles {
        let is_active = profile.name == cfg.active;
        let marker = if is_active {
            format!("{}", "●".green())
        } else {
            format!("{}", "○".dimmed())
        };
        let line = if is_active {
            format!("{} (active)", profile.name.bold())
        } else {
            profile.name.clone()
        };
        println!("{} {}", marker, line);
        if !profile.description.is_empty() {
            println!("    {}", profile.description.dimmed());
        }
        println!("    {} {}", "daemon:".dimmed(), profile.daemon_address);
        println!("    {} {}", "qdrant:".dimmed(), profile.qdrant_url);
        if !profile.qdrant_api_key_env.is_empty() {
            println!(
                "    {} ${}",
                "qdrant api-key env:".dimmed(),
                profile.qdrant_api_key_env
            );
        }
        if !profile.database_path.is_empty() {
            println!(
                "    {} {}",
                "state:".dimmed(),
                home_to_tilde(&profile.database_path)
            );
        }
    }
    Ok(())
}

/// `wqm config use <name>` — switch the active profile.
pub fn use_profile(name: &str) -> Result<()> {
    let (mut cfg, path) = ensure_cli_config().context("Failed to load CLI config")?;

    if cfg.active == name && cfg.find(name).is_some() {
        output::info(&format!("Profile {name:?} is already active."));
        return Ok(());
    }

    cfg.set_active(name)?;
    save_cli_config(&path, &cfg).context("Failed to write CLI config")?;
    output::success(&format!("Active profile → {}", name.bold()));
    Ok(())
}

/// `wqm config show` — describe the currently-active profile + endpoints.
pub fn show_active_profile() -> Result<()> {
    let (cfg, path) = ensure_cli_config().context("Failed to load CLI config")?;
    output::section("Active CLI Profile");
    output::kv("Config file", home_to_tilde(&path.display().to_string()));
    show_profile_section(&cfg, &cfg.active);
    output::separator();
    output::info(&format!(
        "Switch profiles: {}",
        "wqm config use <name>".bold()
    ));
    output::info(&format!("List profiles: {}", "wqm config list".bold()));
    output::info(&format!(
        "Merged YAML config: {}",
        "wqm config yaml-show".bold()
    ));
    Ok(())
}

fn show_profile_section(cfg: &CliConfigFile, name: &str) {
    let profile = cfg.find(name).cloned().unwrap_or_else(Profile::native);
    output::kv("Active profile", profile.name.as_str());
    if !profile.description.is_empty() {
        output::kv("Description", profile.description.as_str());
    }
    output::kv("Daemon address", profile.daemon_address.as_str());
    output::kv("Qdrant URL", profile.qdrant_url.as_str());
    if !profile.qdrant_api_key_env.is_empty() {
        output::kv(
            "Qdrant API key env",
            &format!("${}", profile.qdrant_api_key_env),
        );
    }
    if !profile.database_path.is_empty() {
        output::kv("State database", &home_to_tilde(&profile.database_path));
    }
    // Surface env overrides that would change the effective endpoints so
    // users can tell why `wqm status health` hits a different host than the
    // profile lists.
    if let Ok(v) = std::env::var("WQM_DAEMON_ADDR") {
        output::kv("Env override: WQM_DAEMON_ADDR", v.as_str());
    }
    if let Ok(v) = std::env::var("QDRANT_URL") {
        output::kv("Env override: QDRANT_URL", v.as_str());
    }
    if let Ok(v) = std::env::var("WQM_QDRANT_URL") {
        output::kv("Env override: WQM_QDRANT_URL", v.as_str());
    }
    if let Ok(v) = std::env::var("WQM_PROFILE") {
        output::kv("Env override: WQM_PROFILE", v.as_str());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn isolated_config(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        let path = tmp.path().join("wqm").join("cli-config.toml");
        std::env::set_var("WQM_CLI_CONFIG", &path);
        std::env::remove_var("WQM_PROFILE");
        path
    }

    #[test]
    fn use_profile_switches_active() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let tmp = tempfile::TempDir::new().unwrap();
        let path = isolated_config(&tmp);

        use_profile("docker-local").unwrap();
        let cfg = wqm_common::cli_profiles::load_cli_config_from(&path).unwrap();
        assert_eq!(cfg.active, "docker-local");

        use_profile("native").unwrap();
        let cfg = wqm_common::cli_profiles::load_cli_config_from(&path).unwrap();
        assert_eq!(cfg.active, "native");

        std::env::remove_var("WQM_CLI_CONFIG");
    }

    #[test]
    fn use_profile_errors_on_unknown_name() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let tmp = tempfile::TempDir::new().unwrap();
        let _path = isolated_config(&tmp);

        let err = use_profile("ghost").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ghost"),
            "error should mention the bad name: {msg}"
        );
        assert!(
            msg.contains("native"),
            "error should list known profiles: {msg}"
        );

        std::env::remove_var("WQM_CLI_CONFIG");
    }

    #[test]
    fn list_profiles_bootstraps_default_file() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let tmp = tempfile::TempDir::new().unwrap();
        let path = isolated_config(&tmp);
        assert!(!path.exists());

        list_profiles().unwrap();
        assert!(path.exists(), "list should create default file if missing");

        std::env::remove_var("WQM_CLI_CONFIG");
    }
}
