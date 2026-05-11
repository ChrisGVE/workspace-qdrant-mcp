//! Config command - configuration management
//!
//! Daemon YAML subcommands: generate, yaml-show, path.
//! CLI connection profile subcommands: list, use, show.

mod handlers;
mod profile_handlers;

use anyhow::Result;
use clap::{Args, Subcommand};

use handlers::{generate, show as show_yaml, show_path};
use profile_handlers::{list_profiles, show_active_profile, use_profile};

/// Config command arguments
#[derive(Args)]
pub struct ConfigCmdArgs {
    #[command(subcommand)]
    command: ConfigCommand,
}

/// Config subcommands
#[derive(Subcommand)]
enum ConfigCommand {
    /// List CLI connection profiles
    List,
    /// Switch the active CLI connection profile
    Use {
        /// Profile name (e.g. native, docker-local)
        name: String,
    },
    /// Show the active CLI connection profile and its endpoints
    Show,
    /// Output the default daemon YAML configuration to stdout
    Generate,
    /// Show the merged daemon YAML configuration (defaults + user overrides)
    #[command(name = "yaml-show")]
    YamlShow,
    /// Show configuration file search paths and which one is active
    Path,
}

/// Execute config command
pub async fn execute(args: ConfigCmdArgs) -> Result<()> {
    match args.command {
        ConfigCommand::List => list_profiles(),
        ConfigCommand::Use { name } => use_profile(&name),
        ConfigCommand::Show => show_active_profile(),
        ConfigCommand::Generate => generate(),
        ConfigCommand::YamlShow => show_yaml(),
        ConfigCommand::Path => show_path(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_generate_produces_valid_yaml() {
        // DEFAULT_YAML_CONFIG is validated at compile time by LazyLock.
        // Accessing it forces the parse, confirming the YAML is valid.
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
}
