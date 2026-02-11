//! Hooks command - Claude Code integration hook management
//!
//! Installs/uninstalls file change notification hooks into Claude Code's
//! settings.json, enabling real-time ingestion during active coding sessions.
//!
//! Subcommands: install, uninstall, status

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::output;

/// Hooks command arguments
#[derive(Args)]
pub struct HooksArgs {
    #[command(subcommand)]
    command: HooksCommand,
}

/// Hooks subcommands
#[derive(Subcommand)]
enum HooksCommand {
    /// Install Claude Code hooks for file change notifications
    Install,

    /// Remove wqm hooks from Claude Code settings
    Uninstall,

    /// Check if hooks are installed
    Status,
}

/// Execute the hooks command
pub async fn execute(args: HooksArgs) -> Result<()> {
    match args.command {
        HooksCommand::Install => install_hooks().await,
        HooksCommand::Uninstall => uninstall_hooks().await,
        HooksCommand::Status => status_hooks().await,
    }
}

/// Marker comment embedded in hook matcher to identify wqm-managed hooks
const WQM_HOOK_MARKER: &str = "wqm-file-notify";

/// Get the path to Claude Code's settings.json
fn get_claude_settings_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".claude").join("settings.json"))
}

/// Read Claude Code settings.json, returning empty object if not found
fn read_settings(path: &Path) -> Result<serde_json::Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse {} as JSON", path.display()))
}

/// Write Claude Code settings.json with backup
fn write_settings(path: &Path, config: &serde_json::Value) -> Result<()> {
    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    // Create backup if file exists
    if path.exists() {
        let backup = path.with_extension("json.backup");
        std::fs::copy(path, &backup)
            .with_context(|| format!("Failed to create backup at {}", backup.display()))?;
    }

    // Write with pretty-printing
    let content = serde_json::to_string_pretty(config)
        .context("Failed to serialize settings")?;
    std::fs::write(path, format!("{}\n", content))
        .with_context(|| format!("Failed to write {}", path.display()))?;

    Ok(())
}

/// Build the wqm PostToolUse hook entry for file change notifications
fn build_wqm_hook_entry() -> serde_json::Value {
    json!({
        "matcher": WQM_HOOK_MARKER,
        "hooks": [
            {
                "type": "command",
                "command": "wqm notify file-changed"
            }
        ]
    })
}

/// Check if a hooks array already contains wqm-managed entries
fn has_wqm_hooks(hooks_array: &[serde_json::Value]) -> bool {
    hooks_array.iter().any(|entry| {
        entry.get("matcher")
            .and_then(|m| m.as_str())
            .map(|m| m.contains(WQM_HOOK_MARKER))
            .unwrap_or(false)
    })
}

/// Install Claude Code hooks for file change notifications
async fn install_hooks() -> Result<()> {
    let settings_path = get_claude_settings_path()?;
    output::kv("Settings path", settings_path.display());

    let mut config = read_settings(&settings_path)?;

    // Ensure hooks object exists
    if config.get("hooks").is_none() {
        config["hooks"] = json!({});
    }

    // Ensure PostToolUse array exists
    let hooks = config.get_mut("hooks").unwrap();
    if hooks.get("PostToolUse").is_none() {
        hooks["PostToolUse"] = json!([]);
    }

    let post_tool_use = hooks["PostToolUse"].as_array_mut()
        .context("hooks.PostToolUse is not an array")?;

    // Check if already installed
    if has_wqm_hooks(post_tool_use) {
        output::info("wqm hooks are already installed");
        return Ok(());
    }

    // Add wqm hook entry
    post_tool_use.push(build_wqm_hook_entry());

    write_settings(&settings_path, &config)?;
    output::success("Claude Code hooks installed");
    output::kv("Hook event", "PostToolUse");
    output::kv("Matcher", WQM_HOOK_MARKER);

    Ok(())
}

/// Remove wqm hooks from Claude Code settings
async fn uninstall_hooks() -> Result<()> {
    let settings_path = get_claude_settings_path()?;
    output::kv("Settings path", settings_path.display());

    if !settings_path.exists() {
        output::info("No Claude Code settings found - nothing to uninstall");
        return Ok(());
    }

    let mut config = read_settings(&settings_path)?;

    let hooks = match config.get_mut("hooks") {
        Some(h) => h,
        None => {
            output::info("No hooks section found - nothing to uninstall");
            return Ok(());
        }
    };

    let post_tool_use = match hooks.get_mut("PostToolUse") {
        Some(arr) => arr,
        None => {
            output::info("No PostToolUse hooks found - nothing to uninstall");
            return Ok(());
        }
    };

    let arr = post_tool_use.as_array_mut()
        .context("hooks.PostToolUse is not an array")?;

    let original_len = arr.len();
    arr.retain(|entry| {
        entry.get("matcher")
            .and_then(|m| m.as_str())
            .map(|m| !m.contains(WQM_HOOK_MARKER))
            .unwrap_or(true) // Keep entries without matcher
    });

    let removed = original_len - arr.len();

    if removed == 0 {
        output::info("No wqm hooks found - nothing to uninstall");
        return Ok(());
    }

    write_settings(&settings_path, &config)?;
    output::success(format!("Removed {} wqm hook(s) from Claude Code settings", removed));

    Ok(())
}

/// Check if hooks are installed
async fn status_hooks() -> Result<()> {
    let settings_path = get_claude_settings_path()?;
    output::kv("Settings path", settings_path.display());

    if !settings_path.exists() {
        output::kv("Status", "Not installed (no settings.json)");
        return Ok(());
    }

    let config = read_settings(&settings_path)?;

    let installed = config.get("hooks")
        .and_then(|h| h.get("PostToolUse"))
        .and_then(|arr| arr.as_array())
        .map(|arr| has_wqm_hooks(arr))
        .unwrap_or(false);

    if installed {
        output::success("wqm hooks are installed");
        output::kv("Hook event", "PostToolUse");
        output::kv("Matcher", WQM_HOOK_MARKER);
    } else {
        output::kv("Status", "Not installed");
        output::info("Run 'wqm hooks install' to set up file change notifications");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_settings(dir: &Path, content: &str) -> PathBuf {
        let settings_path = dir.join("settings.json");
        std::fs::write(&settings_path, content).unwrap();
        settings_path
    }

    #[test]
    fn test_build_wqm_hook_entry() {
        let entry = build_wqm_hook_entry();
        assert_eq!(
            entry["matcher"].as_str().unwrap(),
            WQM_HOOK_MARKER
        );
        let hooks = entry["hooks"].as_array().unwrap();
        assert_eq!(hooks.len(), 1);
        assert_eq!(hooks[0]["type"].as_str().unwrap(), "command");
        assert!(hooks[0]["command"].as_str().unwrap().contains("wqm notify"));
    }

    #[test]
    fn test_has_wqm_hooks_empty() {
        assert!(!has_wqm_hooks(&[]));
    }

    #[test]
    fn test_has_wqm_hooks_present() {
        let hooks = vec![build_wqm_hook_entry()];
        assert!(has_wqm_hooks(&hooks));
    }

    #[test]
    fn test_has_wqm_hooks_absent() {
        let hooks = vec![json!({
            "matcher": "other-hook",
            "hooks": [{"type": "command", "command": "echo test"}]
        })];
        assert!(!has_wqm_hooks(&hooks));
    }

    #[test]
    fn test_read_settings_nonexistent() {
        let result = read_settings(Path::new("/tmp/nonexistent_settings.json"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({}));
    }

    #[test]
    fn test_read_settings_valid() {
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            r#"{"permissions": {"allow": []}, "hooks": {}}"#,
        );
        let config = read_settings(&path).unwrap();
        assert!(config.get("permissions").is_some());
        assert!(config.get("hooks").is_some());
    }

    #[test]
    fn test_write_settings_creates_backup() {
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(dir.path(), r#"{"original": true}"#);

        let new_config = json!({"updated": true});
        write_settings(&path, &new_config).unwrap();

        // Verify backup exists
        let backup = path.with_extension("json.backup");
        assert!(backup.exists());
        let backup_content: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&backup).unwrap()).unwrap();
        assert_eq!(backup_content["original"], json!(true));

        // Verify new content
        let new_content: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(new_content["updated"], json!(true));
    }

    #[test]
    fn test_write_settings_creates_directories() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("dir").join("settings.json");

        let config = json!({"test": true});
        write_settings(&path, &config).unwrap();

        assert!(path.exists());
    }

    #[tokio::test]
    async fn test_install_uninstall_cycle() {
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            r#"{"hooks": {"SessionStart": [{"matcher": "startup", "hooks": []}]}}"#,
        );

        // Install: add hooks
        let mut config = read_settings(&path).unwrap();
        config["hooks"]["PostToolUse"] = json!([]);
        config["hooks"]["PostToolUse"]
            .as_array_mut()
            .unwrap()
            .push(build_wqm_hook_entry());
        write_settings(&path, &config).unwrap();

        // Verify installed
        let config = read_settings(&path).unwrap();
        let post = config["hooks"]["PostToolUse"].as_array().unwrap();
        assert!(has_wqm_hooks(post));

        // Verify other hooks preserved
        assert!(config["hooks"]["SessionStart"].is_array());

        // Uninstall: remove wqm hooks
        let mut config = read_settings(&path).unwrap();
        let arr = config["hooks"]["PostToolUse"].as_array_mut().unwrap();
        arr.retain(|e| {
            e.get("matcher")
                .and_then(|m| m.as_str())
                .map(|m| !m.contains(WQM_HOOK_MARKER))
                .unwrap_or(true)
        });
        write_settings(&path, &config).unwrap();

        // Verify uninstalled
        let config = read_settings(&path).unwrap();
        let post = config["hooks"]["PostToolUse"].as_array().unwrap();
        assert!(!has_wqm_hooks(post));

        // Verify SessionStart hooks preserved
        assert!(config["hooks"]["SessionStart"].is_array());
        assert!(!config["hooks"]["SessionStart"].as_array().unwrap().is_empty());
    }
}
