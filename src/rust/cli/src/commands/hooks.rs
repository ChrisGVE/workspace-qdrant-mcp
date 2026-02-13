//! Hooks command - Claude Code integration hook management
//!
//! Installs/uninstalls a SessionStart hook into Claude Code's settings.json
//! that injects workspace-qdrant memory rules into context at session start,
//! `/clear`, and compaction.
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
    /// Install Claude Code SessionStart hook for memory rule injection
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

/// The matcher pattern that triggers the hook
const SESSION_START_MATCHER: &str = "startup|clear|compact";

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

/// Build the wqm hook command entry
fn build_wqm_hook_command() -> serde_json::Value {
    json!({
        "type": "command",
        "command": "wqm memory inject"
    })
}

/// Check if a hook command entry is a wqm command
fn is_wqm_hook_command(hook: &serde_json::Value) -> bool {
    hook.get("command")
        .and_then(|c| c.as_str())
        .map(|c| c.starts_with("wqm "))
        .unwrap_or(false)
}

/// Check if a matcher group has our matcher pattern
fn is_our_matcher(entry: &serde_json::Value) -> bool {
    entry
        .get("matcher")
        .and_then(|m| m.as_str())
        .map(|m| m == SESSION_START_MATCHER)
        .unwrap_or(false)
}

/// Check if a matcher group's hooks array contains a wqm command
fn group_has_wqm_command(entry: &serde_json::Value) -> bool {
    entry
        .get("hooks")
        .and_then(|h| h.as_array())
        .map(|hooks| hooks.iter().any(is_wqm_hook_command))
        .unwrap_or(false)
}

/// Install Claude Code SessionStart hook for memory rule injection
async fn install_hooks() -> Result<()> {
    let settings_path = get_claude_settings_path()?;
    output::kv("Settings path", settings_path.display());

    let mut config = read_settings(&settings_path)?;

    // Ensure hooks.SessionStart array exists
    if config.get("hooks").is_none() {
        config["hooks"] = json!({});
    }
    let hooks = config.get_mut("hooks").unwrap();
    if hooks.get("SessionStart").is_none() {
        hooks["SessionStart"] = json!([]);
    }

    let session_start = hooks["SessionStart"]
        .as_array_mut()
        .context("hooks.SessionStart is not an array")?;

    // Find existing matcher group for our pattern
    let existing_idx = session_start
        .iter()
        .position(is_our_matcher);

    match existing_idx {
        Some(idx) => {
            // Matcher group exists — check if our command is already there
            if group_has_wqm_command(&session_start[idx]) {
                output::info("wqm hooks are already installed");
                return Ok(());
            }
            // Append our command to the existing group's hooks array
            session_start[idx]["hooks"]
                .as_array_mut()
                .context("Matcher group hooks is not an array")?
                .push(build_wqm_hook_command());
        }
        None => {
            // No matching group — add a new one
            session_start.push(json!({
                "matcher": SESSION_START_MATCHER,
                "hooks": [build_wqm_hook_command()]
            }));
        }
    }

    write_settings(&settings_path, &config)?;
    output::success("Claude Code SessionStart hook installed");
    output::kv("Hook event", "SessionStart");
    output::kv("Matcher", SESSION_START_MATCHER);
    output::kv("Command", "wqm memory inject");

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

    let session_start = match hooks.get_mut("SessionStart") {
        Some(arr) => arr,
        None => {
            output::info("No SessionStart hooks found - nothing to uninstall");
            return Ok(());
        }
    };

    let arr = session_start
        .as_array_mut()
        .context("hooks.SessionStart is not an array")?;

    let mut changed = false;

    // For each matcher group, filter out wqm commands
    let mut to_remove = Vec::new();
    for (idx, entry) in arr.iter_mut().enumerate() {
        if let Some(hooks_arr) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) {
            let before = hooks_arr.len();
            hooks_arr.retain(|h| !is_wqm_hook_command(h));
            if hooks_arr.len() < before {
                changed = true;
            }
            // Mark empty groups for removal
            if hooks_arr.is_empty() {
                to_remove.push(idx);
            }
        }
    }

    // Remove empty matcher groups (in reverse to preserve indices)
    for idx in to_remove.into_iter().rev() {
        arr.remove(idx);
    }

    if !changed {
        output::info("No wqm hooks found - nothing to uninstall");
        return Ok(());
    }

    write_settings(&settings_path, &config)?;
    output::success("Removed wqm hook(s) from Claude Code settings");

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

    let installed = config
        .get("hooks")
        .and_then(|h| h.get("SessionStart"))
        .and_then(|arr| arr.as_array())
        .map(|arr| arr.iter().any(|e| is_our_matcher(e) && group_has_wqm_command(e)))
        .unwrap_or(false);

    if installed {
        output::success("wqm hooks are installed");
        output::kv("Hook event", "SessionStart");
        output::kv("Matcher", SESSION_START_MATCHER);
        output::kv("Command", "wqm memory inject");
    } else {
        output::kv("Status", "Not installed");
        output::info("Run 'wqm hooks install' to set up memory rule injection");
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

    // ─── Unit tests ─────────────────────────────────────────────────────

    #[test]
    fn test_build_wqm_hook_command() {
        let cmd = build_wqm_hook_command();
        assert_eq!(cmd["type"].as_str().unwrap(), "command");
        assert_eq!(cmd["command"].as_str().unwrap(), "wqm memory inject");
    }

    #[test]
    fn test_is_wqm_hook_command_true() {
        let cmd = json!({"type": "command", "command": "wqm memory inject"});
        assert!(is_wqm_hook_command(&cmd));
    }

    #[test]
    fn test_is_wqm_hook_command_false() {
        let cmd = json!({"type": "command", "command": "echo hello"});
        assert!(!is_wqm_hook_command(&cmd));
    }

    #[test]
    fn test_is_wqm_hook_command_no_command_field() {
        let cmd = json!({"type": "url", "url": "https://example.com"});
        assert!(!is_wqm_hook_command(&cmd));
    }

    #[test]
    fn test_is_our_matcher() {
        let entry = json!({"matcher": "startup|clear|compact", "hooks": []});
        assert!(is_our_matcher(&entry));
    }

    #[test]
    fn test_is_our_matcher_different() {
        let entry = json!({"matcher": "Write|Edit", "hooks": []});
        assert!(!is_our_matcher(&entry));
    }

    // ─── Settings I/O tests ─────────────────────────────────────────────

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

        let backup = path.with_extension("json.backup");
        assert!(backup.exists());
        let backup_content: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&backup).unwrap()).unwrap();
        assert_eq!(backup_content["original"], json!(true));

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

    // ─── Install tests ──────────────────────────────────────────────────

    #[test]
    fn test_install_empty() {
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(dir.path(), r#"{}"#);

        let mut config = read_settings(&path).unwrap();

        // Simulate install logic
        config["hooks"] = json!({});
        config["hooks"]["SessionStart"] = json!([]);
        config["hooks"]["SessionStart"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "matcher": SESSION_START_MATCHER,
                "hooks": [build_wqm_hook_command()]
            }));
        write_settings(&path, &config).unwrap();

        // Verify
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(ss.len(), 1);
        assert!(is_our_matcher(&ss[0]));
        assert!(group_has_wqm_command(&ss[0]));
    }

    #[test]
    fn test_install_merge_existing_matcher() {
        // Pre-existing matcher group with another hook
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            &serde_json::to_string(&json!({
                "hooks": {
                    "SessionStart": [{
                        "matcher": SESSION_START_MATCHER,
                        "hooks": [{"type": "command", "command": "other-tool init"}]
                    }]
                }
            }))
            .unwrap(),
        );

        let mut config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array_mut().unwrap();

        // Find our matcher and append
        let idx = ss.iter().position(is_our_matcher).unwrap();
        ss[idx]["hooks"]
            .as_array_mut()
            .unwrap()
            .push(build_wqm_hook_command());

        write_settings(&path, &config).unwrap();

        // Verify: one matcher group, two hooks
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(ss.len(), 1);
        let hooks = ss[0]["hooks"].as_array().unwrap();
        assert_eq!(hooks.len(), 2);
        assert_eq!(hooks[0]["command"].as_str().unwrap(), "other-tool init");
        assert_eq!(hooks[1]["command"].as_str().unwrap(), "wqm memory inject");
    }

    #[test]
    fn test_install_new_matcher_group() {
        // Pre-existing different matcher group
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            &serde_json::to_string(&json!({
                "hooks": {
                    "SessionStart": [{
                        "matcher": "Write|Edit",
                        "hooks": [{"type": "command", "command": "other-tool check"}]
                    }]
                }
            }))
            .unwrap(),
        );

        let mut config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array_mut().unwrap();

        // No matching group → add new
        assert!(ss.iter().position(is_our_matcher).is_none());
        ss.push(json!({
            "matcher": SESSION_START_MATCHER,
            "hooks": [build_wqm_hook_command()]
        }));

        write_settings(&path, &config).unwrap();

        // Verify: two matcher groups
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(ss.len(), 2);
        assert_eq!(ss[0]["matcher"].as_str().unwrap(), "Write|Edit");
        assert_eq!(ss[1]["matcher"].as_str().unwrap(), SESSION_START_MATCHER);
    }

    #[test]
    fn test_install_idempotent() {
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            &serde_json::to_string(&json!({
                "hooks": {
                    "SessionStart": [{
                        "matcher": SESSION_START_MATCHER,
                        "hooks": [build_wqm_hook_command()]
                    }]
                }
            }))
            .unwrap(),
        );

        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        let idx = ss.iter().position(is_our_matcher).unwrap();
        assert!(group_has_wqm_command(&ss[idx]));
        // Already installed → no-op
    }

    // ─── Uninstall tests ────────────────────────────────────────────────

    #[test]
    fn test_uninstall_only_wqm_command() {
        // Two hooks in the same group, only wqm should be removed
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            &serde_json::to_string(&json!({
                "hooks": {
                    "SessionStart": [{
                        "matcher": SESSION_START_MATCHER,
                        "hooks": [
                            {"type": "command", "command": "other-tool init"},
                            build_wqm_hook_command()
                        ]
                    }]
                }
            }))
            .unwrap(),
        );

        let mut config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array_mut().unwrap();

        for entry in ss.iter_mut() {
            if let Some(hooks) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) {
                hooks.retain(|h| !is_wqm_hook_command(h));
            }
        }

        write_settings(&path, &config).unwrap();

        // Verify: group preserved with one hook
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(ss.len(), 1);
        let hooks = ss[0]["hooks"].as_array().unwrap();
        assert_eq!(hooks.len(), 1);
        assert_eq!(hooks[0]["command"].as_str().unwrap(), "other-tool init");
    }

    #[test]
    fn test_uninstall_removes_empty_group() {
        // Only wqm hook in the group → entire group should be removed
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            &serde_json::to_string(&json!({
                "hooks": {
                    "SessionStart": [{
                        "matcher": SESSION_START_MATCHER,
                        "hooks": [build_wqm_hook_command()]
                    }]
                }
            }))
            .unwrap(),
        );

        let mut config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array_mut().unwrap();

        let mut to_remove = Vec::new();
        for (idx, entry) in ss.iter_mut().enumerate() {
            if let Some(hooks) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) {
                hooks.retain(|h| !is_wqm_hook_command(h));
                if hooks.is_empty() {
                    to_remove.push(idx);
                }
            }
        }
        for idx in to_remove.into_iter().rev() {
            ss.remove(idx);
        }

        write_settings(&path, &config).unwrap();

        // Verify: SessionStart is now empty
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert!(ss.is_empty());
    }

    #[test]
    fn test_uninstall_preserves_other_hooks() {
        // Two matcher groups, only the one with wqm should be affected
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            &serde_json::to_string(&json!({
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "Write|Edit",
                            "hooks": [{"type": "command", "command": "other-tool check"}]
                        },
                        {
                            "matcher": SESSION_START_MATCHER,
                            "hooks": [build_wqm_hook_command()]
                        }
                    ],
                    "PostToolUse": [
                        {"matcher": "legacy", "hooks": []}
                    ]
                }
            }))
            .unwrap(),
        );

        let mut config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array_mut().unwrap();

        let mut to_remove = Vec::new();
        for (idx, entry) in ss.iter_mut().enumerate() {
            if let Some(hooks) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) {
                hooks.retain(|h| !is_wqm_hook_command(h));
                if hooks.is_empty() {
                    to_remove.push(idx);
                }
            }
        }
        for idx in to_remove.into_iter().rev() {
            ss.remove(idx);
        }

        write_settings(&path, &config).unwrap();

        // Verify: Write|Edit group preserved, wqm group removed
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(ss.len(), 1);
        assert_eq!(ss[0]["matcher"].as_str().unwrap(), "Write|Edit");

        // PostToolUse untouched
        assert!(config["hooks"]["PostToolUse"].is_array());
    }

    // ─── Integration test ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_install_uninstall_cycle() {
        let dir = TempDir::new().unwrap();
        let path = create_test_settings(
            dir.path(),
            r#"{"hooks": {"PostToolUse": [{"matcher": "legacy", "hooks": []}]}}"#,
        );

        // Install: add SessionStart hook
        let mut config = read_settings(&path).unwrap();
        config["hooks"]["SessionStart"] = json!([{
            "matcher": SESSION_START_MATCHER,
            "hooks": [build_wqm_hook_command()]
        }]);
        write_settings(&path, &config).unwrap();

        // Verify installed
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert!(ss.iter().any(|e| is_our_matcher(e) && group_has_wqm_command(e)));

        // Verify PostToolUse preserved
        assert!(config["hooks"]["PostToolUse"].is_array());

        // Uninstall: remove wqm hooks from SessionStart
        let mut config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array_mut().unwrap();
        let mut to_remove = Vec::new();
        for (idx, entry) in ss.iter_mut().enumerate() {
            if let Some(hooks) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) {
                hooks.retain(|h| !is_wqm_hook_command(h));
                if hooks.is_empty() {
                    to_remove.push(idx);
                }
            }
        }
        for idx in to_remove.into_iter().rev() {
            ss.remove(idx);
        }
        write_settings(&path, &config).unwrap();

        // Verify uninstalled
        let config = read_settings(&path).unwrap();
        let ss = config["hooks"]["SessionStart"].as_array().unwrap();
        assert!(!ss.iter().any(|e| group_has_wqm_command(e)));

        // PostToolUse still preserved
        assert!(config["hooks"]["PostToolUse"].is_array());
        assert!(!config["hooks"]["PostToolUse"].as_array().unwrap().is_empty());
    }
}
