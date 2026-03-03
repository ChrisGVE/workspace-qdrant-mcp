//! Install subcommand handler for Claude Code hooks

use anyhow::{Context, Result};
use serde_json::json;

use crate::output;

use super::matchers::{
    SESSION_START_MATCHER, build_wqm_hook_command, group_has_wqm_command, is_our_matcher,
};
use super::settings::{get_claude_settings_path, read_settings, write_settings};

/// Install Claude Code SessionStart hook for rule injection
pub(super) async fn install_hooks() -> Result<()> {
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
    output::kv("Command", "wqm rules inject");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::matchers::{SESSION_START_MATCHER, build_wqm_hook_command, is_our_matcher, group_has_wqm_command};
    use super::super::settings::{read_settings, write_settings};
    use std::path::Path;
    use tempfile::TempDir;

    fn create_test_settings(dir: &Path, content: &str) -> std::path::PathBuf {
        let settings_path = dir.join("settings.json");
        std::fs::write(&settings_path, content).unwrap();
        settings_path
    }

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
        assert_eq!(hooks[1]["command"].as_str().unwrap(), "wqm rules inject");
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
}
