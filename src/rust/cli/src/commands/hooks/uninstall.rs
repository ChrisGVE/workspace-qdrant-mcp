//! Uninstall subcommand handler for Claude Code hooks

use anyhow::{Context, Result};

use crate::output;

use super::matchers::is_wqm_hook_command;
use super::settings::{get_claude_settings_path, read_settings, write_settings};

/// Remove wqm hooks from Claude Code settings
pub(super) async fn uninstall_hooks() -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::super::matchers::{
        build_wqm_hook_command, group_has_wqm_command, SESSION_START_MATCHER,
    };
    use super::super::settings::{read_settings, write_settings};
    use super::*;
    use serde_json::json;
    use std::path::Path;
    use tempfile::TempDir;

    fn create_test_settings(dir: &Path, content: &str) -> std::path::PathBuf {
        let settings_path = dir.join("settings.json");
        std::fs::write(&settings_path, content).unwrap();
        settings_path
    }

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

    #[tokio::test]
    async fn test_install_uninstall_cycle() {
        use super::super::matchers::is_our_matcher;

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
        assert!(ss
            .iter()
            .any(|e| is_our_matcher(e) && group_has_wqm_command(e)));

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
        assert!(!config["hooks"]["PostToolUse"]
            .as_array()
            .unwrap()
            .is_empty());
    }
}
