//! Status subcommand handler for Claude Code hooks

use anyhow::Result;

use crate::output;

use super::matchers::{group_has_wqm_command, is_our_matcher, SESSION_START_MATCHER};
use super::settings::{config_source_label, get_claude_settings_path, read_settings};

/// Check if hooks are installed.
///
/// Respects `CLAUDE_CONFIG_DIR` via [`get_claude_settings_path`] so the
/// reported status matches whichever Claude Code installation the user
/// has active.
pub(super) async fn status_hooks() -> Result<()> {
    let settings_path = get_claude_settings_path()?;
    output::kv("Config source", config_source_label());
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
        .map(|arr| {
            arr.iter()
                .any(|e| is_our_matcher(e) && group_has_wqm_command(e))
        })
        .unwrap_or(false);

    if installed {
        output::success("wqm hooks are installed");
        output::kv("Hook event", "SessionStart");
        output::kv("Matcher", SESSION_START_MATCHER);
        output::kv("Command", "wqm rules inject");
    } else {
        output::kv("Status", "Not installed");
        output::info("Run 'wqm hooks install' to set up rule injection");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::matchers::{
        build_wqm_hook_command, group_has_wqm_command, is_our_matcher, SESSION_START_MATCHER,
    };
    use super::super::settings::{get_claude_settings_path, read_settings};
    use serde_json::json;
    use serial_test::serial;
    use tempfile::TempDir;

    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("CLAUDE_CONFIG_DIR");
        }
    }

    #[test]
    #[serial]
    fn test_status_respects_claude_config_dir() {
        let _g = EnvGuard;
        let dir = TempDir::new().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", dir.path());

        let settings_path = get_claude_settings_path().unwrap();
        assert_eq!(settings_path, dir.path().join("settings.json"));

        std::fs::write(
            &settings_path,
            serde_json::to_string(&json!({
                "hooks": {
                    "SessionStart": [{
                        "matcher": SESSION_START_MATCHER,
                        "hooks": [build_wqm_hook_command()]
                    }]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let config = read_settings(&settings_path).unwrap();
        let installed = config
            .get("hooks")
            .and_then(|h| h.get("SessionStart"))
            .and_then(|arr| arr.as_array())
            .map(|arr| {
                arr.iter()
                    .any(|e| is_our_matcher(e) && group_has_wqm_command(e))
            })
            .unwrap_or(false);
        assert!(installed, "status should detect hook in custom dir");
    }

    #[test]
    #[serial]
    fn test_status_custom_dir_no_settings() {
        let _g = EnvGuard;
        let dir = TempDir::new().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", dir.path());

        let settings_path = get_claude_settings_path().unwrap();
        assert!(!settings_path.exists(), "settings.json absent in empty dir");
    }
}
