//! Status subcommand handler for Claude Code hooks

use anyhow::Result;

use crate::output;

use super::matchers::{group_has_wqm_command, is_our_matcher, SESSION_START_MATCHER};
use super::settings::{get_claude_settings_path, read_settings};

/// Check if hooks are installed
pub(super) async fn status_hooks() -> Result<()> {
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
