//! Hook matching helpers for Claude Code settings

use serde_json::json;

/// The matcher pattern that triggers the hook
pub(super) const SESSION_START_MATCHER: &str = "startup|clear|compact";

/// Build the wqm hook command entry
pub(super) fn build_wqm_hook_command() -> serde_json::Value {
    json!({
        "type": "command",
        "command": "wqm rules inject"
    })
}

/// Check if a hook command entry is a wqm command
pub(super) fn is_wqm_hook_command(hook: &serde_json::Value) -> bool {
    hook.get("command")
        .and_then(|c| c.as_str())
        .map(|c| c.starts_with("wqm "))
        .unwrap_or(false)
}

/// Check if a matcher group has our matcher pattern
pub(super) fn is_our_matcher(entry: &serde_json::Value) -> bool {
    entry
        .get("matcher")
        .and_then(|m| m.as_str())
        .map(|m| m == SESSION_START_MATCHER)
        .unwrap_or(false)
}

/// Check if a matcher group's hooks array contains a wqm command
pub(super) fn group_has_wqm_command(entry: &serde_json::Value) -> bool {
    entry
        .get("hooks")
        .and_then(|h| h.as_array())
        .map(|hooks| hooks.iter().any(is_wqm_hook_command))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_build_wqm_hook_command() {
        let cmd = build_wqm_hook_command();
        assert_eq!(cmd["type"].as_str().unwrap(), "command");
        assert_eq!(cmd["command"].as_str().unwrap(), "wqm rules inject");
    }

    #[test]
    fn test_is_wqm_hook_command_true() {
        let cmd = json!({"type": "command", "command": "wqm rules inject"});
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
}
