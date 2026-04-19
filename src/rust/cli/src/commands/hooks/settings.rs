//! Settings file I/O for Claude Code hooks management

use anyhow::{Context, Result};
use serde_json::json;
use std::path::{Path, PathBuf};

/// Get the Claude Code config directory.
///
/// Respects the `CLAUDE_CONFIG_DIR` environment variable (used by Claude Code
/// Enterprise and custom installations). Falls back to `~/.claude` when the
/// env var is unset or contains only whitespace.
pub(super) fn get_claude_config_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        let s = dir.to_string_lossy();
        if !s.trim().is_empty() {
            return Ok(PathBuf::from(s.as_ref()));
        }
    }
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".claude"))
}

/// Get the path to Claude Code's settings.json
pub(super) fn get_claude_settings_path() -> Result<PathBuf> {
    Ok(get_claude_config_dir()?.join("settings.json"))
}

/// Human-readable label describing which source supplied the config dir,
/// used in CLI output to make path resolution transparent for debugging.
pub(super) fn config_source_label() -> &'static str {
    match std::env::var_os("CLAUDE_CONFIG_DIR") {
        Some(v) if !v.to_string_lossy().trim().is_empty() => "CLAUDE_CONFIG_DIR",
        _ => "default (~/.claude)",
    }
}

/// Diagnostic hint for errors that target the resolved settings path.
///
/// Mentions the active source (env var vs. default) so users can tell
/// whether a bad `CLAUDE_CONFIG_DIR` caused the failure.
pub(super) fn config_source_hint() -> String {
    match std::env::var("CLAUDE_CONFIG_DIR") {
        Ok(v) if !v.trim().is_empty() => format!(
            "resolved via CLAUDE_CONFIG_DIR={} (verify the directory exists and is writable, or unset the variable to fall back to ~/.claude)",
            v
        ),
        _ => "resolved via default ~/.claude (set CLAUDE_CONFIG_DIR to target a custom Claude Code install)".to_string(),
    }
}

/// Read Claude Code settings.json, returning empty object if not found
pub(super) fn read_settings(path: &Path) -> Result<serde_json::Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse {} as JSON", path.display()))
}

/// Write Claude Code settings.json with backup
pub(super) fn write_settings(path: &Path, config: &serde_json::Value) -> Result<()> {
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
    let content = serde_json::to_string_pretty(config).context("Failed to serialize settings")?;
    std::fs::write(path, format!("{}\n", content))
        .with_context(|| format!("Failed to write {}", path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    fn create_test_settings(dir: &Path, content: &str) -> PathBuf {
        let settings_path = dir.join("settings.json");
        std::fs::write(&settings_path, content).unwrap();
        settings_path
    }

    /// Guard that removes CLAUDE_CONFIG_DIR on drop to keep tests hermetic.
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("CLAUDE_CONFIG_DIR");
        }
    }

    #[test]
    #[serial]
    fn test_settings_path_uses_claude_config_dir_when_set() {
        let _g = EnvGuard;
        let dir = TempDir::new().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", dir.path());

        let path = get_claude_settings_path().unwrap();
        assert_eq!(path, dir.path().join("settings.json"));
    }

    #[test]
    #[serial]
    fn test_settings_path_falls_back_to_home_when_unset() {
        let _g = EnvGuard;
        std::env::remove_var("CLAUDE_CONFIG_DIR");

        let path = get_claude_settings_path().unwrap();
        let home = dirs::home_dir().unwrap();
        assert_eq!(path, home.join(".claude").join("settings.json"));
    }

    #[test]
    #[serial]
    fn test_settings_path_falls_back_when_empty() {
        let _g = EnvGuard;
        std::env::set_var("CLAUDE_CONFIG_DIR", "");

        let path = get_claude_settings_path().unwrap();
        let home = dirs::home_dir().unwrap();
        assert_eq!(path, home.join(".claude").join("settings.json"));
    }

    #[test]
    #[serial]
    fn test_settings_path_falls_back_when_whitespace() {
        let _g = EnvGuard;
        std::env::set_var("CLAUDE_CONFIG_DIR", "   \t  ");

        let path = get_claude_settings_path().unwrap();
        let home = dirs::home_dir().unwrap();
        assert_eq!(path, home.join(".claude").join("settings.json"));
    }

    #[test]
    #[serial]
    fn test_config_dir_uses_claude_config_dir_when_set() {
        let _g = EnvGuard;
        let dir = TempDir::new().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", dir.path());

        let resolved = get_claude_config_dir().unwrap();
        assert_eq!(resolved, dir.path());
    }

    #[test]
    #[serial]
    fn test_config_source_label_reflects_env() {
        let _g = EnvGuard;
        std::env::remove_var("CLAUDE_CONFIG_DIR");
        assert_eq!(config_source_label(), "default (~/.claude)");
        std::env::set_var("CLAUDE_CONFIG_DIR", "/tmp/custom");
        assert_eq!(config_source_label(), "CLAUDE_CONFIG_DIR");
        std::env::set_var("CLAUDE_CONFIG_DIR", "   ");
        assert_eq!(config_source_label(), "default (~/.claude)");
    }

    #[test]
    #[serial]
    fn test_config_source_hint_mentions_env_when_set() {
        let _g = EnvGuard;
        std::env::set_var("CLAUDE_CONFIG_DIR", "/tmp/custom");
        let hint = config_source_hint();
        assert!(hint.contains("CLAUDE_CONFIG_DIR"));
        assert!(hint.contains("/tmp/custom"));
    }

    #[test]
    #[serial]
    fn test_config_source_hint_mentions_default_when_unset() {
        let _g = EnvGuard;
        std::env::remove_var("CLAUDE_CONFIG_DIR");
        let hint = config_source_hint();
        assert!(hint.contains("~/.claude"));
        assert!(hint.contains("CLAUDE_CONFIG_DIR"));
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
        let path =
            create_test_settings(dir.path(), r#"{"permissions": {"allow": []}, "hooks": {}}"#);
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
}
