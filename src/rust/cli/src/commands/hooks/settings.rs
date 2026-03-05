//! Settings file I/O for Claude Code hooks management

use anyhow::{Context, Result};
use serde_json::json;
use std::path::{Path, PathBuf};

/// Get the path to Claude Code's settings.json
pub(super) fn get_claude_settings_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".claude").join("settings.json"))
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
    use tempfile::TempDir;

    fn create_test_settings(dir: &Path, content: &str) -> PathBuf {
        let settings_path = dir.join("settings.json");
        std::fs::write(&settings_path, content).unwrap();
        settings_path
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
