//! Config file format detection and parsing into a neutral `serde_json::Value`.

use std::path::Path;

use serde_json::Value;

use super::ConfigError;

/// The on-disk format of a config file.
///
/// Components carry different formats (daemon + MCP use YAML, CLI uses TOML);
/// the shared machinery treats them uniformly by parsing into a
/// [`serde_json::Value`] before merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// YAML (`.yaml` / `.yml`).
    Yaml,
    /// TOML (`.toml`).
    Toml,
}

impl ConfigFormat {
    /// Detect the format from a file extension, case-insensitively.
    ///
    /// Returns `None` for unknown or missing extensions.
    pub fn detect_from_path(path: &Path) -> Option<ConfigFormat> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "yaml" | "yml" => Some(ConfigFormat::Yaml),
            "toml" => Some(ConfigFormat::Toml),
            _ => None,
        }
    }

    /// Parse `content` in this format into a neutral [`Value`].
    ///
    /// TOML is parsed via `toml::Value` then converted to `serde_json::Value`
    /// so the downstream merge primitive is format-agnostic.
    pub fn parse_to_value(&self, content: &str) -> Result<Value, ConfigError> {
        match self {
            ConfigFormat::Yaml => {
                serde_yaml_ng::from_str(content).map_err(|e| ConfigError::Parse(e.to_string()))
            }
            ConfigFormat::Toml => {
                let toml_val: toml::Value =
                    toml::from_str(content).map_err(|e| ConfigError::Parse(e.to_string()))?;
                serde_json::to_value(toml_val).map_err(|e| ConfigError::Parse(e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn detects_yaml_variants() {
        assert_eq!(
            ConfigFormat::detect_from_path(&PathBuf::from("config.yaml")),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::detect_from_path(&PathBuf::from("config.yml")),
            Some(ConfigFormat::Yaml)
        );
        // Case-insensitive.
        assert_eq!(
            ConfigFormat::detect_from_path(&PathBuf::from("Config.YAML")),
            Some(ConfigFormat::Yaml)
        );
    }

    #[test]
    fn detects_toml() {
        assert_eq!(
            ConfigFormat::detect_from_path(&PathBuf::from("/etc/app/config.toml")),
            Some(ConfigFormat::Toml)
        );
    }

    #[test]
    fn unknown_or_missing_extension_is_none() {
        assert_eq!(
            ConfigFormat::detect_from_path(&PathBuf::from("config.json")),
            None
        );
        assert_eq!(
            ConfigFormat::detect_from_path(&PathBuf::from("config")),
            None
        );
    }

    #[test]
    fn parses_yaml_to_value() {
        let v = ConfigFormat::Yaml
            .parse_to_value("qdrant:\n  url: \"http://h:6333\"\n  timeout: 42\n")
            .expect("parse");
        assert_eq!(v["qdrant"]["url"], "http://h:6333");
        assert_eq!(v["qdrant"]["timeout"], 42);
    }

    #[test]
    fn parses_toml_to_value() {
        let v = ConfigFormat::Toml
            .parse_to_value("[qdrant]\nurl = \"http://h:6333\"\ntimeout = 42\n")
            .expect("parse");
        assert_eq!(v["qdrant"]["url"], "http://h:6333");
        assert_eq!(v["qdrant"]["timeout"], 42);
    }

    #[test]
    fn invalid_yaml_errors() {
        assert!(ConfigFormat::Yaml
            .parse_to_value(": invalid: yaml: {{{")
            .is_err());
    }

    #[test]
    fn invalid_toml_errors() {
        assert!(ConfigFormat::Toml.parse_to_value("= = =").is_err());
    }
}
