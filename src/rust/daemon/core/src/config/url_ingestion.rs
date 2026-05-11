//! URL ingestion configuration (SSRF policy, fetch limits, content-type allowlist).
//!
//! Default values are restrictive:
//!   * connect timeout 15s, read timeout 60s
//!   * max 5 redirects
//!   * 10 MiB body cap
//!   * allowed content types: text/*, application/json, application/xhtml+xml,
//!     application/xml
//!   * private networks denied (SSRF protection on by default)

use serde::{Deserialize, Serialize};

/// URL ingestion limits and security configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UrlIngestionConfig {
    /// TCP connect timeout in seconds.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,

    /// Whole-request read timeout in seconds.
    #[serde(default = "default_read_timeout_secs")]
    pub read_timeout_secs: u64,

    /// Maximum number of redirects to follow. Each hop is re-validated
    /// against the SSRF policy.
    #[serde(default = "default_max_redirects")]
    pub max_redirects: usize,

    /// Maximum response body size in bytes. Bodies larger than this are
    /// truncated mid-stream; the truncated content is still ingested.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: u64,

    /// Allow private / loopback / link-local network targets. Disabled by
    /// default. Enable only for trusted, contained environments.
    #[serde(default)]
    pub allow_private_networks: bool,

    /// Content-Type prefixes that are accepted. Matching is case-insensitive
    /// and prefix-based (e.g. "text/" matches "text/html; charset=utf-8").
    #[serde(default = "default_allowed_content_types")]
    pub allowed_content_types: Vec<String>,
}

fn default_connect_timeout_secs() -> u64 {
    15
}
fn default_read_timeout_secs() -> u64 {
    60
}
fn default_max_redirects() -> usize {
    5
}
fn default_max_body_bytes() -> u64 {
    10 * 1024 * 1024
}

fn default_allowed_content_types() -> Vec<String> {
    vec![
        "text/".to_string(),
        "application/json".to_string(),
        "application/xhtml+xml".to_string(),
        "application/xml".to_string(),
    ]
}

impl Default for UrlIngestionConfig {
    fn default() -> Self {
        Self {
            connect_timeout_secs: default_connect_timeout_secs(),
            read_timeout_secs: default_read_timeout_secs(),
            max_redirects: default_max_redirects(),
            max_body_bytes: default_max_body_bytes(),
            allow_private_networks: false,
            allowed_content_types: default_allowed_content_types(),
        }
    }
}

impl UrlIngestionConfig {
    /// True iff the given Content-Type header matches an allowed prefix.
    pub fn is_content_type_allowed(&self, content_type: &str) -> bool {
        let lower = content_type.trim().to_ascii_lowercase();
        self.allowed_content_types
            .iter()
            .any(|prefix| lower.starts_with(&prefix.to_ascii_lowercase()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_are_restrictive() {
        let c = UrlIngestionConfig::default();
        assert_eq!(c.connect_timeout_secs, 15);
        assert_eq!(c.read_timeout_secs, 60);
        assert_eq!(c.max_redirects, 5);
        assert_eq!(c.max_body_bytes, 10 * 1024 * 1024);
        assert!(!c.allow_private_networks);
    }

    #[test]
    fn test_content_type_text_html_allowed() {
        let c = UrlIngestionConfig::default();
        assert!(c.is_content_type_allowed("text/html"));
        assert!(c.is_content_type_allowed("text/html; charset=utf-8"));
        assert!(c.is_content_type_allowed("text/plain"));
        assert!(c.is_content_type_allowed("TEXT/HTML"));
    }

    #[test]
    fn test_content_type_application_json_allowed() {
        let c = UrlIngestionConfig::default();
        assert!(c.is_content_type_allowed("application/json"));
        assert!(c.is_content_type_allowed("application/json; charset=utf-8"));
    }

    #[test]
    fn test_content_type_xml_allowed() {
        let c = UrlIngestionConfig::default();
        assert!(c.is_content_type_allowed("application/xml"));
        assert!(c.is_content_type_allowed("application/xhtml+xml"));
    }

    #[test]
    fn test_content_type_binary_rejected() {
        let c = UrlIngestionConfig::default();
        assert!(!c.is_content_type_allowed("application/octet-stream"));
        assert!(!c.is_content_type_allowed("image/png"));
        assert!(!c.is_content_type_allowed("application/pdf"));
        assert!(!c.is_content_type_allowed("video/mp4"));
    }

    #[test]
    fn test_yaml_roundtrip() {
        let yaml = r#"
connect_timeout_secs: 30
read_timeout_secs: 120
max_redirects: 3
max_body_bytes: 5242880
allow_private_networks: true
allowed_content_types:
  - "text/"
  - "application/json"
"#;
        let c: UrlIngestionConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(c.connect_timeout_secs, 30);
        assert_eq!(c.read_timeout_secs, 120);
        assert_eq!(c.max_redirects, 3);
        assert_eq!(c.max_body_bytes, 5 * 1024 * 1024);
        assert!(c.allow_private_networks);
        assert_eq!(c.allowed_content_types.len(), 2);
    }

    #[test]
    fn test_empty_yaml_uses_defaults() {
        let c: UrlIngestionConfig = serde_yaml_ng::from_str("{}").unwrap();
        assert_eq!(c, UrlIngestionConfig::default());
    }
}
