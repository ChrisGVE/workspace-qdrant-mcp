//! PATH Scanning and Binary Discovery
//!
//! Handles automatic discovery of LSP server executables through PATH scanning,
//! version detection, and executable validation.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use tracing::{debug, info, warn};
use which::which;

use super::registry::build_known_servers;
use super::{DetectedServer, ServerTemplate};
use crate::lsp::{Language, LspResult};

/// LSP server detector that scans the system for available servers
pub struct LspServerDetector {
    /// Known LSP server configurations
    pub(crate) known_servers: HashMap<&'static str, ServerTemplate>,
}

impl LspServerDetector {
    /// Create a new LSP server detector with the built-in server catalog
    pub fn new() -> Self {
        Self {
            known_servers: build_known_servers(),
        }
    }

    /// Detect all available LSP servers on the system
    pub async fn detect_servers(&self) -> LspResult<Vec<DetectedServer>> {
        info!("Starting LSP server detection");
        let mut detected = Vec::new();

        for (name, template) in &self.known_servers {
            debug!("Looking for LSP server: {}", name);

            match self.detect_server(name, template).await {
                Ok(Some(server)) => {
                    info!(
                        "Detected LSP server: {} at {}",
                        server.name,
                        server.path.display()
                    );
                    detected.push(server);
                }
                Ok(None) => {
                    debug!("LSP server not found: {}", name);
                }
                Err(e) => {
                    warn!("Error detecting LSP server {}: {}", name, e);
                }
            }
        }

        // Sort by priority and language coverage
        detected.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| b.languages.len().cmp(&a.languages.len()))
        });

        info!("Detected {} LSP servers", detected.len());
        Ok(detected)
    }

    /// Detect a specific LSP server by looking it up in PATH
    async fn detect_server(
        &self,
        name: &str,
        template: &ServerTemplate,
    ) -> LspResult<Option<DetectedServer>> {
        // Try to find the executable in PATH
        let path = match which(template.executable) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };

        // Verify the executable is actually executable
        if !is_executable(&path).await? {
            return Ok(None);
        }

        // Try to get version information
        let version = get_server_version(&path, template.version_args).await;

        let detected = DetectedServer {
            name: name.to_string(),
            path,
            languages: template.languages.to_vec(),
            version,
            capabilities: template.capabilities.clone(),
            priority: template.priority,
        };

        Ok(Some(detected))
    }

    /// Get servers for a specific language, sorted by priority
    pub fn get_servers_for_language(&self, language: &Language) -> Vec<&str> {
        let mut servers: Vec<(&str, u8)> = self
            .known_servers
            .iter()
            .filter_map(|(name, template)| {
                if template.languages.contains(language) {
                    Some((name.as_ref(), template.priority))
                } else {
                    None
                }
            })
            .collect();

        servers.sort_by_key(|(_, priority)| *priority);
        servers.into_iter().map(|(name, _)| name).collect()
    }

    /// Check if a specific server is known
    pub fn is_known_server(&self, name: &str) -> bool {
        self.known_servers.contains_key(name)
    }
}

impl Default for LspServerDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a path is an executable file
async fn is_executable(path: &Path) -> LspResult<bool> {
    let metadata = tokio::fs::metadata(path).await?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = metadata.permissions();
        Ok(metadata.is_file() && (permissions.mode() & 0o111) != 0)
    }

    #[cfg(windows)]
    {
        Ok(metadata.is_file())
    }
}

/// Try to get version information from an LSP server executable.
///
/// Uses `tokio::process::Command` with a 5-second timeout to prevent hanging
/// on servers that start in stdio mode instead of printing version and exiting.
async fn get_server_version(path: &Path, version_args: &[&str]) -> Option<String> {
    let child = tokio::process::Command::new(path)
        .args(version_args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .spawn()
        .ok()?;

    let output = match tokio::time::timeout(Duration::from_secs(5), child.wait_with_output()).await
    {
        Ok(Ok(output)) => output,
        Ok(Err(_)) | Err(_) => {
            debug!("Timeout or error getting version from {}", path.display());
            return None;
        }
    };

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Try stdout first, then stderr
        let version_text = if !stdout.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        };

        if !version_text.is_empty() {
            Some(extract_version_from_text(version_text))
        } else {
            None
        }
    } else {
        None
    }
}

/// Extract version number from version output text using common patterns
pub(crate) fn extract_version_from_text(text: &str) -> String {
    // Look for common version patterns
    let version_patterns = [
        regex::Regex::new(r"(\d+\.\d+(?:\.\d+)?)").unwrap(),
        regex::Regex::new(r"version\s+(\d+\.\d+(?:\.\d+)?)").unwrap(),
        regex::Regex::new(r"v(\d+\.\d+(?:\.\d+)?)").unwrap(),
    ];

    for pattern in &version_patterns {
        if let Some(captures) = pattern.captures(text) {
            if let Some(version) = captures.get(1) {
                return version.as_str().to_string();
            }
        }
    }

    // If no pattern matches, return the first line cleaned up
    text.lines()
        .next()
        .unwrap_or(text)
        .trim()
        .to_string()
}
