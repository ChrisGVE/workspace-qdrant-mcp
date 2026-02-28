//! PATH environment management for the CLI.
//!
//! Handles reading/writing user PATH from config, expanding environment variables,
//! and merging PATH segments on CLI invocation.

use std::env;
use std::fs;
use std::io::Write;

use super::get_config_file_path;
use wqm_common::paths::get_config_dir;

/// Read user PATH from config file
///
/// Returns None if the config file doesn't exist or doesn't have user_path set.
pub fn read_user_path() -> Option<String> {
    let config_path = get_config_file_path().ok()?;

    if !config_path.exists() {
        return None;
    }

    let content = fs::read_to_string(&config_path).ok()?;

    // Simple YAML parsing for environment.user_path
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("user_path:") {
            let value = trimmed.strip_prefix("user_path:")?.trim();
            // Handle quoted strings
            let value = value.trim_matches('"').trim_matches('\'');
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }

    None
}

/// Write user PATH to config file
///
/// Creates the config directory and file if they don't exist.
/// Updates the environment.user_path value if the file exists.
pub fn write_user_path(path: &str) -> Result<(), String> {
    let config_dir = get_config_dir()
        .map_err(|e| format!("Failed to get config directory: {}", e))?;

    // Create config directory if it doesn't exist
    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
    }

    let config_path = config_dir.join("config.yaml");

    // Read existing config or create new
    let mut config_content = if config_path.exists() {
        fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config file: {}", e))?
    } else {
        String::new()
    };

    // Check if environment section exists
    let has_environment = config_content.contains("\nenvironment:") || config_content.starts_with("environment:");
    let has_user_path = config_content.contains("user_path:");

    if has_user_path {
        // Update existing user_path line
        let mut new_content = String::new();
        for line in config_content.lines() {
            if line.trim().starts_with("user_path:") {
                new_content.push_str(&format!("  user_path: \"{}\"\n", path));
            } else {
                new_content.push_str(line);
                new_content.push('\n');
            }
        }
        config_content = new_content;
    } else if has_environment {
        // Add user_path under existing environment section
        let mut new_content = String::new();
        for line in config_content.lines() {
            new_content.push_str(line);
            new_content.push('\n');
            if line.trim() == "environment:" {
                new_content.push_str(&format!("  user_path: \"{}\"\n", path));
            }
        }
        config_content = new_content;
    } else {
        // Add new environment section
        if !config_content.is_empty() && !config_content.ends_with('\n') {
            config_content.push('\n');
        }
        config_content.push_str(&format!("\nenvironment:\n  user_path: \"{}\"\n", path));
    }

    // Write config file
    let mut file = fs::File::create(&config_path)
        .map_err(|e| format!("Failed to create config file: {}", e))?;
    file.write_all(config_content.as_bytes())
        .map_err(|e| format!("Failed to write config file: {}", e))?;

    Ok(())
}

/// Get current system PATH
pub fn get_current_path() -> String {
    env::var("PATH").unwrap_or_default()
}

/// Platform-specific PATH separator
#[cfg(not(target_os = "windows"))]
pub const PATH_SEPARATOR: char = ':';
#[cfg(target_os = "windows")]
pub const PATH_SEPARATOR: char = ';';

/// Expand a single path segment, resolving `~` and environment variables.
///
/// Handles:
/// - `~` or `~/...` → home directory
/// - `$VAR` or `${VAR}` → environment variable value
/// - Recursive expansion up to a depth limit to prevent infinite loops
pub(super) fn expand_path_segment(segment: &str) -> String {
    expand_path_segment_recursive(segment, 0)
}

fn expand_path_segment_recursive(segment: &str, depth: u8) -> String {
    if depth > 10 || segment.is_empty() {
        return segment.to_string();
    }

    let mut result = segment.to_string();

    // Expand ~ at start
    if result == "~" || result.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            result = if result == "~" {
                home.to_string_lossy().to_string()
            } else {
                format!("{}{}", home.display(), &result[1..])
            };
        }
    }

    // Expand ${VAR} patterns
    let mut expanded = String::with_capacity(result.len());
    let chars: Vec<char> = result.chars().collect();
    let mut i = 0;
    let mut changed = false;

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() {
            if chars[i + 1] == '{' {
                // ${VAR} form
                if let Some(close) = chars[i + 2..].iter().position(|&c| c == '}') {
                    let var_name: String = chars[i + 2..i + 2 + close].iter().collect();
                    if let Ok(val) = env::var(&var_name) {
                        expanded.push_str(&val);
                        changed = true;
                    }
                    // Skip past the closing brace
                    i = i + 2 + close + 1;
                    continue;
                }
            } else if chars[i + 1].is_ascii_alphanumeric() || chars[i + 1] == '_' {
                // $VAR form - collect alphanumeric + underscore
                let start = i + 1;
                let mut end = start;
                while end < chars.len()
                    && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }
                let var_name: String = chars[start..end].iter().collect();
                if let Ok(val) = env::var(&var_name) {
                    expanded.push_str(&val);
                    changed = true;
                }
                i = end;
                continue;
            }
        }
        expanded.push(chars[i]);
        i += 1;
    }

    // Recurse if we made substitutions (to handle nested vars)
    if changed {
        expand_path_segment_recursive(&expanded, depth + 1)
    } else {
        expanded
    }
}

/// Expand all segments in a PATH string.
///
/// Splits by platform separator, expands each segment, and returns the
/// expanded segments.
pub fn expand_path_segments(path: &str) -> Vec<String> {
    path.split(PATH_SEPARATOR)
        .filter(|s| !s.is_empty())
        .map(|s| expand_path_segment(s))
        .collect()
}

/// Merge and deduplicate PATH segments.
///
/// Combines current PATH with saved user_path, keeping first occurrence
/// of each entry. Current PATH entries take precedence.
pub(super) fn merge_and_dedup(current_segments: &[String], saved_segments: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();

    // Current PATH first (higher precedence)
    for seg in current_segments {
        if !seg.is_empty() && seen.insert(seg.clone()) {
            merged.push(seg.clone());
        }
    }

    // Then saved user_path
    for seg in saved_segments {
        if !seg.is_empty() && seen.insert(seg.clone()) {
            merged.push(seg.clone());
        }
    }

    merged
}

/// Join PATH segments with platform separator.
pub(super) fn join_path_segments(segments: &[String]) -> String {
    segments.join(&PATH_SEPARATOR.to_string())
}

/// Set up the environment PATH on CLI invocation.
///
/// Per specification:
/// 1. Expand: Retrieve $PATH and expand all env vars recursively
/// 2. Merge: Append existing user_path from config to expanded $PATH
/// 3. Deduplicate: Remove duplicates, keeping first occurrence
/// 4. Save: Write to config only if different from current value
///
/// Returns Ok(true) if user_path was updated, Ok(false) if unchanged.
pub fn setup_environment_path() -> Result<bool, String> {
    // Step 1: Expand current $PATH
    let current_path = get_current_path();
    let current_segments = expand_path_segments(&current_path);

    // Step 2: Read saved user_path and expand it too
    let saved_path = read_user_path().unwrap_or_default();
    let saved_segments = expand_path_segments(&saved_path);

    // Step 3: Merge and deduplicate
    let merged = merge_and_dedup(&current_segments, &saved_segments);
    let new_user_path = join_path_segments(&merged);

    // Step 4: Save only if different
    if new_user_path == saved_path {
        return Ok(false);
    }

    write_user_path(&new_user_path)?;
    Ok(true)
}

/// Capture and store current PATH to config (legacy compatibility).
///
/// Returns true if PATH was captured, false if it was already stored.
pub fn capture_user_path() -> Result<bool, String> {
    setup_environment_path()
}
