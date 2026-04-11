//! Path formatting for CLI output.
//!
//! Replaces home directory with `~` and known XDG base directories with
//! their variable names for better readability. Paths are never truncated
//! or wrapped.

use std::env;

/// Well-known XDG base directory variables and their defaults relative to `$HOME`.
const XDG_VARS: &[(&str, &str)] = &[
    ("XDG_CONFIG_HOME", ".config"),
    ("XDG_DATA_HOME", ".local/share"),
    ("XDG_STATE_HOME", ".local/state"),
    ("XDG_CACHE_HOME", ".cache"),
    ("XDG_RUNTIME_DIR", ""),
];

/// Format a path for display, replacing home and XDG prefixes.
///
/// Priority order:
/// 1. Try XDG variable substitution (longer match wins)
/// 2. Fall back to `~` for home directory
/// 3. Return the path unchanged if no substitution applies
///
/// Per PRD: paths must never be truncated or wrapped.
pub fn format_path(path: &str) -> String {
    let home = dirs::home_dir()
        .map(|h| h.to_string_lossy().into_owned())
        .unwrap_or_default();

    if home.is_empty() {
        return path.to_string();
    }

    // Try XDG substitution — pick the longest matching prefix
    let mut best_match: Option<(&str, String)> = None;
    let mut best_len = 0;

    for &(var_name, default_suffix) in XDG_VARS {
        let resolved = if let Ok(val) = env::var(var_name) {
            val
        } else if !default_suffix.is_empty() {
            format!("{home}/{default_suffix}")
        } else {
            continue;
        };

        if path.starts_with(&resolved) && resolved.len() > best_len {
            let replacement = format!("${var_name}");
            best_len = resolved.len();
            best_match = Some((var_name, resolved));
            let _ = replacement; // used below
            let _ = var_name;
        }
    }

    if let Some((var_name, resolved)) = best_match {
        let remainder = &path[resolved.len()..];
        return format!("${var_name}{remainder}");
    }

    // Fall back to ~ substitution
    if path.starts_with(&home) {
        return format!("~{}", &path[home.len()..]);
    }

    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_replaced_with_tilde() {
        let home = dirs::home_dir().unwrap();
        let path = format!("{}/projects/foo", home.display());
        let result = format_path(&path);
        assert!(result.starts_with("~/"), "expected ~ prefix, got: {result}");
        assert_eq!(result, "~/projects/foo");
    }

    #[test]
    fn xdg_config_replaced() {
        let home = dirs::home_dir().unwrap();
        // XDG_CONFIG_HOME defaults to ~/.config
        let path = format!("{}/.config/nvim/init.lua", home.display());
        let result = format_path(&path);
        assert!(
            result.starts_with("$XDG_CONFIG_HOME"),
            "expected XDG_CONFIG_HOME prefix, got: {result}"
        );
        assert!(result.ends_with("/nvim/init.lua"));
    }

    #[test]
    fn xdg_data_replaced() {
        let home = dirs::home_dir().unwrap();
        let path = format!("{}/.local/share/wqm/db.sqlite", home.display());
        let result = format_path(&path);
        assert!(
            result.starts_with("$XDG_DATA_HOME"),
            "expected XDG_DATA_HOME prefix, got: {result}"
        );
    }

    #[test]
    fn non_home_path_unchanged() {
        let path = "/tmp/some/path";
        assert_eq!(format_path(path), path);
    }

    #[test]
    fn empty_path() {
        assert_eq!(format_path(""), "");
    }

    #[test]
    fn path_exactly_home() {
        let home = dirs::home_dir().unwrap();
        let result = format_path(&home.to_string_lossy());
        assert_eq!(result, "~");
    }
}
