//! Environment variable expansion utilities
//!
//! Provides functions for expanding environment variables in strings and paths.
//! Used by both daemon and CLI for consistent path resolution.

use std::path::PathBuf;

/// Expand environment variables in a string.
///
/// Supports both `${VAR}` and `$VAR` syntax. If a variable is not set,
/// the reference is left unchanged (literal text preserved).
///
/// # Examples
/// ```
/// use wqm_common::env_expand::expand_env_vars;
///
/// std::env::set_var("WQM_TEST_EXPAND", "/test/path");
/// assert_eq!(expand_env_vars("$WQM_TEST_EXPAND/cache"), "/test/path/cache");
/// assert_eq!(expand_env_vars("${WQM_TEST_EXPAND}/cache"), "/test/path/cache");
/// std::env::remove_var("WQM_TEST_EXPAND");
/// ```
pub fn expand_env_vars(s: &str) -> String {
    shellexpand::env_with_context_no_errors(s, |var| std::env::var(var).ok()).to_string()
}

/// Expand environment variables in an optional path.
///
/// Returns `None` if input is `None`, otherwise expands environment variables
/// in the path string and returns a new `PathBuf`.
pub fn expand_path_env_vars(path: Option<&PathBuf>) -> Option<PathBuf> {
    path.map(|p| {
        let expanded = expand_env_vars(&p.to_string_lossy());
        PathBuf::from(expanded)
    })
}

/// Expand a path string applying tilde and environment variable expansion.
///
/// Applies, in order:
/// 1. Tilde expansion: `~` → home directory, `~user` → that user's home
/// 2. Environment variable expansion: `$VAR`, `${VAR}` → variable value
///
/// Unset variables are left unchanged (not stripped). Command substitution
/// `$(...)` is a shell-level feature and is not expanded here — use double
/// quotes in the shell so the shell expands it before passing to the binary.
///
/// This is the canonical path resolver for all user-supplied path arguments
/// in the wqm CLI, enabling scripts to pass `$HOME/path` or `~/path` without
/// requiring the calling shell to pre-expand them.
pub fn expand_path(s: &str) -> PathBuf {
    let after_tilde = shellexpand::tilde(s);
    let after_env =
        shellexpand::env_with_context_no_errors(&after_tilde, |var| std::env::var(var).ok());
    PathBuf::from(after_env.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_env_vars_braces() {
        std::env::set_var("WQM_COMMON_TEST_VAR", "/test/path");
        assert_eq!(
            expand_env_vars("${WQM_COMMON_TEST_VAR}/cache"),
            "/test/path/cache"
        );
        std::env::remove_var("WQM_COMMON_TEST_VAR");
    }

    #[test]
    fn test_expand_env_vars_dollar() {
        std::env::set_var("WQM_COMMON_TEST_VAR2", "/test/path");
        assert_eq!(
            expand_env_vars("$WQM_COMMON_TEST_VAR2/cache"),
            "/test/path/cache"
        );
        std::env::remove_var("WQM_COMMON_TEST_VAR2");
    }

    #[test]
    fn test_expand_env_vars_unset() {
        let result = expand_env_vars("$WQM_COMMON_NONEXISTENT_VAR/path");
        assert!(result.contains("WQM_COMMON_NONEXISTENT_VAR"));
    }

    #[test]
    fn test_expand_env_vars_no_vars() {
        assert_eq!(expand_env_vars("/static/path"), "/static/path");
    }

    #[test]
    fn test_expand_env_vars_multiple() {
        std::env::set_var("WQM_COMMON_A", "/a");
        std::env::set_var("WQM_COMMON_B", "b");
        assert_eq!(expand_env_vars("${WQM_COMMON_A}/${WQM_COMMON_B}"), "/a/b");
        std::env::remove_var("WQM_COMMON_A");
        std::env::remove_var("WQM_COMMON_B");
    }

    #[test]
    fn test_expand_path_env_vars_some() {
        std::env::set_var("WQM_COMMON_HOME", "/home/testuser");
        let path = PathBuf::from("${WQM_COMMON_HOME}/.cache/models");
        let expanded = expand_path_env_vars(Some(&path));
        assert_eq!(
            expanded.unwrap(),
            PathBuf::from("/home/testuser/.cache/models")
        );
        std::env::remove_var("WQM_COMMON_HOME");
    }

    #[test]
    fn test_expand_path_env_vars_none() {
        assert!(expand_path_env_vars(None).is_none());
    }

    #[test]
    fn test_expand_path_tilde() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let result = expand_path("~/docs");
        assert_eq!(result, PathBuf::from(format!("{}/docs", home)));
    }

    #[test]
    fn test_expand_path_env_var() {
        std::env::set_var("WQM_EXPAND_PATH_TEST", "/tmp/wqm");
        let result = expand_path("$WQM_EXPAND_PATH_TEST/data");
        assert_eq!(result, PathBuf::from("/tmp/wqm/data"));
        std::env::remove_var("WQM_EXPAND_PATH_TEST");
    }

    #[test]
    fn test_expand_path_braces() {
        std::env::set_var("WQM_EXPAND_PATH_TEST2", "/tmp/wqm2");
        let result = expand_path("${WQM_EXPAND_PATH_TEST2}/data");
        assert_eq!(result, PathBuf::from("/tmp/wqm2/data"));
        std::env::remove_var("WQM_EXPAND_PATH_TEST2");
    }

    #[test]
    fn test_expand_path_tilde_and_env() {
        std::env::set_var("WQM_EXPAND_SUBDIR", "projects");
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let result = expand_path("~/$WQM_EXPAND_SUBDIR");
        assert_eq!(result, PathBuf::from(format!("{}/projects", home)));
        std::env::remove_var("WQM_EXPAND_SUBDIR");
    }

    #[test]
    fn test_expand_path_absolute_unchanged() {
        let result = expand_path("/usr/local/bin");
        assert_eq!(result, PathBuf::from("/usr/local/bin"));
    }

    #[test]
    fn test_expand_path_unset_var_preserved() {
        let result = expand_path("$WQM_EXPAND_DEFINITELY_NOT_SET/path");
        // unset variables are left unchanged (not stripped)
        let s = result.to_string_lossy();
        assert!(s.contains("WQM_EXPAND_DEFINITELY_NOT_SET"));
    }
}
