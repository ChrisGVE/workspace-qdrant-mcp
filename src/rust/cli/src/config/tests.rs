use super::*;
use super::path_env::*;
use std::path::PathBuf;

#[test]
fn test_default_config() {
    let config = Config::default();
    assert!(config.daemon_address.contains("50051"));
    assert_eq!(config.connection_timeout_secs, 5);
    assert_eq!(config.output_format, OutputFormat::Table);
    assert!(config.color_enabled);
    assert!(!config.verbose);
}

#[test]
fn test_output_format_parsing() {
    assert_eq!(OutputFormat::from_str("table"), Some(OutputFormat::Table));
    assert_eq!(OutputFormat::from_str("TABLE"), Some(OutputFormat::Table));
    assert_eq!(OutputFormat::from_str("json"), Some(OutputFormat::Json));
    assert_eq!(OutputFormat::from_str("JSON"), Some(OutputFormat::Json));
    assert_eq!(OutputFormat::from_str("plain"), Some(OutputFormat::Plain));
    assert_eq!(OutputFormat::from_str("text"), Some(OutputFormat::Plain));
    assert_eq!(OutputFormat::from_str("invalid"), None);
}

#[test]
fn test_builder_pattern() {
    let config = Config::new()
        .with_daemon_address("http://localhost:9999")
        .with_timeout(10)
        .with_output_format(OutputFormat::Json)
        .with_color(false)
        .with_verbose(true);

    assert_eq!(config.daemon_address, "http://localhost:9999");
    assert_eq!(config.connection_timeout_secs, 10);
    assert_eq!(config.output_format, OutputFormat::Json);
    assert!(!config.color_enabled);
    assert!(config.verbose);
}

#[test]
fn test_validation_success() {
    let config = Config::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validation_invalid_address() {
    let config = Config::new().with_daemon_address("localhost:50051");
    assert!(config.validate().is_err());
}

#[test]
fn test_validation_invalid_timeout() {
    let config = Config::new().with_timeout(0);
    assert!(config.validate().is_err());

    let config = Config::new().with_timeout(500);
    assert!(config.validate().is_err());
}

#[test]
fn test_get_database_path() {
    // This test covers both default path and env override in a single test
    // to avoid race conditions when tests run in parallel (env vars are global).
    let prev = std::env::var("WQM_DATABASE_PATH").ok();

    // 1. Test default path (no env override)
    std::env::remove_var("WQM_DATABASE_PATH");
    let result = get_database_path();
    assert!(result.is_ok());
    let path = result.unwrap();
    assert!(path.to_string_lossy().contains(".workspace-qdrant"));
    assert!(path.to_string_lossy().ends_with("state.db"));

    // 2. Test environment variable override
    std::env::set_var("WQM_DATABASE_PATH", "/custom/path/state.db");
    let result = get_database_path();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), PathBuf::from("/custom/path/state.db"));

    // Restore previous state
    match prev {
        Some(val) => std::env::set_var("WQM_DATABASE_PATH", val),
        None => std::env::remove_var("WQM_DATABASE_PATH"),
    }
}

#[test]
fn test_database_path_error_display() {
    let error = DatabasePathError::NoHomeDirectory;
    assert_eq!(error.to_string(), "could not determine home directory");

    let error = DatabasePathError::DatabaseNotFound {
        path: PathBuf::from("/test/state.db"),
    };
    assert!(error.to_string().contains("run daemon first"));
}

// =========================================================================
// PATH expansion tests
// =========================================================================

#[test]
fn test_expand_path_segment_tilde() {
    let expanded = expand_path_segment("~/bin");
    if let Some(home) = dirs::home_dir() {
        assert_eq!(expanded, format!("{}/bin", home.display()));
    }
}

#[test]
fn test_expand_path_segment_tilde_alone() {
    let expanded = expand_path_segment("~");
    if let Some(home) = dirs::home_dir() {
        assert_eq!(expanded, home.to_string_lossy().to_string());
    }
}

#[test]
fn test_expand_path_segment_env_var_dollar() {
    // Use HOME which is always set on Unix
    let expanded = expand_path_segment("$HOME/bin");
    if let Ok(home) = std::env::var("HOME") {
        assert_eq!(expanded, format!("{}/bin", home));
    }
}

#[test]
fn test_expand_path_segment_env_var_braces() {
    let expanded = expand_path_segment("${HOME}/bin");
    if let Ok(home) = std::env::var("HOME") {
        assert_eq!(expanded, format!("{}/bin", home));
    }
}

#[test]
fn test_expand_path_segment_no_expansion() {
    let expanded = expand_path_segment("/usr/local/bin");
    assert_eq!(expanded, "/usr/local/bin");
}

#[test]
fn test_expand_path_segment_empty() {
    let expanded = expand_path_segment("");
    assert_eq!(expanded, "");
}

#[test]
fn test_expand_path_segment_unknown_var() {
    // Unknown vars should be removed (no value to substitute)
    let expanded =
        expand_path_segment("$WQM_TEST_NONEXISTENT_VAR_12345/bin");
    // The $VAR is consumed but env::var fails, so nothing is written
    assert_eq!(expanded, "/bin");
}

#[test]
fn test_expand_path_segment_recursive() {
    // Set up nested env vars for recursive expansion
    std::env::set_var("WQM_TEST_INNER", "/resolved");
    std::env::set_var("WQM_TEST_OUTER", "$WQM_TEST_INNER/path");
    let expanded = expand_path_segment("$WQM_TEST_OUTER");
    assert_eq!(expanded, "/resolved/path");
    std::env::remove_var("WQM_TEST_INNER");
    std::env::remove_var("WQM_TEST_OUTER");
}

#[test]
fn test_expand_path_segment_depth_limit() {
    // Set up a self-referencing var to test depth limit
    std::env::set_var("WQM_TEST_LOOP", "$WQM_TEST_LOOP");
    let expanded = expand_path_segment("$WQM_TEST_LOOP");
    // Should terminate without infinite recursion
    assert!(!expanded.is_empty() || expanded.is_empty()); // just prove it returns
    std::env::remove_var("WQM_TEST_LOOP");
}

// =========================================================================
// PATH segments expansion tests
// =========================================================================

#[test]
fn test_expand_path_segments_basic() {
    let segments = expand_path_segments("/usr/bin:/usr/local/bin");
    assert_eq!(segments, vec!["/usr/bin", "/usr/local/bin"]);
}

#[test]
fn test_expand_path_segments_with_tilde() {
    let segments = expand_path_segments("~/bin:/usr/local/bin");
    assert_eq!(segments.len(), 2);
    if let Some(home) = dirs::home_dir() {
        assert_eq!(segments[0], format!("{}/bin", home.display()));
    }
    assert_eq!(segments[1], "/usr/local/bin");
}

#[test]
fn test_expand_path_segments_empty() {
    let segments = expand_path_segments("");
    assert!(segments.is_empty());
}

#[test]
fn test_expand_path_segments_filters_empty() {
    // Double separator produces empty segments which should be filtered
    let segments = expand_path_segments("/usr/bin::/usr/local/bin");
    assert_eq!(segments, vec!["/usr/bin", "/usr/local/bin"]);
}

// =========================================================================
// Merge and dedup tests
// =========================================================================

#[test]
fn test_merge_and_dedup_no_overlap() {
    let current = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
    let saved = vec!["/opt/bin".to_string()];
    let result = merge_and_dedup(&current, &saved);
    assert_eq!(result, vec!["/usr/bin", "/usr/local/bin", "/opt/bin"]);
}

#[test]
fn test_merge_and_dedup_with_overlap() {
    let current = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
    let saved = vec![
        "/usr/bin".to_string(),
        "/opt/bin".to_string(),
    ];
    let result = merge_and_dedup(&current, &saved);
    // /usr/bin appears only once, from current (first occurrence wins)
    assert_eq!(result, vec!["/usr/bin", "/usr/local/bin", "/opt/bin"]);
}

#[test]
fn test_merge_and_dedup_preserves_order() {
    let current = vec![
        "/c".to_string(),
        "/a".to_string(),
        "/b".to_string(),
    ];
    let saved = vec![
        "/d".to_string(),
        "/a".to_string(), // duplicate
    ];
    let result = merge_and_dedup(&current, &saved);
    assert_eq!(result, vec!["/c", "/a", "/b", "/d"]);
}

#[test]
fn test_merge_and_dedup_both_empty() {
    let result = merge_and_dedup(&[], &[]);
    assert!(result.is_empty());
}

#[test]
fn test_merge_and_dedup_current_empty() {
    let saved = vec!["/opt/bin".to_string()];
    let result = merge_and_dedup(&[], &saved);
    assert_eq!(result, vec!["/opt/bin"]);
}

#[test]
fn test_merge_and_dedup_saved_empty() {
    let current = vec!["/usr/bin".to_string()];
    let result = merge_and_dedup(&current, &[]);
    assert_eq!(result, vec!["/usr/bin"]);
}

#[test]
fn test_merge_and_dedup_filters_empty_entries() {
    let current = vec!["".to_string(), "/usr/bin".to_string()];
    let saved = vec!["".to_string(), "/opt/bin".to_string()];
    let result = merge_and_dedup(&current, &saved);
    assert_eq!(result, vec!["/usr/bin", "/opt/bin"]);
}

#[test]
fn test_merge_and_dedup_all_duplicates() {
    let current = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
    let saved = vec!["/usr/local/bin".to_string(), "/usr/bin".to_string()];
    let result = merge_and_dedup(&current, &saved);
    assert_eq!(result, vec!["/usr/bin", "/usr/local/bin"]);
}

// =========================================================================
// Join and separator tests
// =========================================================================

#[test]
fn test_join_path_segments() {
    let segments = vec!["/usr/bin".to_string(), "/usr/local/bin".to_string()];
    let joined = join_path_segments(&segments);
    #[cfg(not(target_os = "windows"))]
    assert_eq!(joined, "/usr/bin:/usr/local/bin");
    #[cfg(target_os = "windows")]
    assert_eq!(joined, "/usr/bin;/usr/local/bin");
}

#[test]
fn test_join_path_segments_empty() {
    let segments: Vec<String> = vec![];
    let joined = join_path_segments(&segments);
    assert_eq!(joined, "");
}

#[test]
fn test_join_path_segments_single() {
    let segments = vec!["/usr/bin".to_string()];
    let joined = join_path_segments(&segments);
    assert_eq!(joined, "/usr/bin");
}

#[test]
fn test_path_separator_value() {
    #[cfg(not(target_os = "windows"))]
    assert_eq!(PATH_SEPARATOR, ':');
    #[cfg(target_os = "windows")]
    assert_eq!(PATH_SEPARATOR, ';');
}

// =========================================================================
// Integration: setup_environment_path
// =========================================================================

#[test]
fn test_setup_environment_path_captures_path() {
    // Test the full flow using component functions
    let current = get_current_path();
    assert!(!current.is_empty(), "System PATH should not be empty");

    let segments = expand_path_segments(&current);
    assert!(!segments.is_empty());

    let merged = merge_and_dedup(&segments, &[]);
    // Dedup may reduce count if system PATH has duplicates
    assert!(merged.len() <= segments.len());
    assert!(!merged.is_empty());

    let joined = join_path_segments(&merged);
    assert!(!joined.is_empty());
}
