use super::path_env::*;
use super::*;
use serial_test::serial;
use std::path::PathBuf;

// --- Profile-aware env resolution tests -----------------------------------

fn scrub_profile_env() {
    std::env::remove_var("WQM_DAEMON_ADDR");
    std::env::remove_var("WQM_QDRANT_URL");
    std::env::remove_var("QDRANT_URL");
    std::env::remove_var("QDRANT_API_KEY");
    std::env::remove_var("WQM_PROFILE");
}

fn point_cli_config_at_nonexistent_path() {
    std::env::set_var(
        "WQM_CLI_CONFIG",
        "/tmp/wqm-config-tests-nonexistent-path/cli-config.toml",
    );
}

#[test]
#[serial]
fn resolve_daemon_address_falls_back_to_default_when_no_profile() {
    scrub_profile_env();
    point_cli_config_at_nonexistent_path();
    let addr = resolve_daemon_address();
    assert!(addr.starts_with("http://"), "got {addr}");
    assert!(addr.contains("50051"), "got {addr}");
    std::env::remove_var("WQM_CLI_CONFIG");
}

#[test]
#[serial]
fn resolve_daemon_address_honors_env_override() {
    scrub_profile_env();
    point_cli_config_at_nonexistent_path();
    std::env::set_var("WQM_DAEMON_ADDR", "http://10.0.0.5:60000");
    let addr = resolve_daemon_address();
    assert_eq!(addr, "http://10.0.0.5:60000");
    std::env::remove_var("WQM_DAEMON_ADDR");
    std::env::remove_var("WQM_CLI_CONFIG");
}

#[test]
#[serial]
fn resolve_qdrant_url_prefers_explicit_env_var() {
    scrub_profile_env();
    point_cli_config_at_nonexistent_path();
    std::env::set_var("QDRANT_URL", "http://qdrant.internal:6333");
    let url = resolve_qdrant_url();
    assert_eq!(url, "http://qdrant.internal:6333");
    std::env::remove_var("QDRANT_URL");
    std::env::remove_var("WQM_CLI_CONFIG");
}

#[test]
#[serial]
fn resolve_qdrant_api_key_reads_direct_env() {
    scrub_profile_env();
    point_cli_config_at_nonexistent_path();
    std::env::set_var("QDRANT_API_KEY", "sekret");
    let key = resolve_qdrant_api_key();
    std::env::remove_var("QDRANT_API_KEY");
    std::env::remove_var("WQM_CLI_CONFIG");
    assert_eq!(key.as_deref(), Some("sekret"));
}

#[test]
#[serial]
fn from_env_applies_active_profile_from_cli_config() {
    use wqm_common::cli_profiles::{save_cli_config, CliConfigFile};
    scrub_profile_env();

    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("cli-config.toml");
    let mut cfg = CliConfigFile::default();
    cfg.set_active("docker-local").unwrap();
    save_cli_config(&path, &cfg).unwrap();

    std::env::set_var("WQM_CLI_CONFIG", &path);
    let resolved = Config::from_env();
    assert_eq!(resolved.active_profile, "docker-local");

    std::env::remove_var("WQM_CLI_CONFIG");
}

#[test]
#[serial]
fn from_env_wqm_profile_overrides_active_field() {
    use wqm_common::cli_profiles::{save_cli_config, CliConfigFile};
    scrub_profile_env();

    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("cli-config.toml");
    let cfg = CliConfigFile::default(); // active = native
    save_cli_config(&path, &cfg).unwrap();

    std::env::set_var("WQM_CLI_CONFIG", &path);
    std::env::set_var("WQM_PROFILE", "docker-local");
    let resolved = Config::from_env();
    std::env::remove_var("WQM_PROFILE");
    std::env::remove_var("WQM_CLI_CONFIG");
    assert_eq!(resolved.active_profile, "docker-local");
}

fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: std::collections::HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |key: &str| map.get(key).cloned()
}

#[test]
#[serial]
fn from_env_with_empty_getter_yields_defaults() {
    // AC-a2.4 PURE-DEFAULTS: no env overrides + no cli-config profile → the
    // resolved config equals the compiled-in default view.
    scrub_profile_env();
    point_cli_config_at_nonexistent_path();
    let resolved = Config::from_env_with(&env_from(&[]));
    std::env::remove_var("WQM_CLI_CONFIG");
    let default = Config::default();
    assert_eq!(resolved.daemon_address, default.daemon_address);
    assert_eq!(resolved.qdrant_url, default.qdrant_url);
    assert_eq!(
        resolved.connection_timeout_secs,
        default.connection_timeout_secs
    );
    assert_eq!(resolved.output_format, default.output_format);
    assert_eq!(resolved.color_enabled, default.color_enabled);
    assert_eq!(resolved.verbose, default.verbose);
}

#[test]
#[serial]
fn from_env_with_applies_all_overrides() {
    // AC-a2.2 PARITY: the shared declarative engine reproduces the prior
    // per-variable override behaviour for every CLI env var.
    scrub_profile_env();
    point_cli_config_at_nonexistent_path();
    let resolved = Config::from_env_with(&env_from(&[
        ("WQM_DAEMON_ADDR", "http://10.0.0.5:60000"),
        ("WQM_QDRANT_URL", "http://qd:6333"),
        ("WQM_TIMEOUT", "42"),
        ("WQM_OUTPUT_FORMAT", "json"),
        ("NO_COLOR", "1"),
        ("WQM_VERBOSE", "1"),
    ]));
    std::env::remove_var("WQM_CLI_CONFIG");
    assert_eq!(resolved.daemon_address, "http://10.0.0.5:60000");
    assert_eq!(resolved.qdrant_url, "http://qd:6333");
    assert_eq!(resolved.connection_timeout_secs, 42);
    assert_eq!(resolved.output_format, OutputFormat::Json);
    assert!(!resolved.color_enabled);
    assert!(resolved.verbose);
}

#[test]
#[serial]
fn from_env_with_ignores_unparseable_timeout_and_format() {
    // Parse failures fall back to the default value (prior behaviour preserved).
    scrub_profile_env();
    point_cli_config_at_nonexistent_path();
    let resolved = Config::from_env_with(&env_from(&[
        ("WQM_TIMEOUT", "not-a-number"),
        ("WQM_OUTPUT_FORMAT", "bogus"),
    ]));
    std::env::remove_var("WQM_CLI_CONFIG");
    let default = Config::default();
    assert_eq!(
        resolved.connection_timeout_secs,
        default.connection_timeout_secs
    );
    assert_eq!(resolved.output_format, default.output_format);
}

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
#[serial]
fn test_get_database_path() {
    // Env vars are process-global; serialise this test against any
    // others that touch WQM_DATABASE_PATH so parallel test threads
    // don't observe each other's mutations.
    let prev = std::env::var("WQM_DATABASE_PATH").ok();

    // 1. Test default path (no env override)
    std::env::remove_var("WQM_DATABASE_PATH");
    let result = get_database_path();
    assert!(result.is_ok());
    let path = result.unwrap();
    assert!(
        path.to_string_lossy().contains("workspace-qdrant"),
        "expected path to contain 'workspace-qdrant', got: {path:?}"
    );
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
    let expanded = expand_path_segment("$WQM_TEST_NONEXISTENT_VAR_12345/bin");
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
    let saved = vec!["/usr/bin".to_string(), "/opt/bin".to_string()];
    let result = merge_and_dedup(&current, &saved);
    // /usr/bin appears only once, from current (first occurrence wins)
    assert_eq!(result, vec!["/usr/bin", "/usr/local/bin", "/opt/bin"]);
}

#[test]
fn test_merge_and_dedup_preserves_order() {
    let current = vec!["/c".to_string(), "/a".to_string(), "/b".to_string()];
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
