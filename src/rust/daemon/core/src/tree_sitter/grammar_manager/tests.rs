//! Tests for grammar manager modules.

use super::*;
use crate::config::GrammarConfig;
use crate::tree_sitter::grammar_cache::GrammarMetadata;
use tempfile::TempDir;

fn test_config(temp_dir: &TempDir, auto_download: bool) -> GrammarConfig {
    GrammarConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        required: vec!["rust".to_string(), "python".to_string()],
        auto_download,
        tree_sitter_version: "0.24".to_string(),
        download_base_url: "https://example.com/{language}/v{version}/{platform}.{ext}"
            .to_string(),
        verify_checksums: false,
        lazy_loading: true,
        check_interval_hours: 168, // Weekly
    }
}

// --- Core creation tests ---

#[test]
fn test_grammar_manager_creation() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    assert!(manager.loaded_languages().is_empty());
}

#[test]
fn test_grammar_manager_no_auto_download() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    // Downloader should be None
    assert!(manager.downloader.is_none());
}

#[test]
fn test_grammar_manager_debug() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    let debug_str = format!("{:?}", manager);
    assert!(debug_str.contains("GrammarManager"));
    assert!(debug_str.contains("auto_download"));
}

#[test]
fn test_set_default_version() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let mut manager = GrammarManager::new(config);

    manager.set_default_version("0.24.0");
    assert_eq!(manager.default_version, "0.24.0");
}

#[test]
fn test_default_version_from_config() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    // default_version should match config.tree_sitter_version, not hardcoded
    assert_eq!(manager.default_version, "0.24");
}

#[test]
fn test_default_version_custom_config() {
    let temp_dir = TempDir::new().unwrap();
    let config = GrammarConfig {
        tree_sitter_version: "0.25.1".to_string(),
        ..test_config(&temp_dir, true)
    };
    let manager = GrammarManager::new(config);

    assert_eq!(manager.default_version, "0.25.1");
}

// --- Cache status tests ---

#[test]
fn test_grammar_status_not_cached() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    // With auto_download, uncached grammars should need download
    assert_eq!(
        manager.grammar_status("rust"),
        GrammarStatus::NeedsDownload
    );

    // Without auto_download, uncached grammars should be unavailable
    let temp_dir2 = TempDir::new().unwrap();
    let config2 = test_config(&temp_dir2, false);
    let manager2 = GrammarManager::new(config2);
    assert_eq!(
        manager2.grammar_status("rust"),
        GrammarStatus::NotAvailable
    );
}

#[test]
fn test_grammar_info() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    let info = manager.grammar_info("rust");
    assert_eq!(info.language, "rust");
    assert_eq!(info.status, GrammarStatus::NeedsDownload);
    assert!(info.metadata.is_none());
    assert!(info.compatibility.is_none());
}

#[test]
fn test_missing_required() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    let missing = manager.missing_required();
    assert!(missing.contains(&"rust".to_string()));
    assert!(missing.contains(&"python".to_string()));
}

#[test]
fn test_unload_grammar() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let mut manager = GrammarManager::new(config);

    // Unloading a non-loaded grammar should return false
    assert!(!manager.unload_grammar("rust"));
}

#[test]
fn test_unload_all() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let mut manager = GrammarManager::new(config);

    manager.unload_all();
    assert!(manager.loaded_languages().is_empty());
}

#[test]
fn test_clear_cache_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    // Clearing cache for a language that doesn't exist should return Ok(false)
    let result = manager.clear_cache("nonexistent");
    assert!(result.is_ok());
    assert!(!result.unwrap()); // false = nothing was cleared
}

#[test]
fn test_clear_cache_with_cached_grammar() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    // Create a fake cache in the correct directory structure
    let grammar_path = manager.loader.cache_paths().grammar_path("rust");
    std::fs::create_dir_all(grammar_path.parent().unwrap()).unwrap();
    std::fs::write(&grammar_path, b"fake grammar").unwrap();

    assert!(grammar_path.exists());

    // Clear should succeed and return true
    let result = manager.clear_cache("rust");
    assert!(result.is_ok());
    assert!(result.unwrap()); // true = something was cleared

    // Grammar file should no longer exist
    assert!(!grammar_path.exists());
}

#[test]
fn test_clear_all_cache_empty() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    // Clearing all cache when empty should return 0
    let result = manager.clear_all_cache();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

#[test]
fn test_clear_all_cache_with_multiple_languages() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    // Create fake cache directories for multiple languages with correct structure
    for lang in &["rust", "python", "go"] {
        let grammar_path = manager.loader.cache_paths().grammar_path(lang);
        std::fs::create_dir_all(grammar_path.parent().unwrap()).unwrap();
        std::fs::write(&grammar_path, b"fake grammar").unwrap();
    }

    // Clear all should succeed
    let result = manager.clear_all_cache();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 3); // 3 languages cleared

    // Verify grammar files no longer exist
    for lang in &["rust", "python", "go"] {
        let grammar_path = manager.loader.cache_paths().grammar_path(lang);
        assert!(
            !grammar_path.exists(),
            "Grammar for {} should be cleared",
            lang
        );
    }
}

// --- Loading/reload tests ---

#[tokio::test]
async fn test_reload_grammar_not_loaded() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let mut manager = GrammarManager::new(config);

    // Reloading a grammar that was never loaded and can't be downloaded
    // should fail with AutoDownloadDisabled (since auto_download is false)
    let result = manager.reload_grammar("nonexistent").await;
    assert!(result.is_err());
    match result {
        Err(GrammarError::AutoDownloadDisabled(lang)) => assert_eq!(lang, "nonexistent"),
        other => panic!("Expected AutoDownloadDisabled error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_reload_all_empty() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let mut manager = GrammarManager::new(config);

    // Reloading all when nothing is loaded should return empty map
    let results = manager.reload_all().await;
    assert!(results.is_empty());
}

// --- Validation tests ---

#[test]
fn test_validate_grammars_with_auto_download() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true); // auto_download enabled
    let manager = GrammarManager::new(config);

    let result = manager.validate_grammars();

    // With auto_download, missing grammars should be in needs_download
    assert!(!result.needs_download.is_empty());
    assert!(result.unavailable.is_empty());
    // All required are "available" because they can be downloaded
    assert!(result.is_valid());
}

#[test]
fn test_validate_grammars_without_auto_download() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false); // auto_download disabled
    let manager = GrammarManager::new(config);

    let result = manager.validate_grammars();

    // Without auto_download, missing grammars should be unavailable
    assert!(!result.unavailable.is_empty());
    // Not all required are available
    assert!(!result.is_valid());
}

#[test]
fn test_validation_result_summary() {
    let result = GrammarValidationResult {
        available: vec!["rust".to_string()],
        needs_download: vec!["python".to_string()],
        unavailable: vec!["go".to_string()],
        all_required_available: false,
        messages: vec![],
    };

    let summary = result.summary();
    assert!(summary.contains("1 available"));
    assert!(summary.contains("1 need download"));
    assert!(summary.contains("1 unavailable"));
}

#[test]
fn test_validation_result_is_valid() {
    // Valid: all required available
    let valid_result = GrammarValidationResult {
        available: vec!["rust".to_string()],
        needs_download: vec![],
        unavailable: vec![],
        all_required_available: true,
        messages: vec![],
    };
    assert!(valid_result.is_valid());

    // Invalid: some unavailable
    let invalid_result = GrammarValidationResult {
        available: vec![],
        needs_download: vec![],
        unavailable: vec!["rust".to_string()],
        all_required_available: false,
        messages: vec![],
    };
    assert!(!invalid_result.is_valid());
}

// --- Periodic check tests ---

#[test]
fn test_needs_periodic_check_no_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    // No metadata files exist, so periodic check is needed
    assert!(manager.needs_periodic_check());
}

#[test]
fn test_needs_periodic_check_recent_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    // Create metadata with recent last_checked_at for all required grammars
    let cache_paths = manager.loader.cache_paths();
    for lang in &["rust", "python"] {
        cache_paths.create_directories(lang).unwrap();
        std::fs::write(cache_paths.grammar_path(lang), "fake").unwrap();

        let mut metadata = GrammarMetadata::new(
            *lang,
            "0.24",
            "0.24.0",
            &cache_paths.platform,
            "checksum",
        );
        metadata.mark_checked();
        cache_paths.save_metadata(lang, &metadata).unwrap();
    }

    // Recent check - should not need periodic check
    assert!(!manager.needs_periodic_check());
}

#[test]
fn test_needs_periodic_check_disabled() {
    let temp_dir = TempDir::new().unwrap();
    let config = GrammarConfig {
        check_interval_hours: 0,
        ..test_config(&temp_dir, true)
    };
    let manager = GrammarManager::new(config);

    // check_interval_hours = 0 disables periodic checks
    assert!(!manager.needs_periodic_check());
}

#[test]
fn test_default_url_template_has_all_placeholders() {
    // Verify the default URL template contains all required placeholders
    let config = GrammarConfig::default();
    assert!(config.download_base_url.contains("{language}"));
    assert!(config.download_base_url.contains("{version}"));
    assert!(config.download_base_url.contains("{platform}"));
    assert!(config.download_base_url.contains("{ext}"));
    // Verify it's a complete URL, not just a base path
    assert!(config
        .download_base_url
        .contains("tree-sitter-{language}-{platform}"));
}

#[tokio::test]
async fn test_periodic_version_check_no_downloader() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false); // auto_download disabled
    let mut manager = GrammarManager::new(config);

    let results = manager.periodic_version_check().await;
    assert!(results.is_empty()); // Skipped because no downloader
}

// --- Provider tests ---

#[test]
fn test_loaded_grammars_provider_new() {
    use crate::tree_sitter::parser::LanguageProvider;

    let provider = LoadedGrammarsProvider::new();
    assert!(provider.is_empty());
    assert_eq!(provider.len(), 0);
    assert!(provider.available_languages().is_empty());
}

#[test]
fn test_loaded_grammars_provider_with_grammar() {
    use crate::tree_sitter::parser::{get_static_language, LanguageProvider};

    let mut provider = LoadedGrammarsProvider::new();

    // Add a real grammar from static loading
    if let Some(lang) = get_static_language("rust") {
        provider.add_grammar("rust", lang);
    }

    assert!(!provider.is_empty());
    assert_eq!(provider.len(), 1);
    assert!(provider.supports_language("rust"));
    assert!(!provider.supports_language("python"));

    let languages = provider.available_languages();
    assert!(languages.contains(&"rust"));
}

#[test]
fn test_loaded_grammars_provider_get_language() {
    use crate::tree_sitter::parser::{get_static_language, LanguageProvider};

    let mut provider = LoadedGrammarsProvider::new();

    if let Some(lang) = get_static_language("rust") {
        provider.add_grammar("rust", lang);
    }

    // Should return the language
    assert!(provider.get_language("rust").is_some());

    // Should return None for unknown language
    assert!(provider.get_language("unknown").is_none());
}

#[test]
fn test_loaded_grammars_provider_from_map() {
    use crate::tree_sitter::parser::{get_static_language, LanguageProvider};
    use std::collections::HashMap;

    let mut map = HashMap::new();
    if let Some(lang) = get_static_language("rust") {
        map.insert("rust".to_string(), lang);
    }
    if let Some(lang) = get_static_language("python") {
        map.insert("python".to_string(), lang);
    }

    let provider = LoadedGrammarsProvider::from_loaded(map);

    assert_eq!(provider.len(), 2);
    assert!(provider.supports_language("rust"));
    assert!(provider.supports_language("python"));
}

#[test]
fn test_grammar_manager_create_language_provider() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    // Manager has no loaded grammars yet
    let provider = manager.create_language_provider();
    assert!(provider.is_empty());
}
