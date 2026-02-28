use super::*;
use super::helpers::classify_and_store_pattern;
use std::collections::HashSet;

#[test]
fn test_engine_initialization() {
    let engine = ExclusionEngine::new();
    assert!(engine.is_ok(), "Should initialize exclusion engine");
}

#[test]
fn test_basic_exclusion() {
    let engine = ExclusionEngine::new().unwrap();

    // Version control
    assert!(engine.should_exclude(".git/config").excluded);
    assert!(engine.should_exclude(".gitignore").excluded);

    // Node modules
    assert!(engine.should_exclude("node_modules/package/index.js").excluded);

    // Build artifacts
    assert!(engine.should_exclude("target/debug/main").excluded);

    // Should not exclude source files
    assert!(!engine.should_exclude("src/main.rs").excluded);
    assert!(!engine.should_exclude("README.md").excluded);
}

#[test]
fn test_contextual_exclusion() {
    let engine = ExclusionEngine::new().unwrap();

    // Rust project context
    let result = engine.check_with_context("target/debug/app", Some("rust"));
    assert!(result.excluded);

    // Python project context
    let result = engine.check_with_context("src/__pycache__/module.pyc", Some("python"));
    assert!(result.excluded);

    // JavaScript project context
    let result = engine.check_with_context("node_modules/lodash/index.js", Some("javascript"));
    assert!(result.excluded);
}

#[test]
fn test_critical_patterns() {
    let engine = ExclusionEngine::new().unwrap();

    // System files
    assert!(engine.should_exclude(".DS_Store").excluded);
    assert!(engine.should_exclude("Thumbs.db").excluded);

    // Security files
    assert!(engine.should_exclude(".env").excluded);
    assert!(engine.should_exclude("id_rsa").excluded);

    // Temporary files
    assert!(engine.should_exclude("file.tmp").excluded);
    assert!(engine.should_exclude("document.swp").excluded);
}

#[test]
fn test_pattern_classification() {
    let mut exact = HashSet::new();
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    let mut contains = Vec::new();

    let rule = ExclusionRule {
        pattern: "test".to_string(),
        category: ExclusionCategory::Cache,
        reason: "test".to_string(),
        is_regex: false,
        case_sensitive: true,
    };

    // Test exact pattern
    classify_and_store_pattern("exact", &rule, &mut exact, &mut prefix, &mut suffix, &mut contains);
    assert!(contains.contains(&"exact".to_string()));

    // Test prefix pattern
    classify_and_store_pattern("prefix*", &rule, &mut exact, &mut prefix, &mut suffix, &mut contains);
    assert!(prefix.contains(&"prefix".to_string()));

    // Test suffix pattern
    classify_and_store_pattern("*.suffix", &rule, &mut exact, &mut prefix, &mut suffix, &mut contains);
    assert!(suffix.contains(&".suffix".to_string()));
}

#[test]
fn test_exclusion_stats() {
    let engine = ExclusionEngine::new().unwrap();
    let stats = engine.stats();

    assert!(stats.total_rules > 0);
    assert!(stats.contains_patterns + stats.prefix_patterns + stats.suffix_patterns > 0);
    assert!(!stats.category_counts.is_empty());
}

#[test]
fn test_global_engine() {
    let engine = ExclusionEngine::global();
    assert!(engine.is_ok(), "Global engine should be available");
}

#[test]
fn test_convenience_functions() {
    assert!(should_exclude_file(".git/config"));
    assert!(!should_exclude_file("src/main.rs"));

    let result = should_exclude_file_with_context("target/debug/app", "rust");
    assert!(result.excluded);
}

#[test]
fn test_filename_vs_path_exclusion() {
    let engine = ExclusionEngine::new().unwrap();

    // Test that both full path and filename are checked
    assert!(engine.should_exclude("path/to/.DS_Store").excluded);
    assert!(engine.should_exclude(".DS_Store").excluded);

    // Test directory patterns
    assert!(engine.should_exclude("project/node_modules/package.json").excluded);
    assert!(engine.should_exclude("node_modules/package.json").excluded);
}

#[test]
fn test_hidden_files_excluded_at_all_depths() {
    let engine = ExclusionEngine::new().unwrap();

    // Hidden directories at root level
    assert!(engine.should_exclude(".mypy_cache/something.json").excluded);
    assert!(engine.should_exclude(".vscode/settings.json").excluded);
    assert!(engine.should_exclude(".idea/workspace.xml").excluded);

    // Hidden directories at arbitrary depth
    assert!(engine.should_exclude("src/.cache/file.txt").excluded);
    assert!(engine.should_exclude("deep/path/.mypy_cache/file.json").excluded);
    assert!(engine.should_exclude("a/b/c/.hidden/file.txt").excluded);

    // Hidden files (not just directories)
    assert!(engine.should_exclude("src/.hidden_file").excluded);
    assert!(engine.should_exclude("deep/path/.secret").excluded);

    // Multiple hidden components
    assert!(engine.should_exclude(".hidden1/.hidden2/file.txt").excluded);
}

#[test]
fn test_github_directory_not_excluded() {
    let engine = ExclusionEngine::new().unwrap();

    // .github is explicitly allowed (useful for CI/CD understanding)
    assert!(!engine.should_exclude(".github/workflows/ci.yml").excluded);
    assert!(!engine.should_exclude(".github/CODEOWNERS").excluded);
    assert!(!engine.should_exclude("project/.github/workflows/test.yml").excluded);

    // But other .g* directories should still be excluded
    assert!(engine.should_exclude(".gradle/cache/file").excluded);
}

#[test]
fn test_non_hidden_paths_not_excluded_by_hidden_rule() {
    let engine = ExclusionEngine::new().unwrap();

    // Normal source files should NOT be excluded by hidden rule
    let result = engine.should_exclude("src/main.rs");
    if result.excluded {
        assert!(!result.reason.contains("Hidden path component"));
    }

    let result = engine.should_exclude("lib/utils/helper.py");
    if result.excluded {
        assert!(!result.reason.contains("Hidden path component"));
    }

    // Files with dots in name (not at start) are fine
    let result = engine.should_exclude("config.json");
    if result.excluded {
        assert!(!result.reason.contains("Hidden path component"));
    }

    let result = engine.should_exclude("src/my.module.ts");
    if result.excluded {
        assert!(!result.reason.contains("Hidden path component"));
    }
}

#[test]
fn test_hidden_component_with_various_formats() {
    let engine = ExclusionEngine::new().unwrap();

    // Windows-style paths (with backslashes) - test robustness
    assert!(engine.should_exclude("src/.hidden/file").excluded);

    // Leading/trailing slashes
    assert!(engine.should_exclude("/.hidden/file").excluded);
    assert!(engine.should_exclude(".hidden/file/").excluded);

    // Multiple slashes (should handle gracefully)
    assert!(engine.should_exclude("src//.hidden//file").excluded);
}

#[test]
fn test_should_exclude_directory() {
    // Well-known excluded directories
    assert!(should_exclude_directory("target"));
    assert!(should_exclude_directory("node_modules"));
    assert!(should_exclude_directory("__pycache__"));

    // Hidden directories
    assert!(should_exclude_directory(".git"));
    assert!(should_exclude_directory(".venv"));
    assert!(should_exclude_directory(".mypy_cache"));

    // .github is whitelisted - should NOT be excluded
    assert!(!should_exclude_directory(".github"));

    // Normal directories should NOT be excluded
    assert!(!should_exclude_directory("src"));
    assert!(!should_exclude_directory("lib"));
    assert!(!should_exclude_directory("tests"));
    assert!(!should_exclude_directory("docs"));
}
