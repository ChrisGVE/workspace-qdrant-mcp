use super::helpers::classify_and_store_pattern;
use super::*;
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
    assert!(
        engine
            .should_exclude("node_modules/package/index.js")
            .excluded
    );

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
    classify_and_store_pattern(
        "exact",
        &rule,
        &mut exact,
        &mut prefix,
        &mut suffix,
        &mut contains,
    );
    assert!(contains.contains(&"exact".to_string()));

    // Test prefix pattern
    classify_and_store_pattern(
        "prefix*",
        &rule,
        &mut exact,
        &mut prefix,
        &mut suffix,
        &mut contains,
    );
    assert!(prefix.contains(&"prefix".to_string()));

    // Test suffix pattern
    classify_and_store_pattern(
        "*.suffix",
        &rule,
        &mut exact,
        &mut prefix,
        &mut suffix,
        &mut contains,
    );
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
    assert!(
        engine
            .should_exclude("project/node_modules/package.json")
            .excluded
    );
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
    assert!(
        engine
            .should_exclude("deep/path/.mypy_cache/file.json")
            .excluded
    );
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
    assert!(
        !engine
            .should_exclude("project/.github/workflows/test.yml")
            .excluded
    );

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

// ── WI-b1 (#82 task 15): thin wqm-common helpers vs full daemon engine ──────
//
// The CLI now uses the dependency-free `wqm_common::exclusion` helpers for its
// filesystem walks instead of the daemon's `ExclusionEngine`. For the
// well-known sample below the two must return identical verdicts; the thin
// helpers intentionally cover only this critical set, not the engine's full
// configurable pattern space.

#[test]
fn thin_helpers_match_engine_for_directory_sample() {
    let sample = [
        "target",
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        ".mypy_cache",
        ".github",
        "src",
        "lib",
        "tests",
        "docs",
    ];
    for dir in sample {
        assert_eq!(
            wqm_common::exclusion::should_exclude_directory(dir),
            should_exclude_directory(dir),
            "directory verdict mismatch for {dir:?}"
        );
    }
}

#[test]
fn thin_helpers_match_engine_for_file_sample() {
    let sample = [
        ".git/config",
        "target/debug/app",
        "node_modules/package.json",
        ".DS_Store",
        "path/to/.DS_Store",
        "src/main.rs",
    ];
    for file in sample {
        assert_eq!(
            wqm_common::exclusion::should_exclude_file(file),
            should_exclude_file(file),
            "file verdict mismatch for {file:?}"
        );
    }
}

// ── should_exclude_file_in_root (#97) ────────────────────────────────────────
//
// Regression: projects registered under a dotted directory (e.g.
// `~/.config/main-docker`) had every file excluded because the absolute path
// was fed to the engine and `.config` matched the hidden-component rule.

#[test]
fn in_root_dot_dir_above_watch_root_not_excluded() {
    let root = "/Users/x/.config/main-docker";
    assert!(!should_exclude_file_in_root(
        "/Users/x/.config/main-docker/prometheus.yml",
        root
    ));
    assert!(!should_exclude_file_in_root(
        "/Users/x/.config/main-docker/provisioning/dashboards/services.json",
        root
    ));
}

#[test]
fn in_root_hidden_components_inside_root_still_excluded() {
    let root = "/Users/x/.config/main-docker";
    assert!(should_exclude_file_in_root(
        "/Users/x/.config/main-docker/.env",
        root
    ));
    assert!(should_exclude_file_in_root(
        "/Users/x/.config/main-docker/.git/config",
        root
    ));
    assert!(should_exclude_file_in_root(
        "/Users/x/.config/main-docker/sub/.hidden/file.txt",
        root
    ));
}

#[test]
fn in_root_pattern_rules_inside_root_still_apply() {
    let root = "/Users/x/.config/proj";
    assert!(should_exclude_file_in_root(
        "/Users/x/.config/proj/node_modules/pkg/index.js",
        root
    ));
    assert!(should_exclude_file_in_root(
        "/Users/x/.config/proj/target/debug/app",
        root
    ));
    assert!(!should_exclude_file_in_root(
        "/Users/x/.config/proj/src/main.rs",
        root
    ));
}

#[test]
fn in_root_github_whitelist_preserved() {
    let root = "/Users/x/.config/proj";
    assert!(!should_exclude_file_in_root(
        "/Users/x/.config/proj/.github/workflows/ci.yml",
        root
    ));
}

#[test]
fn in_root_component_boundary_no_partial_strip() {
    // `/a/b` must not strip against `/a/bc/...` — the remainder would lose
    // its leading component and bypass exclusion rules.
    assert!(should_exclude_file_in_root("/a/bc/.git/config", "/a/b"));
}

#[test]
fn in_root_path_outside_root_falls_back_to_full_check() {
    // Not under the root: full-path semantics apply (defensive).
    assert!(should_exclude_file_in_root(
        "/elsewhere/.hidden/file.txt",
        "/Users/x/proj"
    ));
    assert!(!should_exclude_file_in_root(
        "/elsewhere/src/main.rs",
        "/Users/x/proj"
    ));
}

#[test]
fn in_root_root_itself_never_excluded() {
    assert!(!should_exclude_file_in_root(
        "/Users/x/.config/main-docker",
        "/Users/x/.config/main-docker"
    ));
    // Trailing-slash root normalizes.
    assert!(!should_exclude_file_in_root(
        "/Users/x/.config/main-docker/compose.yml",
        "/Users/x/.config/main-docker/"
    ));
}
