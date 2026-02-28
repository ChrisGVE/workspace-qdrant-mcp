//! Helper functions for project detection.

use std::collections::HashMap;
use std::path::Path;

/// Calculate weight for build system importance
pub(super) fn calculate_build_system_weight(name: &str) -> u32 {
    match name {
        "cargo" => 95,  // Rust
        "npm" | "yarn" | "pnpm" => 90,  // JavaScript/Node.js
        "maven" | "gradle" => 85,  // Java
        "poetry" | "pip" => 80,  // Python
        "go" => 95,  // Go modules
        "composer" => 75,  // PHP
        "gem" | "bundle" => 75,  // Ruby
        "mix" => 80,  // Elixir
        "stack" | "cabal" => 70,  // Haskell
        "dune" => 70,  // OCaml
        "leiningen" => 70,  // Clojure
        "cmake" | "make" => 60,  // C/C++
        "meson" => 65,  // Alternative C/C++
        _ => 50,
    }
}

/// Detect frameworks for a specific language
pub(super) fn detect_frameworks_for_language(language: &str) -> HashMap<String, Vec<String>> {
    let mut frameworks = HashMap::new();

    match language {
        "javascript" | "typescript" => {
            frameworks.insert("react".to_string(), vec!["package.json".to_string()]);
            frameworks.insert("vue".to_string(), vec!["vue.config.js".to_string()]);
            frameworks.insert("angular".to_string(), vec!["angular.json".to_string()]);
            frameworks.insert("next".to_string(), vec!["next.config.js".to_string()]);
            frameworks.insert("nuxt".to_string(), vec!["nuxt.config.js".to_string()]);
        },
        "python" => {
            frameworks.insert("django".to_string(), vec!["manage.py".to_string(), "settings.py".to_string()]);
            frameworks.insert("flask".to_string(), vec!["app.py".to_string()]);
            frameworks.insert("fastapi".to_string(), vec!["main.py".to_string()]);
        },
        "rust" => {
            frameworks.insert("actix".to_string(), vec!["Cargo.toml".to_string()]);
            frameworks.insert("rocket".to_string(), vec!["Cargo.toml".to_string()]);
            frameworks.insert("warp".to_string(), vec!["Cargo.toml".to_string()]);
        },
        "go" => {
            frameworks.insert("gin".to_string(), vec!["go.mod".to_string()]);
            frameworks.insert("echo".to_string(), vec!["go.mod".to_string()]);
            frameworks.insert("fiber".to_string(), vec!["go.mod".to_string()]);
        },
        _ => {},
    }

    frameworks
}

/// Get framework detection patterns
pub(super) fn get_framework_patterns() -> HashMap<String, Vec<String>> {
    let mut patterns = HashMap::new();

    // Web frameworks
    patterns.insert("react".to_string(), vec![
        "package.json".to_string(),
        "src/App.jsx".to_string(),
        "src/App.js".to_string(),
        "public/index.html".to_string(),
    ]);

    patterns.insert("vue".to_string(), vec![
        "vue.config.js".to_string(),
        "src/App.vue".to_string(),
        "src/main.js".to_string(),
    ]);

    patterns.insert("angular".to_string(), vec![
        "angular.json".to_string(),
        "src/app/app.module.ts".to_string(),
        "src/main.ts".to_string(),
    ]);

    patterns.insert("django".to_string(), vec![
        "manage.py".to_string(),
        "*/settings.py".to_string(),
        "*/urls.py".to_string(),
    ]);

    patterns.insert("flask".to_string(), vec![
        "app.py".to_string(),
        "application.py".to_string(),
        "run.py".to_string(),
    ]);

    patterns.insert("rails".to_string(), vec![
        "Gemfile".to_string(),
        "config/application.rb".to_string(),
        "app/controllers/application_controller.rb".to_string(),
    ]);

    patterns.insert("spring".to_string(), vec![
        "pom.xml".to_string(),
        "src/main/java/**/Application.java".to_string(),
        "application.properties".to_string(),
    ]);

    patterns
}

/// Check if a file matches a pattern (with glob-like support)
pub(super) fn file_matches_pattern(file: &str, pattern: &str) -> bool {
    // Exact match
    if file == pattern {
        return true;
    }

    // Ends with pattern
    if pattern.starts_with('*') && file.ends_with(&pattern[1..]) {
        return true;
    }

    // Contains pattern
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return file.starts_with(parts[0]) && file.ends_with(parts[1]);
        }
    }

    // Directory-based matching
    if pattern.contains('/') && file.contains(pattern) {
        return true;
    }

    // Filename only matching
    if let Some(filename) = Path::new(file).file_name().and_then(|f| f.to_str()) {
        return filename == pattern;
    }

    false
}
