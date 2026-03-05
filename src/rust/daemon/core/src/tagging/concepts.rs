//! Dependency-derived concept tagging.
//!
//! Maps dependency names from Cargo.toml, package.json, requirements.txt,
//! and go.mod to concept tags using a built-in concept map covering the
//! most common packages across Rust, JavaScript/TypeScript, Python, and Go.

use crate::keyword_extraction::tag_selector::SelectedTag;

use super::tier1::make_tier1_tag;

// ── Concept map ──────────────────────────────────────────────────────────

/// A mapping from dependency name to concept tags.
///
/// This is a built-in subset covering the most common packages.
/// Can be extended via `assets/concept_normalization.yaml` in future.
const CONCEPT_MAP: &[(&str, &[&str])] = &[
    // Rust
    ("tokio", &["async-runtime", "concurrency"]),
    ("serde", &["serialization"]),
    ("serde_json", &["json", "serialization"]),
    ("reqwest", &["http-client", "networking"]),
    ("actix-web", &["web-framework", "http-server"]),
    ("axum", &["web-framework", "http-server"]),
    ("warp", &["web-framework", "http-server"]),
    ("diesel", &["orm", "database"]),
    ("sqlx", &["database", "sql"]),
    ("clap", &["cli", "argument-parsing"]),
    ("tracing", &["observability", "logging"]),
    ("tonic", &["grpc", "rpc"]),
    ("prost", &["protobuf", "serialization"]),
    ("rayon", &["parallelism", "concurrency"]),
    ("qdrant-client", &["vector-database", "search"]),
    ("fastembed", &["embeddings", "ml"]),
    // JavaScript/TypeScript
    ("react", &["ui-framework", "frontend"]),
    ("vue", &["ui-framework", "frontend"]),
    ("angular", &["ui-framework", "frontend"]),
    ("express", &["web-framework", "http-server"]),
    ("fastify", &["web-framework", "http-server"]),
    ("next", &["web-framework", "ssr"]),
    ("jest", &["testing"]),
    ("mocha", &["testing"]),
    ("vitest", &["testing"]),
    ("webpack", &["bundler", "build-tool"]),
    ("vite", &["bundler", "build-tool"]),
    ("typescript", &["type-system"]),
    ("prisma", &["orm", "database"]),
    ("mongoose", &["orm", "database"]),
    ("axios", &["http-client", "networking"]),
    // Python
    ("django", &["web-framework", "http-server"]),
    ("flask", &["web-framework", "http-server"]),
    ("fastapi", &["web-framework", "http-server"]),
    ("pandas", &["data-analysis", "dataframes"]),
    ("numpy", &["numerical-computing"]),
    ("scikit-learn", &["machine-learning"]),
    ("tensorflow", &["deep-learning", "ml"]),
    ("pytorch", &["deep-learning", "ml"]),
    ("torch", &["deep-learning", "ml"]),
    ("sqlalchemy", &["orm", "database"]),
    ("pytest", &["testing"]),
    ("requests", &["http-client", "networking"]),
    ("celery", &["task-queue", "async"]),
    // Go
    ("gin", &["web-framework", "http-server"]),
    ("echo", &["web-framework", "http-server"]),
    ("gorm", &["orm", "database"]),
    ("cobra", &["cli", "argument-parsing"]),
    ("zap", &["logging", "observability"]),
    ("grpc", &["grpc", "rpc"]),
];

// ── Extraction functions ─────────────────────────────────────────────────

/// Extract concept tags from a Cargo.toml file.
pub fn extract_cargo_concepts(content: &str) -> Vec<SelectedTag> {
    let table = match content.parse::<toml::Table>() {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut dep_names: Vec<String> = Vec::new();

    // [dependencies]
    if let Some(deps) = table.get("dependencies").and_then(|d| d.as_table()) {
        dep_names.extend(deps.keys().cloned());
    }
    // [dev-dependencies]
    if let Some(deps) = table.get("dev-dependencies").and_then(|d| d.as_table()) {
        dep_names.extend(deps.keys().cloned());
    }
    // [build-dependencies]
    if let Some(deps) = table.get("build-dependencies").and_then(|d| d.as_table()) {
        dep_names.extend(deps.keys().cloned());
    }

    map_deps_to_concepts(&dep_names)
}

/// Extract concept tags from a package.json file.
pub fn extract_npm_concepts(content: &str) -> Vec<SelectedTag> {
    let parsed: serde_json::Value = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut dep_names: Vec<String> = Vec::new();

    for field in &["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(deps) = parsed.get(field).and_then(|d| d.as_object()) {
            dep_names.extend(deps.keys().cloned());
        }
    }

    map_deps_to_concepts(&dep_names)
}

/// Extract concept tags from a requirements.txt file.
pub fn extract_pip_concepts(content: &str) -> Vec<SelectedTag> {
    let dep_names: Vec<String> = content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#') && !l.starts_with('-'))
        .filter_map(|l| {
            // Extract package name before version specifier
            let name = l
                .split(|c: char| c == '=' || c == '>' || c == '<' || c == '!' || c == '[')
                .next()?
                .trim()
                .to_string();
            if name.is_empty() {
                None
            } else {
                Some(name)
            }
        })
        .collect();

    map_deps_to_concepts(&dep_names)
}

/// Extract concept tags from a go.mod file.
pub fn extract_gomod_concepts(content: &str) -> Vec<SelectedTag> {
    let mut dep_names: Vec<String> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        // Module paths like "github.com/gin-gonic/gin v1.9.0"
        if trimmed.starts_with("require") || trimmed.contains('/') {
            if let Some(path) = trimmed.split_whitespace().next() {
                // Use the last segment of the module path
                if let Some(name) = path.rsplit('/').next() {
                    dep_names.push(name.to_string());
                }
            }
        }
    }

    map_deps_to_concepts(&dep_names)
}

/// Map dependency names to concept tags using the built-in concept map.
fn map_deps_to_concepts(dep_names: &[String]) -> Vec<SelectedTag> {
    let mut seen = std::collections::HashSet::new();
    let mut tags = Vec::new();

    for dep_name in dep_names {
        let normalized = dep_name.to_lowercase().replace('_', "-");
        for &(name, concepts) in CONCEPT_MAP {
            if normalized == name || normalized.starts_with(&format!("{}-", name)) {
                for &concept in concepts {
                    if seen.insert(concept.to_string()) {
                        tags.push(make_tier1_tag(&format!("dep:{}", concept)));
                    }
                }
            }
        }
    }

    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Cargo concept extraction ─────────────────────────────────────

    #[test]
    fn test_cargo_concepts() {
        let cargo = r#"
[package]
name = "my-project"

[dependencies]
tokio = { version = "1", features = ["full"] }
serde = "1"
serde_json = "1"
axum = "0.7"
"#;
        let tags = extract_cargo_concepts(cargo);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"dep:async-runtime"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:serialization"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:web-framework"), "Got: {:?}", phrases);
    }

    #[test]
    fn test_cargo_concepts_no_match() {
        let cargo = r#"
[dependencies]
my-internal-crate = "0.1"
"#;
        let tags = extract_cargo_concepts(cargo);
        assert!(tags.is_empty());
    }

    // ── NPM concept extraction ──────────────────────────────────────

    #[test]
    fn test_npm_concepts() {
        let pkg = r#"{
  "dependencies": {
    "react": "^18.0.0",
    "axios": "^1.0.0"
  },
  "devDependencies": {
    "jest": "^29.0.0",
    "typescript": "^5.0.0"
  }
}"#;
        let tags = extract_npm_concepts(pkg);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"dep:ui-framework"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:http-client"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:testing"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:type-system"), "Got: {:?}", phrases);
    }

    // ── Pip concept extraction ───────────────────────────────────────

    #[test]
    fn test_pip_concepts() {
        let reqs = "pandas>=1.5.0\nnumpy\nscikit-learn==1.2.0\n# comment\nflask>=2.0\n";
        let tags = extract_pip_concepts(reqs);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"dep:data-analysis"), "Got: {:?}", phrases);
        assert!(
            phrases.contains(&"dep:numerical-computing"),
            "Got: {:?}",
            phrases
        );
        assert!(
            phrases.contains(&"dep:machine-learning"),
            "Got: {:?}",
            phrases
        );
        assert!(phrases.contains(&"dep:web-framework"), "Got: {:?}", phrases);
    }

    #[test]
    fn test_pip_concepts_empty() {
        let reqs = "# just comments\n";
        let tags = extract_pip_concepts(reqs);
        assert!(tags.is_empty());
    }

    // ── Go mod concept extraction ────────────────────────────────────

    #[test]
    fn test_gomod_concepts() {
        let gomod = r#"
module myproject

go 1.21

require (
    github.com/gin-gonic/gin v1.9.0
    go.uber.org/zap v1.24.0
)
"#;
        let tags = extract_gomod_concepts(gomod);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"dep:web-framework"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:logging"), "Got: {:?}", phrases);
    }

    // ── map_deps_to_concepts ─────────────────────────────────────────

    #[test]
    fn test_map_deps_deduplication() {
        let deps = vec!["actix-web".to_string(), "axum".to_string()];
        let tags = map_deps_to_concepts(&deps);
        let web_count = tags
            .iter()
            .filter(|t| t.phrase == "dep:web-framework")
            .count();
        assert_eq!(web_count, 1, "Duplicate concepts should be deduplicated");
    }
}
