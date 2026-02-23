/// Tier 1 automated tagging — zero-cost heuristics.
///
/// Extracts tags from metadata without ML inference:
/// 1. **Path-derived tags**: directory names → kebab-case tags
/// 2. **PDF metadata tags**: subject + keywords from PDF Info dict
/// 3. **Dependency-derived concepts**: map dependency names → concept tags

use std::path::Path;

use crate::keyword_extraction::tag_selector::{SelectedTag, TagType};

// ─── Structural directories to skip ──────────────────────────────────────

/// Directories that are structural (not semantically meaningful).
const SKIP_DIRS: &[&str] = &[
    "src", "lib", "test", "tests", "spec", "specs", "utils", "util",
    "docs", "doc", "build", "dist", "out", "bin", "target", "vendor",
    "node_modules", ".git", ".github", ".vscode", "assets", "static",
    "public", "private", "internal", "pkg", "cmd", "include",
];

// ─── Path-derived tags ──────────────────────────────────────────────────

/// Extract tags from directory names in a file path.
///
/// Normalizes directory names to kebab-case and filters out structural
/// directories (src/, lib/, test/, etc.) that carry no semantic meaning.
///
/// # Examples
/// ```ignore
/// // "computer_science/design_patterns/observer.pdf"
/// // → ["computer-science", "design-patterns"]
/// ```
pub fn extract_path_tags(file_path: &Path) -> Vec<SelectedTag> {
    let mut tags = Vec::new();

    for component in file_path.components() {
        let name = match component {
            std::path::Component::Normal(os) => os.to_string_lossy(),
            _ => continue,
        };

        let name_str = name.as_ref();

        // Skip the filename itself (last component)
        if file_path.file_name().map(|f| f == component.as_os_str()).unwrap_or(false) {
            continue;
        }

        // Skip structural directories
        if SKIP_DIRS.iter().any(|&d| d.eq_ignore_ascii_case(name_str)) {
            continue;
        }

        // Skip hidden directories
        if name_str.starts_with('.') {
            continue;
        }

        // Skip single-character directories
        if name_str.len() <= 1 {
            continue;
        }

        let tag = normalize_to_kebab(name_str);
        if tag.len() >= 2 {
            tags.push(make_tier1_tag(&format!("path:{}", tag)));
        }
    }

    tags
}

/// Normalize a string to kebab-case.
///
/// Handles underscores, camelCase, PascalCase, and spaces.
fn normalize_to_kebab(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut prev_was_separator = false;

    for (i, ch) in input.chars().enumerate() {
        if ch == '_' || ch == ' ' || ch == '-' {
            if !result.is_empty() && !prev_was_separator {
                result.push('-');
                prev_was_separator = true;
            }
        } else if ch.is_uppercase() && i > 0 {
            // camelCase boundary
            let prev = input.as_bytes().get(i.saturating_sub(1)).copied();
            if let Some(p) = prev {
                if (p as char).is_lowercase() && !prev_was_separator {
                    result.push('-');
                }
            }
            result.push(ch.to_ascii_lowercase());
            prev_was_separator = false;
        } else if ch.is_alphanumeric() {
            result.push(ch.to_ascii_lowercase());
            prev_was_separator = false;
        }
    }

    // Trim trailing separator
    if result.ends_with('-') {
        result.pop();
    }

    result
}

// ─── PDF metadata tags ──────────────────────────────────────────────────

/// Metadata extracted from a PDF Info dictionary.
#[derive(Debug, Default)]
pub struct PdfMetadataTags {
    /// Tags from the /Keywords field
    pub keyword_tags: Vec<String>,
    /// Tag from the /Subject field
    pub subject_tag: Option<String>,
    /// Author (stored as metadata, not a tag)
    pub author: Option<String>,
}

/// Extract tags from PDF metadata (subject, keywords).
///
/// Uses lopdf to read the Info dictionary. Returns empty result
/// for non-PDF files or files without metadata.
pub fn extract_pdf_metadata_tags(file_path: &Path) -> PdfMetadataTags {
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if !ext.eq_ignore_ascii_case("pdf") {
        return PdfMetadataTags::default();
    }

    let doc = match lopdf::Document::load(file_path) {
        Ok(d) => d,
        Err(_) => return PdfMetadataTags::default(),
    };

    let info_ref = match doc.trailer.get(b"Info") {
        Ok(lopdf::Object::Reference(r)) => *r,
        _ => return PdfMetadataTags::default(),
    };

    let info_dict = match doc.get_object(info_ref) {
        Ok(lopdf::Object::Dictionary(d)) => d,
        _ => return PdfMetadataTags::default(),
    };

    let mut result = PdfMetadataTags::default();

    // Extract /Keywords → split on comma or semicolon
    if let Ok(obj) = info_dict.get(b"Keywords") {
        if let Some(text) = pdf_object_to_string(obj) {
            result.keyword_tags = text
                .split(|c: char| c == ',' || c == ';')
                .map(|s| normalize_to_kebab(s.trim()))
                .filter(|s| s.len() >= 2)
                .collect();
        }
    }

    // Extract /Subject → single tag
    if let Ok(obj) = info_dict.get(b"Subject") {
        if let Some(text) = pdf_object_to_string(obj) {
            let tag = normalize_to_kebab(text.trim());
            if tag.len() >= 2 {
                result.subject_tag = Some(tag);
            }
        }
    }

    // Extract /Author (informational, not a tag)
    if let Ok(obj) = info_dict.get(b"Author") {
        if let Some(text) = pdf_object_to_string(obj) {
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                result.author = Some(trimmed);
            }
        }
    }

    result
}

/// Convert PDF metadata tags into SelectedTag list.
pub fn pdf_metadata_to_tags(meta: &PdfMetadataTags) -> Vec<SelectedTag> {
    let mut tags = Vec::new();

    for kw in &meta.keyword_tags {
        tags.push(make_tier1_tag(&format!("pdf-keyword:{}", kw)));
    }

    if let Some(ref subj) = meta.subject_tag {
        tags.push(make_tier1_tag(&format!("pdf-subject:{}", subj)));
    }

    tags
}

/// Extract a string from a lopdf Object.
fn pdf_object_to_string(obj: &lopdf::Object) -> Option<String> {
    match obj {
        lopdf::Object::String(bytes, _) => {
            let s = String::from_utf8_lossy(bytes).trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        }
        _ => None,
    }
}

// ─── Dependency-derived concepts ────────────────────────────────────────

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
            let name = l.split(|c: char| c == '=' || c == '>' || c == '<' || c == '!' || c == '[')
                .next()?
                .trim()
                .to_string();
            if name.is_empty() { None } else { Some(name) }
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

// ─── Aggregation ────────────────────────────────────────────────────────

/// Aggregate all Tier 1 tags for a file.
///
/// Combines path-derived, PDF metadata, and dependency-derived tags.
/// Deduplicates by tag phrase.
pub fn extract_tier1_tags(
    file_path: &Path,
    manifest_content: Option<(&str, &str)>, // (format, content)
) -> Vec<SelectedTag> {
    let mut all_tags = Vec::new();

    // 1. Path-derived tags
    all_tags.extend(extract_path_tags(file_path));

    // 2. PDF metadata tags
    let pdf_meta = extract_pdf_metadata_tags(file_path);
    all_tags.extend(pdf_metadata_to_tags(&pdf_meta));

    // 3. Dependency-derived concepts
    if let Some((format, content)) = manifest_content {
        let dep_tags = match format {
            "cargo" => extract_cargo_concepts(content),
            "npm" => extract_npm_concepts(content),
            "pip" => extract_pip_concepts(content),
            "gomod" => extract_gomod_concepts(content),
            _ => Vec::new(),
        };
        all_tags.extend(dep_tags);
    }

    // Deduplicate by phrase
    let mut seen = std::collections::HashSet::new();
    all_tags.retain(|t| seen.insert(t.phrase.clone()));

    all_tags
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn make_tier1_tag(phrase: &str) -> SelectedTag {
    SelectedTag {
        phrase: phrase.to_string(),
        tag_type: TagType::Structural,
        score: 0.8, // high confidence for metadata-derived
        diversity_score: 1.0,
        semantic_score: 0.0, // not semantically scored
        ngram_size: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ─── normalize_to_kebab ─────────────────────────────────────────

    #[test]
    fn test_normalize_snake_case() {
        assert_eq!(normalize_to_kebab("design_patterns"), "design-patterns");
    }

    #[test]
    fn test_normalize_camel_case() {
        assert_eq!(normalize_to_kebab("designPatterns"), "design-patterns");
    }

    #[test]
    fn test_normalize_pascal_case() {
        assert_eq!(normalize_to_kebab("DesignPatterns"), "design-patterns");
    }

    #[test]
    fn test_normalize_spaces() {
        assert_eq!(normalize_to_kebab("design patterns"), "design-patterns");
    }

    #[test]
    fn test_normalize_already_kebab() {
        assert_eq!(normalize_to_kebab("design-patterns"), "design-patterns");
    }

    #[test]
    fn test_normalize_mixed() {
        assert_eq!(normalize_to_kebab("Computer_Science"), "computer-science");
    }

    // ─── extract_path_tags ──────────────────────────────────────────

    #[test]
    fn test_path_tags_basic() {
        let path = PathBuf::from("computer_science/design_patterns/observer.pdf");
        let tags = extract_path_tags(&path);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"path:computer-science"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"path:design-patterns"), "Got: {:?}", phrases);
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn test_path_tags_skip_structural() {
        let path = PathBuf::from("src/lib/utils/helpers.rs");
        let tags = extract_path_tags(&path);
        assert!(tags.is_empty(), "Structural dirs should be skipped: {:?}",
            tags.iter().map(|t| &t.phrase).collect::<Vec<_>>());
    }

    #[test]
    fn test_path_tags_skip_hidden() {
        let path = PathBuf::from(".hidden/deep/file.txt");
        let tags = extract_path_tags(&path);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(!phrases.iter().any(|p| p.contains("hidden")));
    }

    #[test]
    fn test_path_tags_mixed() {
        let path = PathBuf::from("machine_learning/src/models/transformer.py");
        let tags = extract_path_tags(&path);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"path:machine-learning"));
        assert!(phrases.contains(&"path:models"));
        // "src" should be skipped
        assert!(!phrases.iter().any(|p| p.contains("src")));
    }

    // ─── PDF metadata (unit tests without real files) ───────────────

    #[test]
    fn test_pdf_metadata_non_pdf() {
        let path = PathBuf::from("test.txt");
        let meta = extract_pdf_metadata_tags(&path);
        assert!(meta.keyword_tags.is_empty());
        assert!(meta.subject_tag.is_none());
    }

    #[test]
    fn test_pdf_metadata_to_tags_empty() {
        let meta = PdfMetadataTags::default();
        let tags = pdf_metadata_to_tags(&meta);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_pdf_metadata_to_tags_populated() {
        let meta = PdfMetadataTags {
            keyword_tags: vec!["machine-learning".to_string(), "neural-networks".to_string()],
            subject_tag: Some("artificial-intelligence".to_string()),
            author: Some("John Doe".to_string()),
        };
        let tags = pdf_metadata_to_tags(&meta);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert_eq!(phrases.len(), 3);
        assert!(phrases.contains(&"pdf-keyword:machine-learning"));
        assert!(phrases.contains(&"pdf-keyword:neural-networks"));
        assert!(phrases.contains(&"pdf-subject:artificial-intelligence"));
    }

    // ─── Cargo concept extraction ───────────────────────────────────

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

    // ─── NPM concept extraction ────────────────────────────────────

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

    // ─── Pip concept extraction ─────────────────────────────────────

    #[test]
    fn test_pip_concepts() {
        let reqs = "pandas>=1.5.0\nnumpy\nscikit-learn==1.2.0\n# comment\nflask>=2.0\n";
        let tags = extract_pip_concepts(reqs);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"dep:data-analysis"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:numerical-computing"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:machine-learning"), "Got: {:?}", phrases);
        assert!(phrases.contains(&"dep:web-framework"), "Got: {:?}", phrases);
    }

    #[test]
    fn test_pip_concepts_empty() {
        let reqs = "# just comments\n";
        let tags = extract_pip_concepts(reqs);
        assert!(tags.is_empty());
    }

    // ─── Go mod concept extraction ──────────────────────────────────

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

    // ─── map_deps_to_concepts ───────────────────────────────────────

    #[test]
    fn test_map_deps_deduplication() {
        let deps = vec![
            "actix-web".to_string(),
            "axum".to_string(),
        ];
        let tags = map_deps_to_concepts(&deps);
        let web_count = tags.iter().filter(|t| t.phrase == "dep:web-framework").count();
        assert_eq!(web_count, 1, "Duplicate concepts should be deduplicated");
    }

    // ─── extract_tier1_tags aggregation ─────────────────────────────

    #[test]
    fn test_tier1_aggregation() {
        let path = PathBuf::from("machine_learning/models/train.py");
        let tags = extract_tier1_tags(&path, None);
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"path:machine-learning"));
        assert!(phrases.contains(&"path:models"));
    }

    #[test]
    fn test_tier1_with_manifest() {
        let path = PathBuf::from("project/Cargo.toml");
        let cargo = r#"
[dependencies]
tokio = "1"
"#;
        let tags = extract_tier1_tags(&path, Some(("cargo", cargo)));
        let phrases: Vec<&str> = tags.iter().map(|t| t.phrase.as_str()).collect();
        assert!(phrases.contains(&"dep:async-runtime"));
    }

    #[test]
    fn test_tier1_deduplication() {
        let path = PathBuf::from("test.txt");
        // No sources → no duplicates possible, just verify empty is fine
        let tags = extract_tier1_tags(&path, None);
        let mut phrases: Vec<String> = tags.iter().map(|t| t.phrase.clone()).collect();
        let original_len = phrases.len();
        phrases.sort();
        phrases.dedup();
        assert_eq!(phrases.len(), original_len, "Should have no duplicates");
    }
}
