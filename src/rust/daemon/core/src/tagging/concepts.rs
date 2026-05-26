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
    // ── Rust ────────────────────────────────────────────────────────
    ("tokio", &["async-runtime", "concurrency"]),
    ("async-std", &["async-runtime", "concurrency"]),
    ("serde", &["serialization"]),
    ("serde_json", &["json", "serialization"]),
    ("serde_yaml", &["yaml", "serialization"]),
    ("toml", &["toml", "configuration"]),
    ("reqwest", &["http-client", "networking"]),
    ("hyper", &["http", "networking"]),
    ("actix-web", &["web-framework", "http-server"]),
    ("axum", &["web-framework", "http-server"]),
    ("warp", &["web-framework", "http-server"]),
    ("rocket", &["web-framework", "http-server"]),
    ("diesel", &["orm", "database"]),
    ("sqlx", &["database", "sql"]),
    ("sea-orm", &["orm", "database"]),
    ("rusqlite", &["sqlite", "database"]),
    ("clap", &["cli", "argument-parsing"]),
    ("structopt", &["cli", "argument-parsing"]),
    ("tracing", &["observability", "logging"]),
    ("log", &["logging"]),
    ("env-logger", &["logging"]),
    ("tonic", &["grpc", "rpc"]),
    ("prost", &["protobuf", "serialization"]),
    ("rayon", &["parallelism", "concurrency"]),
    ("crossbeam", &["concurrency"]),
    ("qdrant-client", &["vector-database", "search"]),
    ("fastembed", &["embeddings", "ml"]),
    ("anyhow", &["error-handling"]),
    ("thiserror", &["error-handling"]),
    ("rand", &["randomness"]),
    ("regex", &["regex", "text-processing"]),
    ("chrono", &["datetime"]),
    ("time", &["datetime"]),
    ("uuid", &["identifiers"]),
    ("sha2", &["cryptography", "hashing"]),
    ("ring", &["cryptography"]),
    ("rustls", &["tls", "cryptography"]),
    ("tower", &["middleware", "networking"]),
    ("bytes", &["binary", "networking"]),
    ("futures", &["async-runtime"]),
    ("tokio-stream", &["streaming", "async-runtime"]),
    ("nom", &["parsing"]),
    ("pest", &["parsing"]),
    ("tree-sitter", &["parsing", "syntax-analysis"]),
    ("image", &["image-processing"]),
    ("wasm-bindgen", &["webassembly"]),
    ("napi", &["ffi", "node-binding"]),
    ("pyo3", &["ffi", "python-binding"]),
    ("bindgen", &["ffi", "code-generation"]),
    ("proc-macro2", &["metaprogramming"]),
    ("syn", &["metaprogramming", "parsing"]),
    ("quote", &["metaprogramming", "code-generation"]),
    ("criterion", &["benchmarking", "testing"]),
    ("proptest", &["property-testing", "testing"]),
    ("tempfile", &["testing", "filesystem"]),
    ("notify", &["file-watching", "filesystem"]),
    ("walkdir", &["filesystem"]),
    ("glob", &["filesystem"]),
    ("redis", &["cache", "database"]),
    ("lapin", &["message-queue"]),
    ("rdkafka", &["message-queue", "streaming"]),
    ("aws-sdk", &["cloud", "aws"]),
    ("rusoto", &["cloud", "aws"]),
    ("bollard", &["docker", "containers"]),
    ("k8s-openapi", &["kubernetes", "containers"]),
    ("ort", &["ml-inference", "ml"]),
    ("candle", &["deep-learning", "ml"]),
    ("polars", &["data-analysis", "dataframes"]),
    ("arrow", &["data-processing", "columnar"]),
    // ── JavaScript / TypeScript ─────────────────────────────────────
    ("react", &["ui-framework", "frontend"]),
    ("react-dom", &["ui-framework", "frontend"]),
    ("vue", &["ui-framework", "frontend"]),
    ("angular", &["ui-framework", "frontend"]),
    ("svelte", &["ui-framework", "frontend"]),
    ("solid-js", &["ui-framework", "frontend"]),
    ("preact", &["ui-framework", "frontend"]),
    ("express", &["web-framework", "http-server"]),
    ("fastify", &["web-framework", "http-server"]),
    ("koa", &["web-framework", "http-server"]),
    ("hono", &["web-framework", "http-server"]),
    ("next", &["web-framework", "ssr"]),
    ("nuxt", &["web-framework", "ssr"]),
    ("remix", &["web-framework", "ssr"]),
    ("astro", &["web-framework", "ssg"]),
    ("gatsby", &["web-framework", "ssg"]),
    ("jest", &["testing"]),
    ("mocha", &["testing"]),
    ("vitest", &["testing"]),
    ("cypress", &["e2e-testing", "testing"]),
    ("playwright", &["e2e-testing", "testing"]),
    ("webpack", &["bundler", "build-tool"]),
    ("vite", &["bundler", "build-tool"]),
    ("esbuild", &["bundler", "build-tool"]),
    ("rollup", &["bundler", "build-tool"]),
    ("turbo", &["monorepo", "build-tool"]),
    ("typescript", &["type-system"]),
    ("prisma", &["orm", "database"]),
    ("drizzle-orm", &["orm", "database"]),
    ("typeorm", &["orm", "database"]),
    ("sequelize", &["orm", "database"]),
    ("knex", &["query-builder", "database"]),
    ("mongoose", &["orm", "database"]),
    ("axios", &["http-client", "networking"]),
    ("node-fetch", &["http-client", "networking"]),
    ("socket.io", &["websocket", "realtime"]),
    ("ws", &["websocket", "networking"]),
    ("graphql", &["graphql", "api"]),
    ("apollo-server", &["graphql", "api"]),
    ("trpc", &["rpc", "api"]),
    ("zod", &["validation", "schema"]),
    ("joi", &["validation", "schema"]),
    ("yup", &["validation", "schema"]),
    ("tailwindcss", &["css", "styling"]),
    ("styled-components", &["css-in-js", "styling"]),
    ("emotion", &["css-in-js", "styling"]),
    ("redux", &["state-management", "frontend"]),
    ("zustand", &["state-management", "frontend"]),
    ("mobx", &["state-management", "frontend"]),
    ("tanstack-query", &["data-fetching", "frontend"]),
    ("react-query", &["data-fetching", "frontend"]),
    ("swr", &["data-fetching", "frontend"]),
    ("d3", &["data-visualization"]),
    ("chart.js", &["data-visualization"]),
    ("three", &["3d-graphics", "webgl"]),
    ("lodash", &["utility"]),
    ("date-fns", &["datetime"]),
    ("dayjs", &["datetime"]),
    ("moment", &["datetime"]),
    ("uuid", &["identifiers"]),
    ("winston", &["logging"]),
    ("pino", &["logging"]),
    ("bull", &["task-queue", "job-processing"]),
    ("bullmq", &["task-queue", "job-processing"]),
    ("ioredis", &["cache", "database"]),
    ("kafkajs", &["message-queue", "streaming"]),
    ("amqplib", &["message-queue"]),
    ("aws-sdk", &["cloud", "aws"]),
    ("firebase", &["cloud", "baas"]),
    ("supabase", &["cloud", "baas"]),
    ("stripe", &["payments"]),
    ("passport", &["authentication"]),
    ("jsonwebtoken", &["authentication", "jwt"]),
    ("bcrypt", &["cryptography", "authentication"]),
    ("sharp", &["image-processing"]),
    ("puppeteer", &["browser-automation"]),
    ("cheerio", &["web-scraping"]),
    ("electron", &["desktop-app"]),
    ("tauri", &["desktop-app"]),
    ("react-native", &["mobile", "frontend"]),
    ("expo", &["mobile", "frontend"]),
    ("storybook", &["component-docs", "frontend"]),
    ("eslint", &["linting", "code-quality"]),
    ("prettier", &["formatting", "code-quality"]),
    // ── Python ──────────────────────────────────────────────────────
    ("django", &["web-framework", "http-server"]),
    ("flask", &["web-framework", "http-server"]),
    ("fastapi", &["web-framework", "http-server"]),
    ("starlette", &["web-framework", "http-server"]),
    ("tornado", &["web-framework", "http-server"]),
    ("aiohttp", &["http-client", "async", "http-server"]),
    ("httpx", &["http-client", "networking"]),
    ("pandas", &["data-analysis", "dataframes"]),
    ("polars", &["data-analysis", "dataframes"]),
    ("numpy", &["numerical-computing"]),
    ("scipy", &["scientific-computing"]),
    ("matplotlib", &["data-visualization"]),
    ("seaborn", &["data-visualization"]),
    ("plotly", &["data-visualization"]),
    ("scikit-learn", &["machine-learning"]),
    ("xgboost", &["machine-learning"]),
    ("lightgbm", &["machine-learning"]),
    ("tensorflow", &["deep-learning", "ml"]),
    ("keras", &["deep-learning", "ml"]),
    ("pytorch", &["deep-learning", "ml"]),
    ("torch", &["deep-learning", "ml"]),
    ("transformers", &["nlp", "deep-learning"]),
    ("langchain", &["llm", "ai-agents"]),
    ("openai", &["llm", "ai-api"]),
    ("anthropic", &["llm", "ai-api"]),
    ("sqlalchemy", &["orm", "database"]),
    ("alembic", &["database-migration", "database"]),
    ("psycopg2", &["postgresql", "database"]),
    ("pymongo", &["mongodb", "database"]),
    ("redis", &["cache", "database"]),
    ("pytest", &["testing"]),
    ("unittest", &["testing"]),
    ("hypothesis", &["property-testing", "testing"]),
    ("requests", &["http-client", "networking"]),
    ("beautifulsoup4", &["web-scraping"]),
    ("scrapy", &["web-scraping"]),
    ("celery", &["task-queue", "async"]),
    ("rq", &["task-queue"]),
    ("pydantic", &["validation", "schema"]),
    ("marshmallow", &["serialization", "validation"]),
    ("click", &["cli", "argument-parsing"]),
    ("typer", &["cli", "argument-parsing"]),
    ("argparse", &["cli", "argument-parsing"]),
    ("boto3", &["cloud", "aws"]),
    ("google-cloud", &["cloud", "gcp"]),
    ("pillow", &["image-processing"]),
    ("opencv-python", &["computer-vision", "image-processing"]),
    ("spacy", &["nlp", "text-processing"]),
    ("nltk", &["nlp", "text-processing"]),
    ("black", &["formatting", "code-quality"]),
    ("mypy", &["type-checking", "code-quality"]),
    ("ruff", &["linting", "code-quality"]),
    ("streamlit", &["dashboard", "data-visualization"]),
    ("gradio", &["ml-demo", "ui"]),
    ("dask", &["distributed-computing", "data-processing"]),
    ("airflow", &["workflow-orchestration", "data-engineering"]),
    ("prefect", &["workflow-orchestration", "data-engineering"]),
    ("docker", &["docker", "containers"]),
    ("kubernetes", &["kubernetes", "containers"]),
    ("cryptography", &["cryptography"]),
    // ── Go ──────────────────────────────────────────────────────────
    ("gin", &["web-framework", "http-server"]),
    ("echo", &["web-framework", "http-server"]),
    ("fiber", &["web-framework", "http-server"]),
    ("chi", &["web-framework", "http-server"]),
    ("gorm", &["orm", "database"]),
    ("ent", &["orm", "database"]),
    ("cobra", &["cli", "argument-parsing"]),
    ("viper", &["configuration"]),
    ("zap", &["logging", "observability"]),
    ("logrus", &["logging"]),
    ("grpc", &["grpc", "rpc"]),
    ("protobuf", &["protobuf", "serialization"]),
    ("testify", &["testing"]),
    ("gomock", &["testing", "mocking"]),
    ("wire", &["dependency-injection"]),
    ("fx", &["dependency-injection"]),
    ("sarama", &["message-queue", "streaming"]),
    ("nats", &["message-queue"]),
    ("go-redis", &["cache", "database"]),
    ("pgx", &["postgresql", "database"]),
    ("mongo-driver", &["mongodb", "database"]),
    ("aws-sdk-go", &["cloud", "aws"]),
    ("mux", &["http-router"]),
    ("gorilla", &["websocket", "http"]),
    ("prometheus", &["metrics", "observability"]),
    ("otel", &["observability", "tracing"]),
    ("jwt-go", &["authentication", "jwt"]),
    ("casbin", &["authorization", "access-control"]),
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

// ── 4-step concept normalization pipeline ───────────────────────────────

/// Result of concept normalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedConcept {
    /// Exact match from CONCEPT_MAP.
    Mapped(Vec<String>),
    /// No match found — raw library prefix.
    Raw(String),
}

/// Normalize a dependency name to concept(s) using the 4-step pipeline:
/// 1. Exact match against CONCEPT_MAP
/// 2. Prefix match (e.g. "serde-derive" matches "serde")
/// 3. (Reserved for embedding similarity — returns None, caller falls through)
/// 4. Raw `library:{name}` fallback
pub fn normalize_dependency(dep_name: &str) -> NormalizedConcept {
    let normalized = dep_name.to_lowercase().replace('_', "-");

    // Step 1: exact match (normalize map keys the same way)
    for &(name, concepts) in CONCEPT_MAP {
        let norm_name = name.replace('_', "-");
        if normalized == norm_name {
            return NormalizedConcept::Mapped(concepts.iter().map(|s| s.to_string()).collect());
        }
    }

    // Step 2: prefix match (e.g. "serde-derive" → serde's concepts)
    for &(name, concepts) in CONCEPT_MAP {
        let norm_name = name.replace('_', "-");
        if normalized.starts_with(&format!("{}-", norm_name)) {
            return NormalizedConcept::Mapped(concepts.iter().map(|s| s.to_string()).collect());
        }
    }

    // Step 3: embedding similarity — deferred, requires EmbeddingGenerator
    // Callers can implement this step externally before falling through to step 4.

    // Step 4: raw fallback
    NormalizedConcept::Raw(format!("library:{}", normalized))
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

    // ── normalize_dependency pipeline ───────────────────────────────

    #[test]
    fn test_normalize_exact_match() {
        match normalize_dependency("reqwest") {
            NormalizedConcept::Mapped(concepts) => {
                assert!(concepts.contains(&"http-client".to_string()));
                assert!(concepts.contains(&"networking".to_string()));
            }
            other => panic!("Expected Mapped, got {:?}", other),
        }
    }

    #[test]
    fn test_normalize_prefix_match() {
        match normalize_dependency("serde-derive") {
            NormalizedConcept::Mapped(concepts) => {
                assert!(concepts.contains(&"serialization".to_string()));
            }
            other => panic!("Expected Mapped, got {:?}", other),
        }
    }

    #[test]
    fn test_normalize_underscore_handling() {
        match normalize_dependency("serde_json") {
            NormalizedConcept::Mapped(concepts) => {
                assert!(concepts.contains(&"json".to_string()));
            }
            other => panic!("Expected Mapped, got {:?}", other),
        }
    }

    #[test]
    fn test_normalize_raw_fallback() {
        match normalize_dependency("my-internal-lib") {
            NormalizedConcept::Raw(label) => {
                assert_eq!(label, "library:my-internal-lib");
            }
            other => panic!("Expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn test_normalize_case_insensitive() {
        match normalize_dependency("Django") {
            NormalizedConcept::Mapped(concepts) => {
                assert!(concepts.contains(&"web-framework".to_string()));
            }
            other => panic!("Expected Mapped, got {:?}", other),
        }
    }

    #[test]
    fn test_concept_map_coverage() {
        assert!(
            CONCEPT_MAP.len() >= 200,
            "CONCEPT_MAP should have >=200 entries, got {}",
            CONCEPT_MAP.len()
        );
    }
}
