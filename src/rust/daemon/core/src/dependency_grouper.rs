/// Automated project grouping by shared dependencies.
///
/// Parses dependency manifests (Cargo.toml, package.json, pyproject.toml,
/// requirements.txt, go.mod) and groups projects with Jaccard similarity
/// >= 0.3 into `project_groups` entries with type `dependency`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use crate::project_groups_schema;
use wqm_common::timestamps::now_utc;

// ─── Configuration ──────────────────────────────────────────────────────

/// Minimum Jaccard similarity to group two projects.
const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.3;

/// Dependency manifest filenames we recognize.
const DEPENDENCY_FILES: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
];

// ─── SQLite schema for dependency cache ─────────────────────────────────

/// SQL to create the project_dependencies table (part of schema v24).
pub const CREATE_PROJECT_DEPENDENCIES_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS project_dependencies (
    tenant_id TEXT NOT NULL,
    dependency_name TEXT NOT NULL,
    ecosystem TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, dependency_name)
)
"#;

/// SQL to create indexes on project_dependencies.
pub const CREATE_PROJECT_DEPENDENCIES_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_proj_deps_tenant ON project_dependencies(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_proj_deps_name ON project_dependencies(dependency_name)",
];

// ─── Dependency parsing ─────────────────────────────────────────────────

/// Check if a file path is a dependency manifest.
pub fn is_dependency_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|name| DEPENDENCY_FILES.contains(&name))
        .unwrap_or(false)
}

/// Parse dependencies from a file's content based on its filename.
pub fn parse_dependencies(filename: &str, content: &str) -> Vec<(String, String)> {
    match filename {
        "Cargo.toml" => parse_cargo_toml(content),
        "package.json" => parse_package_json(content),
        "pyproject.toml" => parse_pyproject_toml(content),
        "requirements.txt" => parse_requirements_txt(content),
        "go.mod" => parse_go_mod(content),
        _ => Vec::new(),
    }
}

/// Parse Cargo.toml dependencies.
fn parse_cargo_toml(content: &str) -> Vec<(String, String)> {
    let mut deps = Vec::new();
    let mut in_deps_section = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Track section headers
        if trimmed.starts_with('[') {
            in_deps_section = trimmed == "[dependencies]"
                || trimmed == "[dev-dependencies]"
                || trimmed == "[build-dependencies]"
                || trimmed.starts_with("[dependencies.")
                || trimmed.starts_with("[dev-dependencies.")
                || trimmed.starts_with("[build-dependencies.");
            continue;
        }

        if !in_deps_section {
            continue;
        }

        // Parse "name = ..." or "name = { version = ... }"
        if let Some(eq_pos) = trimmed.find('=') {
            let name = trimmed[..eq_pos].trim().trim_matches('"');
            if !name.is_empty() && !name.contains(' ') {
                deps.push((name.to_string(), "rust".to_string()));
            }
        }
    }

    deps
}

/// Parse package.json dependencies.
fn parse_package_json(content: &str) -> Vec<(String, String)> {
    let mut deps = Vec::new();

    let parsed: serde_json::Value = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(_) => return deps,
    };

    for section in &["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(obj) = parsed.get(section).and_then(|v| v.as_object()) {
            for key in obj.keys() {
                deps.push((key.clone(), "npm".to_string()));
            }
        }
    }

    deps
}

/// Parse pyproject.toml dependencies.
fn parse_pyproject_toml(content: &str) -> Vec<(String, String)> {
    let mut deps = Vec::new();
    let mut in_deps = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed == "dependencies = [" || trimmed.starts_with("dependencies = [") {
            in_deps = true;
            // Handle inline list on same line
            if let Some(rest) = trimmed.strip_prefix("dependencies = [") {
                for dep in extract_quoted_strings(rest) {
                    if let Some(name) = normalize_python_dep(&dep) {
                        deps.push((name, "python".to_string()));
                    }
                }
            }
            continue;
        }

        if in_deps {
            if trimmed == "]" {
                in_deps = false;
                continue;
            }
            let cleaned = trimmed.trim_matches(',').trim_matches('"').trim_matches('\'').trim();
            if let Some(name) = normalize_python_dep(cleaned) {
                deps.push((name, "python".to_string()));
            }
        }
    }

    deps
}

/// Parse requirements.txt dependencies.
fn parse_requirements_txt(content: &str) -> Vec<(String, String)> {
    let mut deps = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('-') {
            continue;
        }
        if let Some(name) = normalize_python_dep(trimmed) {
            deps.push((name, "python".to_string()));
        }
    }

    deps
}

/// Parse go.mod dependencies.
fn parse_go_mod(content: &str) -> Vec<(String, String)> {
    let mut deps = Vec::new();
    let mut in_require = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed == "require (" {
            in_require = true;
            continue;
        }
        if trimmed == ")" {
            in_require = false;
            continue;
        }

        if in_require {
            // Lines like: github.com/foo/bar v1.2.3
            if let Some(module) = trimmed.split_whitespace().next() {
                if module.contains('/') && !module.starts_with("//") {
                    deps.push((module.to_string(), "go".to_string()));
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("require ") {
            // Single-line: require github.com/foo/bar v1.2.3
            if let Some(module) = rest.split_whitespace().next() {
                if module.contains('/') {
                    deps.push((module.to_string(), "go".to_string()));
                }
            }
        }
    }

    deps
}

/// Normalize a Python dependency string (strip version specs, extras).
fn normalize_python_dep(dep: &str) -> Option<String> {
    if dep.is_empty() {
        return None;
    }
    // Strip version specifiers: >=, ==, ~=, !=, <, >, [extras]
    let name = dep
        .split(&['>', '<', '=', '~', '!', '[', ';'][..])
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .replace('_', "-");

    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Extract quoted strings from a line fragment.
fn extract_quoted_strings(s: &str) -> Vec<String> {
    let mut results = Vec::new();
    let mut in_quote = false;
    let mut current = String::new();

    for ch in s.chars() {
        if ch == '"' || ch == '\'' {
            if in_quote {
                results.push(current.clone());
                current.clear();
            }
            in_quote = !in_quote;
        } else if in_quote {
            current.push(ch);
        }
    }

    results
}

// ─── Grouping logic ─────────────────────────────────────────────────────

/// Compute Jaccard similarity between two dependency sets.
pub fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Store parsed dependencies for a project in SQLite.
pub async fn store_dependencies(
    pool: &SqlitePool,
    tenant_id: &str,
    deps: &[(String, String)],
) -> Result<(), sqlx::Error> {
    let now = now_utc();

    // Clear existing deps for this tenant
    sqlx::query("DELETE FROM project_dependencies WHERE tenant_id = ?")
        .bind(tenant_id)
        .execute(pool)
        .await?;

    // Insert new deps
    for (name, ecosystem) in deps {
        sqlx::query(
            r#"
            INSERT OR IGNORE INTO project_dependencies
                (tenant_id, dependency_name, ecosystem, updated_at)
            VALUES (?, ?, ?, ?)
            "#,
        )
        .bind(tenant_id)
        .bind(name)
        .bind(ecosystem)
        .bind(&now)
        .execute(pool)
        .await?;
    }

    debug!(tenant_id, count = deps.len(), "Stored project dependencies");
    Ok(())
}

/// Load all dependency sets from SQLite, grouped by tenant.
pub async fn load_all_dependency_sets(
    pool: &SqlitePool,
) -> Result<HashMap<String, HashSet<String>>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT tenant_id, dependency_name FROM project_dependencies ORDER BY tenant_id",
    )
    .fetch_all(pool)
    .await?;

    let mut sets: HashMap<String, HashSet<String>> = HashMap::new();
    for row in rows {
        let tenant: String = row.get("tenant_id");
        let dep: String = row.get("dependency_name");
        sets.entry(tenant).or_default().insert(dep);
    }

    Ok(sets)
}

/// Recompute dependency-based groups for all projects.
///
/// Compares every pair of projects' dependency sets, creates groups
/// for pairs exceeding the similarity threshold.
pub async fn compute_dependency_groups(
    pool: &SqlitePool,
    threshold: Option<f64>,
) -> Result<usize, sqlx::Error> {
    let threshold = threshold.unwrap_or(DEFAULT_SIMILARITY_THRESHOLD);
    let dep_sets = load_all_dependency_sets(pool).await?;

    let tenants: Vec<&String> = dep_sets.keys().collect();
    let mut groups_created = 0;

    // Clear existing dependency groups
    sqlx::query("DELETE FROM project_groups WHERE group_type = 'dependency'")
        .execute(pool)
        .await?;

    // Compare all pairs
    for i in 0..tenants.len() {
        for j in (i + 1)..tenants.len() {
            let a = &dep_sets[tenants[i]];
            let b = &dep_sets[tenants[j]];
            let sim = jaccard_similarity(a, b);

            if sim >= threshold {
                // Generate deterministic group_id from sorted tenant pair
                let mut pair = vec![tenants[i].as_str(), tenants[j].as_str()];
                pair.sort();
                let group_id = format!("dep:{}+{}", pair[0], pair[1]);

                project_groups_schema::add_to_group(
                    pool, &group_id, tenants[i], "dependency", sim,
                )
                .await?;
                project_groups_schema::add_to_group(
                    pool, &group_id, tenants[j], "dependency", sim,
                )
                .await?;

                debug!(
                    tenant_a = tenants[i].as_str(),
                    tenant_b = tenants[j].as_str(),
                    similarity = sim,
                    "Created dependency group"
                );
                groups_created += 1;
            }
        }
    }

    info!(
        projects = tenants.len(),
        groups = groups_created,
        threshold,
        "Dependency group computation complete"
    );

    Ok(groups_created)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    // ─── Parser tests ───────────────────────────────────────────────────

    #[test]
    fn test_parse_cargo_toml() {
        let content = r#"
[package]
name = "my-crate"

[dependencies]
serde = "1.0"
tokio = { version = "1.35", features = ["full"] }

[dev-dependencies]
tempfile = "3.8"
"#;
        let deps = parse_cargo_toml(content);
        let names: Vec<&str> = deps.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"serde"));
        assert!(names.contains(&"tokio"));
        assert!(names.contains(&"tempfile"));
        assert!(deps.iter().all(|(_, e)| e == "rust"));
    }

    #[test]
    fn test_parse_package_json() {
        let content = r#"
{
  "name": "my-app",
  "dependencies": {
    "express": "^4.18",
    "lodash": "^4.17"
  },
  "devDependencies": {
    "jest": "^29.0"
  }
}
"#;
        let deps = parse_package_json(content);
        let names: Vec<&str> = deps.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"express"));
        assert!(names.contains(&"lodash"));
        assert!(names.contains(&"jest"));
        assert!(deps.iter().all(|(_, e)| e == "npm"));
    }

    #[test]
    fn test_parse_requirements_txt() {
        let content = r#"
# Core deps
flask>=2.0
requests==2.31.0
numpy~=1.24
# Optional
pandas[sql]>=1.5
"#;
        let deps = parse_requirements_txt(content);
        let names: Vec<&str> = deps.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"flask"));
        assert!(names.contains(&"requests"));
        assert!(names.contains(&"numpy"));
        assert!(names.contains(&"pandas"));
    }

    #[test]
    fn test_parse_go_mod() {
        let content = r#"
module github.com/example/myapp

go 1.21

require (
	github.com/gin-gonic/gin v1.9.1
	github.com/go-sql-driver/mysql v1.7.1
)

require github.com/stretchr/testify v1.8.4
"#;
        let deps = parse_go_mod(content);
        let names: Vec<&str> = deps.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"github.com/gin-gonic/gin"));
        assert!(names.contains(&"github.com/go-sql-driver/mysql"));
        assert!(names.contains(&"github.com/stretchr/testify"));
    }

    #[test]
    fn test_parse_pyproject_toml() {
        let content = r#"
[project]
name = "my-project"
dependencies = [
    "flask>=2.0",
    "requests",
]
"#;
        let deps = parse_pyproject_toml(content);
        let names: Vec<&str> = deps.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"flask"));
        assert!(names.contains(&"requests"));
    }

    #[test]
    fn test_normalize_python_dep() {
        assert_eq!(normalize_python_dep("Flask>=2.0"), Some("flask".to_string()));
        assert_eq!(normalize_python_dep("my_package"), Some("my-package".to_string()));
        assert_eq!(normalize_python_dep("pandas[sql]>=1.5"), Some("pandas".to_string()));
        assert_eq!(normalize_python_dep(""), None);
    }

    #[test]
    fn test_is_dependency_file() {
        assert!(is_dependency_file(Path::new("/a/b/Cargo.toml")));
        assert!(is_dependency_file(Path::new("package.json")));
        assert!(is_dependency_file(Path::new("/x/requirements.txt")));
        assert!(!is_dependency_file(Path::new("main.rs")));
        assert!(!is_dependency_file(Path::new("Cargo.lock")));
    }

    // ─── Similarity tests ───────────────────────────────────────────────

    #[test]
    fn test_jaccard_identical() {
        let a: HashSet<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();
        let b = a.clone();
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: HashSet<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a: HashSet<String> = ["serde", "tokio", "anyhow"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["serde", "tokio", "reqwest"].iter().map(|s| s.to_string()).collect();
        // Intersection: 2, Union: 4 → 0.5
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_empty() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = HashSet::new();
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    // ─── SQLite integration tests ───────────────────────────────────────

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        sqlx::query(CREATE_PROJECT_DEPENDENCIES_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for idx in CREATE_PROJECT_DEPENDENCIES_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }
        sqlx::query(crate::project_groups_schema::CREATE_PROJECT_GROUPS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for idx in crate::project_groups_schema::CREATE_PROJECT_GROUPS_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }

        pool
    }

    #[tokio::test]
    async fn test_store_and_load_dependencies() {
        let pool = setup_pool().await;

        let deps = vec![
            ("serde".to_string(), "rust".to_string()),
            ("tokio".to_string(), "rust".to_string()),
        ];
        store_dependencies(&pool, "proj-a", &deps).await.unwrap();

        let all = load_all_dependency_sets(&pool).await.unwrap();
        assert_eq!(all.len(), 1);
        assert!(all["proj-a"].contains("serde"));
        assert!(all["proj-a"].contains("tokio"));
    }

    #[tokio::test]
    async fn test_store_replaces_old_deps() {
        let pool = setup_pool().await;

        let deps1 = vec![("serde".to_string(), "rust".to_string())];
        store_dependencies(&pool, "proj-a", &deps1).await.unwrap();

        let deps2 = vec![("tokio".to_string(), "rust".to_string())];
        store_dependencies(&pool, "proj-a", &deps2).await.unwrap();

        let all = load_all_dependency_sets(&pool).await.unwrap();
        assert_eq!(all["proj-a"].len(), 1);
        assert!(all["proj-a"].contains("tokio"));
        assert!(!all["proj-a"].contains("serde"));
    }

    #[tokio::test]
    async fn test_compute_groups_similar_projects() {
        let pool = setup_pool().await;

        // proj-a: serde, tokio, anyhow
        store_dependencies(
            &pool,
            "proj-a",
            &[
                ("serde".into(), "rust".into()),
                ("tokio".into(), "rust".into()),
                ("anyhow".into(), "rust".into()),
            ],
        )
        .await
        .unwrap();

        // proj-b: serde, tokio, reqwest → Jaccard = 2/4 = 0.5
        store_dependencies(
            &pool,
            "proj-b",
            &[
                ("serde".into(), "rust".into()),
                ("tokio".into(), "rust".into()),
                ("reqwest".into(), "rust".into()),
            ],
        )
        .await
        .unwrap();

        let groups = compute_dependency_groups(&pool, Some(0.3)).await.unwrap();
        assert_eq!(groups, 1);

        // Verify they share a group
        let members = project_groups_schema::get_group_members(&pool, "proj-a")
            .await
            .unwrap();
        assert!(members.contains(&"proj-b".to_string()));
    }

    #[tokio::test]
    async fn test_compute_groups_dissimilar_projects() {
        let pool = setup_pool().await;

        store_dependencies(
            &pool,
            "proj-a",
            &[("serde".into(), "rust".into()), ("tokio".into(), "rust".into())],
        )
        .await
        .unwrap();

        store_dependencies(
            &pool,
            "proj-b",
            &[("express".into(), "npm".into()), ("lodash".into(), "npm".into())],
        )
        .await
        .unwrap();

        let groups = compute_dependency_groups(&pool, Some(0.3)).await.unwrap();
        assert_eq!(groups, 0);
    }
}
