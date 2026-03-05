/// Automated project grouping by shared dependencies.
///
/// Parses dependency manifests (Cargo.toml, package.json, pyproject.toml,
/// requirements.txt, go.mod) and groups projects with Jaccard similarity
/// >= 0.3 into `project_groups` entries with type `dependency`.
use std::collections::{HashMap, HashSet};
use std::path::Path;

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use super::schema;
use wqm_common::timestamps::now_utc;

// ---- Configuration ---------------------------------------------------------

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

// ---- SQLite schema for dependency cache ------------------------------------

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

// ---- Dependency parsing ----------------------------------------------------

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
            let cleaned = trimmed
                .trim_matches(',')
                .trim_matches('"')
                .trim_matches('\'')
                .trim();
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

// ---- Grouping logic --------------------------------------------------------

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

                schema::add_to_group(pool, &group_id, tenants[i], "dependency", sim).await?;
                schema::add_to_group(pool, &group_id, tenants[j], "dependency", sim).await?;

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
#[path = "dependency_tests.rs"]
mod tests;
