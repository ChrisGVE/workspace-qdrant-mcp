use super::*;
use sqlx::sqlite::SqlitePoolOptions;

// ---- Parser tests ----------------------------------------------------------

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
    assert_eq!(
        normalize_python_dep("Flask>=2.0"),
        Some("flask".to_string())
    );
    assert_eq!(
        normalize_python_dep("my_package"),
        Some("my-package".to_string())
    );
    assert_eq!(
        normalize_python_dep("pandas[sql]>=1.5"),
        Some("pandas".to_string())
    );
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

// ---- Similarity tests ------------------------------------------------------

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
    let a: HashSet<String> = ["serde", "tokio", "anyhow"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let b: HashSet<String> = ["serde", "tokio", "reqwest"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    // Intersection: 2, Union: 4 -> 0.5
    assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-6);
}

#[test]
fn test_jaccard_empty() {
    let a: HashSet<String> = HashSet::new();
    let b: HashSet<String> = HashSet::new();
    assert_eq!(jaccard_similarity(&a, &b), 0.0);
}

// ---- SQLite integration tests ----------------------------------------------

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
    sqlx::query(schema::CREATE_PROJECT_GROUPS_SQL)
        .execute(&pool)
        .await
        .unwrap();
    for idx in schema::CREATE_PROJECT_GROUPS_INDEXES_SQL {
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

    // proj-b: serde, tokio, reqwest -> Jaccard = 2/4 = 0.5
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
    let members = schema::get_group_members(&pool, "proj-a").await.unwrap();
    assert!(members.contains(&"proj-b".to_string()));
}

#[tokio::test]
async fn test_compute_groups_dissimilar_projects() {
    let pool = setup_pool().await;

    store_dependencies(
        &pool,
        "proj-a",
        &[
            ("serde".into(), "rust".into()),
            ("tokio".into(), "rust".into()),
        ],
    )
    .await
    .unwrap();

    store_dependencies(
        &pool,
        "proj-b",
        &[
            ("express".into(), "npm".into()),
            ("lodash".into(), "npm".into()),
        ],
    )
    .await
    .unwrap();

    let groups = compute_dependency_groups(&pool, Some(0.3)).await.unwrap();
    assert_eq!(groups, 0);
}
