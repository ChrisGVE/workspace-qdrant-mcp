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

#[test]
fn test_is_dependency_file_all_supported() {
    assert!(is_dependency_file(Path::new("Cargo.toml")));
    assert!(is_dependency_file(Path::new("package.json")));
    assert!(is_dependency_file(Path::new("pyproject.toml")));
    assert!(is_dependency_file(Path::new("requirements.txt")));
    assert!(is_dependency_file(Path::new("go.mod")));
}

#[test]
fn test_parse_dependencies_dispatch() {
    // Verify parse_dependencies dispatches to the right parser
    let cargo = parse_dependencies("Cargo.toml", "[dependencies]\nserde = \"1.0\"\n");
    assert_eq!(cargo.len(), 1);
    assert_eq!(cargo[0].1, "rust");

    let npm = parse_dependencies("package.json", r#"{"dependencies":{"react":"^18"}}"#);
    assert_eq!(npm.len(), 1);
    assert_eq!(npm[0].1, "npm");

    let unknown = parse_dependencies("setup.py", "install_requires=['foo']");
    assert!(unknown.is_empty());
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

#[test]
fn test_jaccard_one_empty_one_not() {
    let a: HashSet<String> = ["x"].iter().map(|s| s.to_string()).collect();
    let b: HashSet<String> = HashSet::new();
    assert_eq!(jaccard_similarity(&a, &b), 0.0);
}

#[test]
fn test_jaccard_subset() {
    // a is a strict subset of b
    let a: HashSet<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let b: HashSet<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();
    // Intersection: 2, Union: 3 -> 2/3
    let sim = jaccard_similarity(&a, &b);
    assert!((sim - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_jaccard_threshold_boundary() {
    // 3 deps each, 1 shared: Jaccard = 1/5 = 0.2 (below 0.3 threshold)
    let a: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let b: HashSet<String> = ["a", "d", "e"].iter().map(|s| s.to_string()).collect();
    let sim = jaccard_similarity(&a, &b);
    assert!((sim - 0.2).abs() < 1e-6);
    assert!(sim < DEFAULT_SIMILARITY_THRESHOLD);
}

#[test]
fn test_jaccard_at_threshold() {
    // 4 deps each, 2 shared: Jaccard = 2/6 = 0.333... (at threshold)
    let a: HashSet<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
    let b: HashSet<String> = ["a", "b", "e", "f"].iter().map(|s| s.to_string()).collect();
    let sim = jaccard_similarity(&a, &b);
    assert!(sim >= DEFAULT_SIMILARITY_THRESHOLD);
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

// ---- End-to-end wiring tests -----------------------------------------------

#[tokio::test]
async fn test_parse_store_and_group_cargo() {
    let pool = setup_pool().await;

    // Simulate ingesting two Cargo.toml files for different projects
    let cargo_a = r#"
[dependencies]
serde = "1.0"
tokio = "1.0"
sqlx = "0.7"
"#;
    let cargo_b = r#"
[dependencies]
serde = "1.0"
tokio = "1.0"
reqwest = "0.11"
"#;

    let deps_a = parse_dependencies("Cargo.toml", cargo_a);
    assert_eq!(deps_a.len(), 3);
    store_dependencies(&pool, "proj-rust-a", &deps_a)
        .await
        .unwrap();

    let deps_b = parse_dependencies("Cargo.toml", cargo_b);
    assert_eq!(deps_b.len(), 3);
    store_dependencies(&pool, "proj-rust-b", &deps_b)
        .await
        .unwrap();

    // Jaccard: intersection {serde, tokio} = 2, union {serde, tokio, sqlx, reqwest} = 4
    // Similarity = 0.5, above default threshold of 0.3
    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 1);

    let members = schema::get_group_members(&pool, "proj-rust-a")
        .await
        .unwrap();
    assert!(members.contains(&"proj-rust-b".to_string()));
}

#[tokio::test]
async fn test_parse_store_and_group_npm() {
    let pool = setup_pool().await;

    let pkg_a = r#"{"dependencies":{"react":"^18","next":"^14","tailwind":"^3"}}"#;
    let pkg_b = r#"{"dependencies":{"react":"^18","next":"^14","express":"^4"}}"#;

    let deps_a = parse_dependencies("package.json", pkg_a);
    store_dependencies(&pool, "proj-npm-a", &deps_a)
        .await
        .unwrap();

    let deps_b = parse_dependencies("package.json", pkg_b);
    store_dependencies(&pool, "proj-npm-b", &deps_b)
        .await
        .unwrap();

    // Jaccard: {react, next} / {react, next, tailwind, express} = 2/4 = 0.5
    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 1);
}

#[tokio::test]
async fn test_cross_ecosystem_no_false_grouping() {
    let pool = setup_pool().await;

    // Rust project with common dep names
    let cargo = r#"
[dependencies]
serde = "1.0"
tokio = "1.0"
"#;
    let deps_rs = parse_dependencies("Cargo.toml", cargo);
    store_dependencies(&pool, "proj-rust", &deps_rs)
        .await
        .unwrap();

    // Python project with completely different deps
    let reqs = "flask>=2.0\ndjango>=4.0\n";
    let deps_py = parse_dependencies("requirements.txt", reqs);
    store_dependencies(&pool, "proj-python", &deps_py)
        .await
        .unwrap();

    // No overlap -> no group
    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 0);
}

#[tokio::test]
async fn test_three_project_transitive_grouping() {
    let pool = setup_pool().await;

    // Three Rust projects, all sharing significant overlap
    store_dependencies(
        &pool,
        "proj-1",
        &[
            ("serde".into(), "rust".into()),
            ("tokio".into(), "rust".into()),
            ("anyhow".into(), "rust".into()),
        ],
    )
    .await
    .unwrap();

    store_dependencies(
        &pool,
        "proj-2",
        &[
            ("serde".into(), "rust".into()),
            ("tokio".into(), "rust".into()),
            ("tracing".into(), "rust".into()),
        ],
    )
    .await
    .unwrap();

    store_dependencies(
        &pool,
        "proj-3",
        &[
            ("serde".into(), "rust".into()),
            ("tokio".into(), "rust".into()),
            ("clap".into(), "rust".into()),
        ],
    )
    .await
    .unwrap();

    let groups = compute_dependency_groups(&pool, Some(0.3)).await.unwrap();
    // Each pair has Jaccard = 2/4 = 0.5, so 3 groups (pairs): 1+2, 1+3, 2+3
    assert_eq!(groups, 3);

    // All three projects should be discoverable from any one
    let members = schema::get_group_members(&pool, "proj-1").await.unwrap();
    assert!(members.contains(&"proj-2".to_string()));
    assert!(members.contains(&"proj-3".to_string()));
}

#[tokio::test]
async fn test_recompute_clears_stale_groups() {
    let pool = setup_pool().await;

    // Initial: two similar projects
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
            ("serde".into(), "rust".into()),
            ("tokio".into(), "rust".into()),
        ],
    )
    .await
    .unwrap();

    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 1);

    // Update proj-b to have completely different deps
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

    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 0);

    // proj-a should have no group members after recompute
    let members = schema::get_group_members(&pool, "proj-a").await.unwrap();
    assert!(
        members.is_empty() || members == vec!["proj-a".to_string()],
        "proj-a should have no dependency group members after recompute"
    );
}

#[tokio::test]
async fn test_threshold_below_default() {
    let pool = setup_pool().await;

    // 5 deps each, 1 shared: Jaccard = 1/9 = 0.111 (well below 0.3)
    store_dependencies(
        &pool,
        "proj-a",
        &[
            ("a".into(), "x".into()),
            ("b".into(), "x".into()),
            ("c".into(), "x".into()),
            ("d".into(), "x".into()),
            ("shared".into(), "x".into()),
        ],
    )
    .await
    .unwrap();

    store_dependencies(
        &pool,
        "proj-b",
        &[
            ("e".into(), "x".into()),
            ("f".into(), "x".into()),
            ("g".into(), "x".into()),
            ("h".into(), "x".into()),
            ("shared".into(), "x".into()),
        ],
    )
    .await
    .unwrap();

    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 0, "1/9 similarity should be below 0.3 threshold");
}

#[tokio::test]
async fn test_custom_threshold() {
    let pool = setup_pool().await;

    // 5 deps each, 1 shared: Jaccard = 1/9 ~ 0.111
    store_dependencies(
        &pool,
        "proj-a",
        &[
            ("a".into(), "x".into()),
            ("b".into(), "x".into()),
            ("c".into(), "x".into()),
            ("d".into(), "x".into()),
            ("shared".into(), "x".into()),
        ],
    )
    .await
    .unwrap();

    store_dependencies(
        &pool,
        "proj-b",
        &[
            ("e".into(), "x".into()),
            ("f".into(), "x".into()),
            ("g".into(), "x".into()),
            ("h".into(), "x".into()),
            ("shared".into(), "x".into()),
        ],
    )
    .await
    .unwrap();

    // With a very low threshold, they should group
    let groups = compute_dependency_groups(&pool, Some(0.1)).await.unwrap();
    assert_eq!(groups, 1, "1/9 ~ 0.111 is above 0.1 threshold");
}

#[tokio::test]
async fn test_single_project_no_groups() {
    let pool = setup_pool().await;

    store_dependencies(&pool, "solo-project", &[("serde".into(), "rust".into())])
        .await
        .unwrap();

    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 0, "A single project cannot form a group");
}

#[tokio::test]
async fn test_empty_deps_no_groups() {
    let pool = setup_pool().await;

    // Store empty dependency sets
    store_dependencies(&pool, "proj-a", &[]).await.unwrap();
    store_dependencies(&pool, "proj-b", &[]).await.unwrap();

    let all = load_all_dependency_sets(&pool).await.unwrap();
    assert!(all.is_empty(), "Empty deps should not appear in dep sets");

    let groups = compute_dependency_groups(&pool, None).await.unwrap();
    assert_eq!(groups, 0);
}

#[tokio::test]
async fn test_load_multiple_tenants() {
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
            ("react".into(), "npm".into()),
        ],
    )
    .await
    .unwrap();
    store_dependencies(
        &pool,
        "proj-c",
        &[
            ("flask".into(), "python".into()),
            ("requests".into(), "python".into()),
        ],
    )
    .await
    .unwrap();

    let all = load_all_dependency_sets(&pool).await.unwrap();
    assert_eq!(all.len(), 3);
    assert!(all["proj-a"].contains("serde"));
    assert!(all["proj-b"].contains("express"));
    assert!(all["proj-c"].contains("flask"));
}

#[tokio::test]
async fn test_group_id_deterministic() {
    let pool = setup_pool().await;

    store_dependencies(&pool, "proj-a", &[("serde".into(), "rust".into())])
        .await
        .unwrap();
    store_dependencies(&pool, "proj-b", &[("serde".into(), "rust".into())])
        .await
        .unwrap();

    compute_dependency_groups(&pool, Some(0.1)).await.unwrap();

    let groups_a = schema::list_tenant_groups(&pool, "proj-a").await.unwrap();
    let groups_b = schema::list_tenant_groups(&pool, "proj-b").await.unwrap();

    // Both should be in the same group with type "dependency"
    assert_eq!(groups_a.len(), 1);
    assert_eq!(groups_b.len(), 1);
    assert_eq!(groups_a[0].0, groups_b[0].0); // same group_id
    assert_eq!(groups_a[0].1, "dependency");
    assert_eq!(groups_b[0].1, "dependency");

    // Group ID should be deterministic: "dep:proj-a+proj-b" (sorted)
    assert_eq!(groups_a[0].0, "dep:proj-a+proj-b");
}
