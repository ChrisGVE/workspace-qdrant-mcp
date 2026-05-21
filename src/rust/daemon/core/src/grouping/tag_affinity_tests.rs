use std::collections::{HashMap, HashSet};

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;

use super::schema;
use super::tag_affinity::{
    build_tag_affinity_groups, compute_group_mean_jaccard, compute_tag_affinities,
    compute_tag_affinity_groups, load_project_tag_profiles, tag_affinity_group_id,
    tag_jaccard_similarity, TagAffinity, TagAffinityConfig,
};
use crate::keywords_schema::{CREATE_TAGS_INDEXES_SQL, CREATE_TAGS_SQL};

async fn setup_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();

    sqlx::query(CREATE_TAGS_SQL).execute(&pool).await.unwrap();
    for idx in CREATE_TAGS_INDEXES_SQL {
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

async fn insert_tag(pool: &SqlitePool, tenant_id: &str, doc_id: &str, tag: &str) {
    sqlx::query(
        r#"
        INSERT INTO tags (doc_id, tag, tag_type, score, diversity_score, collection, tenant_id)
        VALUES (?, ?, 'concept', 1.0, 0.5, 'projects', ?)
        "#,
    )
    .bind(doc_id)
    .bind(tag)
    .bind(tenant_id)
    .execute(pool)
    .await
    .unwrap();
}

// ---- Jaccard similarity ----------------------------------------------------

#[test]
fn test_jaccard_identical_sets() {
    let a: HashSet<String> = ["rust", "async", "web"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let b = a.clone();
    assert!((tag_jaccard_similarity(&a, &b) - 1.0).abs() < 1e-10);
}

#[test]
fn test_jaccard_disjoint_sets() {
    let a: HashSet<String> = ["rust", "async"].iter().map(|s| s.to_string()).collect();
    let b: HashSet<String> = ["python", "django"].iter().map(|s| s.to_string()).collect();
    assert!((tag_jaccard_similarity(&a, &b)).abs() < 1e-10);
}

#[test]
fn test_jaccard_partial_overlap() {
    let a: HashSet<String> = ["rust", "async", "web"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let b: HashSet<String> = ["rust", "web", "cli"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    // intersection = {rust, web} = 2, union = {rust, async, web, cli} = 4
    assert!((tag_jaccard_similarity(&a, &b) - 0.5).abs() < 1e-10);
}

#[test]
fn test_jaccard_both_empty() {
    let a: HashSet<String> = HashSet::new();
    let b: HashSet<String> = HashSet::new();
    assert!((tag_jaccard_similarity(&a, &b)).abs() < 1e-10);
}

#[test]
fn test_jaccard_one_empty() {
    let a: HashSet<String> = ["rust"].iter().map(|s| s.to_string()).collect();
    let b: HashSet<String> = HashSet::new();
    assert!((tag_jaccard_similarity(&a, &b)).abs() < 1e-10);
}

#[test]
fn test_jaccard_subset() {
    let a: HashSet<String> = ["rust", "async", "web"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let b: HashSet<String> = ["rust", "async"].iter().map(|s| s.to_string()).collect();
    // intersection = 2, union = 3
    assert!((tag_jaccard_similarity(&a, &b) - 2.0 / 3.0).abs() < 1e-10);
}

// ---- Pairwise computation --------------------------------------------------

#[test]
fn test_compute_affinities_above_threshold() {
    let mut profiles: HashMap<String, HashSet<String>> = HashMap::new();
    profiles.insert(
        "proj-a".into(),
        ["rust", "async", "web"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );
    profiles.insert(
        "proj-b".into(),
        ["rust", "web", "cli"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );
    profiles.insert(
        "proj-c".into(),
        ["python", "django", "ml"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );

    // a-b Jaccard = 0.5, a-c = 0.0, b-c = 0.0
    let affinities = compute_tag_affinities(&profiles, 0.25);
    assert_eq!(affinities.len(), 1);
    // The pair should be proj-a and proj-b (order may vary)
    let pair: HashSet<&str> = [
        affinities[0].tenant_a.as_str(),
        affinities[0].tenant_b.as_str(),
    ]
    .into_iter()
    .collect();
    assert!(pair.contains("proj-a"));
    assert!(pair.contains("proj-b"));
    assert!((affinities[0].similarity - 0.5).abs() < 1e-10);
}

#[test]
fn test_compute_affinities_none_above() {
    let mut profiles: HashMap<String, HashSet<String>> = HashMap::new();
    profiles.insert(
        "proj-a".into(),
        ["rust"].iter().map(|s| s.to_string()).collect(),
    );
    profiles.insert(
        "proj-b".into(),
        ["python"].iter().map(|s| s.to_string()).collect(),
    );

    let affinities = compute_tag_affinities(&profiles, 0.25);
    assert!(affinities.is_empty());
}

#[test]
fn test_compute_affinities_empty_profiles() {
    let profiles: HashMap<String, HashSet<String>> = HashMap::new();
    let affinities = compute_tag_affinities(&profiles, 0.25);
    assert!(affinities.is_empty());
}

// ---- Connected components --------------------------------------------------

#[test]
fn test_build_groups_single_pair() {
    let affinities = vec![TagAffinity {
        tenant_a: "a".into(),
        tenant_b: "b".into(),
        similarity: 0.5,
    }];

    let groups = build_tag_affinity_groups(&affinities);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec!["a", "b"]);
}

#[test]
fn test_build_groups_transitive() {
    let affinities = vec![
        TagAffinity {
            tenant_a: "a".into(),
            tenant_b: "b".into(),
            similarity: 0.4,
        },
        TagAffinity {
            tenant_a: "b".into(),
            tenant_b: "c".into(),
            similarity: 0.3,
        },
    ];

    let groups = build_tag_affinity_groups(&affinities);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec!["a", "b", "c"]);
}

#[test]
fn test_build_groups_two_separate() {
    let affinities = vec![
        TagAffinity {
            tenant_a: "a".into(),
            tenant_b: "b".into(),
            similarity: 0.5,
        },
        TagAffinity {
            tenant_a: "c".into(),
            tenant_b: "d".into(),
            similarity: 0.4,
        },
    ];

    let groups = build_tag_affinity_groups(&affinities);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0], vec!["a", "b"]);
    assert_eq!(groups[1], vec!["c", "d"]);
}

#[test]
fn test_build_groups_empty() {
    let groups = build_tag_affinity_groups(&[]);
    assert!(groups.is_empty());
}

// ---- Group ID --------------------------------------------------------------

#[test]
fn test_group_id_deterministic() {
    let members = vec!["proj-b".to_string(), "proj-a".to_string()];
    let id1 = tag_affinity_group_id(&members);
    let id2 = tag_affinity_group_id(&members);
    assert_eq!(id1, id2);

    // Order should not matter
    let members_rev = vec!["proj-a".to_string(), "proj-b".to_string()];
    let id3 = tag_affinity_group_id(&members_rev);
    assert_eq!(id1, id3);
}

#[test]
fn test_group_id_unique() {
    let id1 = tag_affinity_group_id(&["a".into(), "b".into()]);
    let id2 = tag_affinity_group_id(&["a".into(), "c".into()]);
    assert_ne!(id1, id2);
}

#[test]
fn test_group_id_prefix() {
    let id = tag_affinity_group_id(&["a".into(), "b".into()]);
    assert!(id.starts_with("tag_aff:"));
}

// ---- Group mean Jaccard ----------------------------------------------------

#[test]
fn test_group_mean_jaccard() {
    let members = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let affinities = vec![
        TagAffinity {
            tenant_a: "a".into(),
            tenant_b: "b".into(),
            similarity: 0.6,
        },
        TagAffinity {
            tenant_a: "a".into(),
            tenant_b: "c".into(),
            similarity: 0.4,
        },
        TagAffinity {
            tenant_a: "b".into(),
            tenant_b: "c".into(),
            similarity: 0.5,
        },
    ];

    let mean = compute_group_mean_jaccard(&members, &affinities);
    assert!((mean - 0.5).abs() < 1e-10);
}

#[test]
fn test_group_mean_jaccard_no_pairs() {
    let members = vec!["x".to_string()];
    let mean = compute_group_mean_jaccard(&members, &[]);
    assert_eq!(mean, 0.0);
}

// ---- SQLite integration: load profiles -------------------------------------

#[tokio::test]
async fn test_load_profiles_empty() {
    let pool = setup_pool().await;
    let profiles = load_project_tag_profiles(&pool).await.unwrap();
    assert!(profiles.is_empty());
}

#[tokio::test]
async fn test_load_profiles_single_project() {
    let pool = setup_pool().await;

    insert_tag(&pool, "proj-a", "doc-1", "rust").await;
    insert_tag(&pool, "proj-a", "doc-1", "async").await;
    insert_tag(&pool, "proj-a", "doc-2", "web").await;
    // Duplicate tag across docs should appear only once
    insert_tag(&pool, "proj-a", "doc-2", "rust").await;

    let profiles = load_project_tag_profiles(&pool).await.unwrap();
    assert_eq!(profiles.len(), 1);
    let tags = &profiles["proj-a"];
    assert_eq!(tags.len(), 3);
    assert!(tags.contains("rust"));
    assert!(tags.contains("async"));
    assert!(tags.contains("web"));
}

#[tokio::test]
async fn test_load_profiles_multiple_projects() {
    let pool = setup_pool().await;

    insert_tag(&pool, "proj-a", "doc-1", "rust").await;
    insert_tag(&pool, "proj-a", "doc-1", "async").await;
    insert_tag(&pool, "proj-b", "doc-2", "python").await;
    insert_tag(&pool, "proj-b", "doc-2", "django").await;

    let profiles = load_project_tag_profiles(&pool).await.unwrap();
    assert_eq!(profiles.len(), 2);
    assert_eq!(profiles["proj-a"].len(), 2);
    assert_eq!(profiles["proj-b"].len(), 2);
}

// ---- SQLite integration: full grouping -------------------------------------

#[tokio::test]
async fn test_compute_groups_shared_tags() {
    let pool = setup_pool().await;

    // proj-a: {rust, async, web}
    insert_tag(&pool, "proj-a", "doc-1", "rust").await;
    insert_tag(&pool, "proj-a", "doc-1", "async").await;
    insert_tag(&pool, "proj-a", "doc-2", "web").await;

    // proj-b: {rust, web, cli} -- Jaccard with proj-a = 2/4 = 0.5
    insert_tag(&pool, "proj-b", "doc-3", "rust").await;
    insert_tag(&pool, "proj-b", "doc-3", "web").await;
    insert_tag(&pool, "proj-b", "doc-4", "cli").await;

    // proj-c: {python, django, ml} -- disjoint from both
    insert_tag(&pool, "proj-c", "doc-5", "python").await;
    insert_tag(&pool, "proj-c", "doc-5", "django").await;
    insert_tag(&pool, "proj-c", "doc-6", "ml").await;

    let config = TagAffinityConfig::default(); // threshold = 0.25
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 1, "Only proj-a and proj-b share enough tags");

    let members = schema::get_group_members(&pool, "proj-a").await.unwrap();
    assert_eq!(members.len(), 2);
    assert!(members.contains(&"proj-a".to_string()));
    assert!(members.contains(&"proj-b".to_string()));

    // proj-c should have no group members
    let c_members = schema::get_group_members(&pool, "proj-c").await.unwrap();
    assert!(c_members.is_empty());
}

#[tokio::test]
async fn test_compute_groups_no_overlap() {
    let pool = setup_pool().await;

    insert_tag(&pool, "proj-a", "doc-1", "rust").await;
    insert_tag(&pool, "proj-b", "doc-2", "python").await;

    let config = TagAffinityConfig::default();
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 0);
}

#[tokio::test]
async fn test_compute_groups_single_project() {
    let pool = setup_pool().await;

    insert_tag(&pool, "proj-a", "doc-1", "rust").await;

    let config = TagAffinityConfig::default();
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 0, "Need at least 2 projects");
}

#[tokio::test]
async fn test_compute_groups_custom_threshold() {
    let pool = setup_pool().await;

    // proj-a: {rust, async, web, api, tokio}
    for tag in &["rust", "async", "web", "api", "tokio"] {
        insert_tag(&pool, "proj-a", "doc-1", tag).await;
    }
    // proj-b: {rust, cli} -- Jaccard = 1/6 ≈ 0.167
    insert_tag(&pool, "proj-b", "doc-2", "rust").await;
    insert_tag(&pool, "proj-b", "doc-2", "cli").await;

    // Default threshold 0.25 should NOT group them
    let config = TagAffinityConfig::default();
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 0, "0.167 < 0.25 threshold");

    // Lower threshold SHOULD group them
    let config = TagAffinityConfig {
        similarity_threshold: 0.15,
    };
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 1, "0.167 >= 0.15 threshold");
}

#[tokio::test]
async fn test_recompute_clears_stale_groups() {
    let pool = setup_pool().await;

    // First pass: two similar projects
    insert_tag(&pool, "proj-a", "doc-1", "rust").await;
    insert_tag(&pool, "proj-a", "doc-1", "async").await;
    insert_tag(&pool, "proj-b", "doc-2", "rust").await;
    insert_tag(&pool, "proj-b", "doc-2", "async").await;

    let config = TagAffinityConfig::default();
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 1);

    // Remove proj-b's tags and add disjoint ones
    sqlx::query("DELETE FROM tags WHERE tenant_id = 'proj-b'")
        .execute(&pool)
        .await
        .unwrap();
    insert_tag(&pool, "proj-b", "doc-2", "python").await;
    insert_tag(&pool, "proj-b", "doc-2", "django").await;

    // Recompute: should clear old group
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 0, "No overlap after tag change");

    let members = schema::get_group_members(&pool, "proj-a").await.unwrap();
    assert!(members.is_empty());
}

#[tokio::test]
async fn test_multiple_groups_formed() {
    let pool = setup_pool().await;

    // Group 1: proj-a and proj-b share {rust, async}
    insert_tag(&pool, "proj-a", "d1", "rust").await;
    insert_tag(&pool, "proj-a", "d1", "async").await;
    insert_tag(&pool, "proj-b", "d2", "rust").await;
    insert_tag(&pool, "proj-b", "d2", "async").await;
    insert_tag(&pool, "proj-b", "d2", "tokio").await;

    // Group 2: proj-c and proj-d share {python, flask}
    insert_tag(&pool, "proj-c", "d3", "python").await;
    insert_tag(&pool, "proj-c", "d3", "flask").await;
    insert_tag(&pool, "proj-d", "d4", "python").await;
    insert_tag(&pool, "proj-d", "d4", "flask").await;
    insert_tag(&pool, "proj-d", "d4", "sqlalchemy").await;

    let config = TagAffinityConfig::default();
    let groups = compute_tag_affinity_groups(&pool, &config).await.unwrap();
    assert_eq!(groups, 2, "Two separate tag-affinity groups");

    // proj-a should see proj-b
    let a_members = schema::get_group_members(&pool, "proj-a").await.unwrap();
    assert_eq!(a_members.len(), 2);
    assert!(a_members.contains(&"proj-b".to_string()));

    // proj-c should see proj-d
    let c_members = schema::get_group_members(&pool, "proj-c").await.unwrap();
    assert_eq!(c_members.len(), 2);
    assert!(c_members.contains(&"proj-d".to_string()));

    // proj-a should NOT see proj-c
    assert!(!a_members.contains(&"proj-c".to_string()));
}

#[tokio::test]
async fn test_group_type_stored_as_tag_affinity() {
    let pool = setup_pool().await;

    insert_tag(&pool, "proj-a", "d1", "rust").await;
    insert_tag(&pool, "proj-b", "d2", "rust").await;

    let config = TagAffinityConfig::default();
    compute_tag_affinity_groups(&pool, &config).await.unwrap();

    let groups = schema::list_tenant_groups(&pool, "proj-a").await.unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(
        groups[0].1, "tag_affinity",
        "group_type must be tag_affinity"
    );
}

#[tokio::test]
async fn test_confidence_reflects_mean_jaccard() {
    let pool = setup_pool().await;

    // Identical tag sets: Jaccard = 1.0
    insert_tag(&pool, "proj-a", "d1", "rust").await;
    insert_tag(&pool, "proj-a", "d1", "web").await;
    insert_tag(&pool, "proj-b", "d2", "rust").await;
    insert_tag(&pool, "proj-b", "d2", "web").await;

    let config = TagAffinityConfig::default();
    compute_tag_affinity_groups(&pool, &config).await.unwrap();

    let groups = schema::list_tenant_groups(&pool, "proj-a").await.unwrap();
    assert_eq!(groups.len(), 1);
    assert!(
        (groups[0].2 - 1.0).abs() < 1e-10,
        "Confidence should be 1.0 for identical tag sets"
    );
}
