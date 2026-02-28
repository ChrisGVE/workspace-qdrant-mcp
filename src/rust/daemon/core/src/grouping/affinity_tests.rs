use super::*;
use sqlx::sqlite::SqlitePoolOptions;

async fn setup_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();

    sqlx::query(CREATE_PROJECT_EMBEDDINGS_SQL)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(CREATE_AFFINITY_LABELS_SQL)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query(schema::CREATE_PROJECT_GROUPS_SQL)
        .execute(&pool)
        .await
        .unwrap();
    for idx in schema::CREATE_PROJECT_GROUPS_INDEXES_SQL {
        sqlx::query(idx).execute(&pool).await.unwrap();
    }

    pool
}

// ---- Blob serialization ----------------------------------------------------

#[test]
fn test_embedding_to_blob_roundtrip() {
    let embedding = vec![1.0f32, -0.5, 0.0, 3.14159];
    let blob = embedding_to_blob(&embedding);
    let recovered = blob_to_embedding(&blob, 4).unwrap();
    assert_eq!(embedding, recovered);
}

#[test]
fn test_blob_to_embedding_wrong_dim() {
    let blob = vec![0u8; 16]; // 4 floats
    assert!(blob_to_embedding(&blob, 5).is_none());
}

#[test]
fn test_blob_to_embedding_empty() {
    assert!(blob_to_embedding(&[], 0).is_some()); // empty is valid
    assert!(blob_to_embedding(&[], 1).is_none()); // mismatch
}

// ---- Store / load ----------------------------------------------------------

#[tokio::test]
async fn test_store_and_load_embedding() {
    let pool = setup_pool().await;
    let emb = vec![0.1, 0.2, 0.3, 0.4];

    store_project_embedding(&pool, "proj-a", &emb, 10, None)
        .await
        .unwrap();

    let loaded = load_project_embedding(&pool, "proj-a").await.unwrap();
    assert!(loaded.is_some());
    let loaded = loaded.unwrap();
    assert_eq!(loaded.len(), 4);
    assert!((loaded[0] - 0.1).abs() < 1e-6);
}

#[tokio::test]
async fn test_load_nonexistent_embedding() {
    let pool = setup_pool().await;
    let loaded = load_project_embedding(&pool, "nonexistent").await.unwrap();
    assert!(loaded.is_none());
}

#[tokio::test]
async fn test_store_replaces_embedding() {
    let pool = setup_pool().await;

    store_project_embedding(&pool, "proj-a", &[1.0, 2.0], 5, None)
        .await
        .unwrap();
    store_project_embedding(&pool, "proj-a", &[3.0, 4.0], 10, Some("updated"))
        .await
        .unwrap();

    let loaded = load_project_embedding(&pool, "proj-a").await.unwrap().unwrap();
    assert!((loaded[0] - 3.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_load_all_embeddings() {
    let pool = setup_pool().await;

    store_project_embedding(&pool, "proj-a", &[1.0, 0.0], 5, None)
        .await
        .unwrap();
    store_project_embedding(&pool, "proj-b", &[0.0, 1.0], 3, None)
        .await
        .unwrap();

    let all = load_all_project_embeddings(&pool).await.unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].0, "proj-a");
    assert_eq!(all[1].0, "proj-b");
}

#[tokio::test]
async fn test_delete_embedding() {
    let pool = setup_pool().await;

    store_project_embedding(&pool, "proj-a", &[1.0], 1, None)
        .await
        .unwrap();

    let deleted = delete_project_embedding(&pool, "proj-a").await.unwrap();
    assert!(deleted);

    let loaded = load_project_embedding(&pool, "proj-a").await.unwrap();
    assert!(loaded.is_none());

    // Delete nonexistent
    let deleted = delete_project_embedding(&pool, "proj-a").await.unwrap();
    assert!(!deleted);
}

// ---- Pairwise affinity -----------------------------------------------------

#[test]
fn test_pairwise_identical_projects() {
    let embeddings = vec![
        ("proj-a".to_string(), vec![1.0, 0.0, 0.0]),
        ("proj-b".to_string(), vec![1.0, 0.0, 0.0]),
    ];

    let affinities = compute_pairwise_affinities(&embeddings, 0.7);
    assert_eq!(affinities.len(), 1);
    assert!((affinities[0].similarity - 1.0).abs() < 1e-6);
}

#[test]
fn test_pairwise_orthogonal_projects() {
    let embeddings = vec![
        ("proj-a".to_string(), vec![1.0, 0.0, 0.0]),
        ("proj-b".to_string(), vec![0.0, 1.0, 0.0]),
    ];

    let affinities = compute_pairwise_affinities(&embeddings, 0.7);
    assert!(affinities.is_empty(), "Orthogonal vectors should not group");
}

#[test]
fn test_pairwise_three_projects() {
    let embeddings = vec![
        ("proj-a".to_string(), vec![1.0, 0.0]),
        ("proj-b".to_string(), vec![0.95, 0.31]),  // cos sim with a ~ 0.95
        ("proj-c".to_string(), vec![0.0, 1.0]),     // orthogonal to a
    ];

    let affinities = compute_pairwise_affinities(&embeddings, 0.7);
    assert_eq!(affinities.len(), 1);
    assert_eq!(affinities[0].tenant_a, "proj-a");
    assert_eq!(affinities[0].tenant_b, "proj-b");
}

// ---- Connected components --------------------------------------------------

#[test]
fn test_build_groups_single_pair() {
    let affinities = vec![ProjectAffinity {
        tenant_a: "a".into(),
        tenant_b: "b".into(),
        similarity: 0.9,
    }];

    let groups = build_affinity_groups(&affinities);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec!["a", "b"]);
}

#[test]
fn test_build_groups_transitive() {
    let affinities = vec![
        ProjectAffinity {
            tenant_a: "a".into(),
            tenant_b: "b".into(),
            similarity: 0.8,
        },
        ProjectAffinity {
            tenant_a: "b".into(),
            tenant_b: "c".into(),
            similarity: 0.75,
        },
    ];

    let groups = build_affinity_groups(&affinities);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec!["a", "b", "c"]);
}

#[test]
fn test_build_groups_two_separate() {
    let affinities = vec![
        ProjectAffinity {
            tenant_a: "a".into(),
            tenant_b: "b".into(),
            similarity: 0.9,
        },
        ProjectAffinity {
            tenant_a: "c".into(),
            tenant_b: "d".into(),
            similarity: 0.8,
        },
    ];

    let groups = build_affinity_groups(&affinities);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0], vec!["a", "b"]);
    assert_eq!(groups[1], vec!["c", "d"]);
}

#[test]
fn test_build_groups_empty() {
    let groups = build_affinity_groups(&[]);
    assert!(groups.is_empty());
}

// ---- Group ID --------------------------------------------------------------

#[test]
fn test_affinity_group_id_deterministic() {
    let members = vec!["proj-b".to_string(), "proj-a".to_string()];
    let id1 = affinity_group_id(&members);
    let id2 = affinity_group_id(&members);
    assert_eq!(id1, id2);

    // Order shouldn't matter
    let members_rev = vec!["proj-a".to_string(), "proj-b".to_string()];
    let id3 = affinity_group_id(&members_rev);
    assert_eq!(id1, id3);
}

#[test]
fn test_affinity_group_id_unique() {
    let id1 = affinity_group_id(&["a".into(), "b".into()]);
    let id2 = affinity_group_id(&["a".into(), "c".into()]);
    assert_ne!(id1, id2);
}

// ---- Group mean similarity -------------------------------------------------

#[test]
fn test_group_mean_similarity() {
    let members = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let affinities = vec![
        ProjectAffinity {
            tenant_a: "a".into(),
            tenant_b: "b".into(),
            similarity: 0.8,
        },
        ProjectAffinity {
            tenant_a: "a".into(),
            tenant_b: "c".into(),
            similarity: 0.9,
        },
        ProjectAffinity {
            tenant_a: "b".into(),
            tenant_b: "c".into(),
            similarity: 0.7,
        },
    ];

    let mean = compute_group_mean_similarity(&members, &affinities);
    assert!((mean - 0.8).abs() < 1e-6);
}

#[test]
fn test_group_mean_similarity_no_pairs() {
    let members = vec!["x".to_string()];
    let mean = compute_group_mean_similarity(&members, &[]);
    assert_eq!(mean, 0.0);
}

// ---- AffinityGrouper integration -------------------------------------------

#[tokio::test]
async fn test_grouper_similar_projects() {
    let pool = setup_pool().await;

    // Two very similar embeddings (cosine sim ~ 1.0)
    store_project_embedding(&pool, "proj-a", &[1.0, 0.0, 0.0], 5, None)
        .await
        .unwrap();
    store_project_embedding(&pool, "proj-b", &[0.99, 0.14, 0.0], 5, None)
        .await
        .unwrap();

    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
    let groups = grouper.compute_affinity_groups().await.unwrap();
    assert_eq!(groups, 1);

    let members = schema::get_group_members(&pool, "proj-a")
        .await
        .unwrap();
    assert_eq!(members.len(), 2);
    assert!(members.contains(&"proj-a".to_string()));
    assert!(members.contains(&"proj-b".to_string()));
}

#[tokio::test]
async fn test_grouper_dissimilar_projects() {
    let pool = setup_pool().await;

    store_project_embedding(&pool, "proj-a", &[1.0, 0.0, 0.0], 5, None)
        .await
        .unwrap();
    store_project_embedding(&pool, "proj-b", &[0.0, 1.0, 0.0], 5, None)
        .await
        .unwrap();

    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
    let groups = grouper.compute_affinity_groups().await.unwrap();
    assert_eq!(groups, 0);
}

#[tokio::test]
async fn test_grouper_single_project() {
    let pool = setup_pool().await;

    store_project_embedding(&pool, "proj-a", &[1.0, 0.0], 5, None)
        .await
        .unwrap();

    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
    let groups = grouper.compute_affinity_groups().await.unwrap();
    assert_eq!(groups, 0, "Need at least 2 projects");
}

#[tokio::test]
async fn test_grouper_recompute_clears_old() {
    let pool = setup_pool().await;

    // First: two similar projects
    store_project_embedding(&pool, "proj-a", &[1.0, 0.0], 5, None)
        .await
        .unwrap();
    store_project_embedding(&pool, "proj-b", &[1.0, 0.0], 5, None)
        .await
        .unwrap();

    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
    grouper.compute_affinity_groups().await.unwrap();

    // Now make them dissimilar
    store_project_embedding(&pool, "proj-b", &[0.0, 1.0], 5, None)
        .await
        .unwrap();

    let groups = grouper.compute_affinity_groups().await.unwrap();
    assert_eq!(groups, 0);

    let members = schema::get_group_members(&pool, "proj-a")
        .await
        .unwrap();
    assert!(members.is_empty() || members == vec!["proj-a".to_string()]);
}

#[tokio::test]
async fn test_grouper_get_project_groups_empty() {
    let pool = setup_pool().await;
    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());

    let groups = grouper.get_project_affinity_groups("nonexistent").await.unwrap();
    assert!(groups.is_empty());
}

#[tokio::test]
async fn test_grouper_with_labels() {
    let pool = setup_pool().await;

    // Create two similar projects
    store_project_embedding(&pool, "proj-a", &[1.0, 0.0, 0.0], 5, None)
        .await
        .unwrap();
    store_project_embedding(&pool, "proj-b", &[0.99, 0.14, 0.0], 5, None)
        .await
        .unwrap();

    let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
    grouper.compute_affinity_groups().await.unwrap();

    // Manually store a label (since we can't easily create a Tier2Tagger in tests)
    let groups = grouper.get_project_affinity_groups("proj-a").await.unwrap();
    assert_eq!(groups.len(), 1);

    store_affinity_label(
        &pool,
        &groups[0].group_id,
        "web development",
        "web-development",
        0.82,
    )
    .await
    .unwrap();

    // Re-query with labels
    let groups = grouper.get_project_affinity_groups("proj-a").await.unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].label.as_deref(), Some("web development"));
    assert_eq!(groups[0].category.as_deref(), Some("web-development"));
}

// ---- Affinity label storage ------------------------------------------------

#[tokio::test]
async fn test_store_and_load_label() {
    let pool = setup_pool().await;

    store_affinity_label(&pool, "grp-1", "machine learning", "ml", 0.9)
        .await
        .unwrap();

    let label = load_affinity_label(&pool, "grp-1").await.unwrap();
    assert!(label.is_some());
    let (name, cat, score) = label.unwrap();
    assert_eq!(name, "machine learning");
    assert_eq!(cat, "ml");
    assert!((score - 0.9).abs() < 1e-6);
}

#[tokio::test]
async fn test_load_label_nonexistent() {
    let pool = setup_pool().await;
    let label = load_affinity_label(&pool, "nonexistent").await.unwrap();
    assert!(label.is_none());
}

// ---- Group mean embedding --------------------------------------------------

#[test]
fn test_group_mean_embedding() {
    let embeddings = vec![
        ("a".to_string(), vec![1.0, 0.0]),
        ("b".to_string(), vec![0.0, 1.0]),
        ("c".to_string(), vec![0.5, 0.5]),
    ];

    let members = vec!["a".to_string(), "b".to_string()];
    let mean = group_mean_embedding(&members, &embeddings).unwrap();
    assert!((mean[0] - 0.5).abs() < 1e-5);
    assert!((mean[1] - 0.5).abs() < 1e-5);
}

#[test]
fn test_group_mean_embedding_missing_member() {
    let embeddings = vec![("a".to_string(), vec![1.0, 0.0])];
    let members = vec!["a".to_string(), "z".to_string()]; // z not in embeddings
    let mean = group_mean_embedding(&members, &embeddings).unwrap();
    // Only "a" contributes -> [1.0, 0.0]
    assert!((mean[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_group_mean_embedding_no_members() {
    let embeddings: Vec<(String, Vec<f32>)> = Vec::new();
    let members: Vec<String> = Vec::new();
    assert!(group_mean_embedding(&members, &embeddings).is_none());
}
