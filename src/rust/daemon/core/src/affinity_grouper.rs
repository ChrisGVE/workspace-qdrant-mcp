/// Automated project affinity grouping via embeddings.
///
/// Computes aggregate embeddings per project (mean of all chunk embeddings),
/// stores them as binary blobs in SQLite, then groups projects whose pairwise
/// cosine similarity exceeds a threshold into `project_groups` entries with
/// `group_type = "affinity"`.
///
/// Group labels are derived from zero-shot taxonomy classification
/// (see `tier2_tagging`).

use sqlx::{Row, SqlitePool};
use tracing::{debug, info, warn};

use crate::keyword_extraction::semantic_rerank::cosine_similarity;
use crate::project_groups_schema;
use crate::tier2_tagging::{Tier2Tagger, aggregate_document_embedding};

// ─── Configuration ──────────────────────────────────────────────────────

/// Configuration for affinity grouping.
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    /// Minimum cosine similarity between two project aggregate embeddings
    /// to group them (default: 0.7).
    pub similarity_threshold: f64,
    /// Maximum number of projects in a single affinity group (0 = unlimited).
    pub max_group_size: usize,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_group_size: 0,
        }
    }
}

// ─── SQLite schema ──────────────────────────────────────────────────────

/// SQL to create the project_embeddings table (schema v25).
pub const CREATE_PROJECT_EMBEDDINGS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS project_embeddings (
    tenant_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    label TEXT,
    updated_at TEXT NOT NULL
)
"#;

/// SQL to create the affinity_labels table (schema v25).
pub const CREATE_AFFINITY_LABELS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS affinity_labels (
    group_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    category TEXT NOT NULL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL
)
"#;

// ─── Embedding storage ─────────────────────────────────────────────────

/// Serialize an f32 vector to a byte blob (little-endian).
fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize a byte blob back to an f32 vector.
fn blob_to_embedding(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
    if blob.len() != dim * 4 {
        return None;
    }
    let mut vec = Vec::with_capacity(dim);
    for chunk in blob.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().ok()?;
        vec.push(f32::from_le_bytes(arr));
    }
    Some(vec)
}

/// Store or update a project's aggregate embedding.
pub async fn store_project_embedding(
    pool: &SqlitePool,
    tenant_id: &str,
    embedding: &[f32],
    chunk_count: usize,
    label: Option<&str>,
) -> Result<(), sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();
    let blob = embedding_to_blob(embedding);
    let dim = embedding.len() as i64;
    let chunk_count = chunk_count as i64;

    sqlx::query(
        r#"
        INSERT OR REPLACE INTO project_embeddings
            (tenant_id, embedding, dim, chunk_count, label, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(tenant_id)
    .bind(&blob)
    .bind(dim)
    .bind(chunk_count)
    .bind(label)
    .bind(&now)
    .execute(pool)
    .await?;

    debug!(
        tenant_id,
        dim = embedding.len(),
        chunk_count,
        "Stored project aggregate embedding"
    );

    Ok(())
}

/// Load a single project's aggregate embedding.
pub async fn load_project_embedding(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<Option<Vec<f32>>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT embedding, dim FROM project_embeddings WHERE tenant_id = ?",
    )
    .bind(tenant_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.and_then(|r| {
        let blob: Vec<u8> = r.get("embedding");
        let dim: i64 = r.get("dim");
        blob_to_embedding(&blob, dim as usize)
    }))
}

/// Load all project aggregate embeddings.
///
/// Returns a map of tenant_id → embedding vector.
pub async fn load_all_project_embeddings(
    pool: &SqlitePool,
) -> Result<Vec<(String, Vec<f32>)>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT tenant_id, embedding, dim FROM project_embeddings ORDER BY tenant_id",
    )
    .fetch_all(pool)
    .await?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let tenant: String = row.get("tenant_id");
        let blob: Vec<u8> = row.get("embedding");
        let dim: i64 = row.get("dim");
        if let Some(emb) = blob_to_embedding(&blob, dim as usize) {
            results.push((tenant, emb));
        } else {
            warn!(tenant_id = tenant.as_str(), "Corrupt embedding blob, skipping");
        }
    }

    Ok(results)
}

/// Delete a project's aggregate embedding.
pub async fn delete_project_embedding(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query("DELETE FROM project_embeddings WHERE tenant_id = ?")
        .bind(tenant_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// ─── Affinity computation ───────────────────────────────────────────────

/// A computed affinity between two projects.
#[derive(Debug, Clone)]
pub struct ProjectAffinity {
    pub tenant_a: String,
    pub tenant_b: String,
    pub similarity: f64,
}

/// Compute pairwise cosine similarities and return pairs above threshold.
pub fn compute_pairwise_affinities(
    embeddings: &[(String, Vec<f32>)],
    threshold: f64,
) -> Vec<ProjectAffinity> {
    let mut affinities = Vec::new();

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            if sim >= threshold {
                affinities.push(ProjectAffinity {
                    tenant_a: embeddings[i].0.clone(),
                    tenant_b: embeddings[j].0.clone(),
                    similarity: sim,
                });
            }
        }
    }

    affinities
}

/// Build connected components from pairwise affinities.
///
/// Groups projects transitively: if A~B and B~C, then {A,B,C} form one group.
pub fn build_affinity_groups(
    affinities: &[ProjectAffinity],
) -> Vec<Vec<String>> {
    use std::collections::{HashMap, HashSet};

    if affinities.is_empty() {
        return Vec::new();
    }

    // Build adjacency list
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    for a in affinities {
        adj.entry(&a.tenant_a).or_default().insert(&a.tenant_b);
        adj.entry(&a.tenant_b).or_default().insert(&a.tenant_a);
    }

    // BFS to find connected components
    let mut visited: HashSet<&str> = HashSet::new();
    let mut groups: Vec<Vec<String>> = Vec::new();

    for start in adj.keys() {
        if visited.contains(*start) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = vec![*start];

        while let Some(node) = queue.pop() {
            if !visited.insert(node) {
                continue;
            }
            component.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        queue.push(neighbor);
                    }
                }
            }
        }

        component.sort();
        groups.push(component);
    }

    groups.sort_by(|a, b| a[0].cmp(&b[0]));
    groups
}

/// Generate a deterministic group_id for an affinity group.
///
/// Uses sorted tenant_ids to produce a stable hash.
fn affinity_group_id(members: &[String]) -> String {
    use sha2::{Digest, Sha256};

    let mut sorted = members.to_vec();
    sorted.sort();
    let input = sorted.join("|");
    let hash = Sha256::digest(input.as_bytes());
    format!("affinity:{:x}", hash)[..24].to_string()
}

/// Compute the mean embedding of a group of projects.
fn group_mean_embedding(
    members: &[String],
    embeddings: &[(String, Vec<f32>)],
) -> Option<Vec<f32>> {
    let member_embeddings: Vec<Vec<f32>> = members
        .iter()
        .filter_map(|t| {
            embeddings
                .iter()
                .find(|(id, _)| id == t)
                .map(|(_, e)| e.clone())
        })
        .collect();

    aggregate_document_embedding(&member_embeddings)
}

// ─── AffinityGrouper ───────────────────────────────────────────────────

/// Main affinity grouper: computes and stores project affinity groups.
pub struct AffinityGrouper {
    pool: SqlitePool,
    config: AffinityConfig,
}

impl AffinityGrouper {
    pub fn new(pool: SqlitePool, config: AffinityConfig) -> Self {
        Self { pool, config }
    }

    /// Compute and store affinity groups for all projects.
    ///
    /// 1. Load all project aggregate embeddings
    /// 2. Compute pairwise cosine similarities
    /// 3. Build connected components
    /// 4. Store in project_groups with group_type = "affinity"
    ///
    /// Returns the number of groups created.
    pub async fn compute_affinity_groups(&self) -> Result<usize, sqlx::Error> {
        // Clear existing affinity groups
        sqlx::query("DELETE FROM project_groups WHERE group_type = 'affinity'")
            .execute(&self.pool)
            .await?;

        // Also clear stale labels
        sqlx::query("DELETE FROM affinity_labels WHERE 1=1")
            .execute(&self.pool)
            .await?;

        let embeddings = load_all_project_embeddings(&self.pool).await?;
        if embeddings.len() < 2 {
            info!(
                projects = embeddings.len(),
                "Not enough projects for affinity grouping"
            );
            return Ok(0);
        }

        let affinities =
            compute_pairwise_affinities(&embeddings, self.config.similarity_threshold);

        if affinities.is_empty() {
            info!("No project pairs exceed affinity threshold");
            return Ok(0);
        }

        let groups = build_affinity_groups(&affinities);
        let mut groups_created = 0;

        for members in &groups {
            if members.len() < 2 {
                continue;
            }

            let group_id = affinity_group_id(members);

            // Compute mean similarity for confidence score
            let mean_sim = compute_group_mean_similarity(members, &affinities);

            for tenant_id in members {
                project_groups_schema::add_to_group(
                    &self.pool,
                    &group_id,
                    tenant_id,
                    "affinity",
                    mean_sim,
                )
                .await?;
            }

            debug!(
                group_id = group_id.as_str(),
                members = members.len(),
                mean_similarity = mean_sim,
                "Created affinity group"
            );
            groups_created += 1;
        }

        info!(
            projects = embeddings.len(),
            pairs = affinities.len(),
            groups = groups_created,
            threshold = self.config.similarity_threshold,
            "Affinity group computation complete"
        );

        Ok(groups_created)
    }

    /// Compute affinity groups and label them using taxonomy classification.
    ///
    /// Uses `Tier2Tagger` to classify each group's mean embedding against
    /// the taxonomy, producing a human-readable group label.
    pub async fn compute_and_label_groups(
        &self,
        tagger: &Tier2Tagger,
    ) -> Result<usize, sqlx::Error> {
        let groups_created = self.compute_affinity_groups().await?;

        if groups_created == 0 {
            return Ok(0);
        }

        // Load all embeddings again for label computation
        let embeddings = load_all_project_embeddings(&self.pool).await?;

        // Load the groups we just created
        let group_rows = sqlx::query(
            r#"
            SELECT DISTINCT group_id
            FROM project_groups
            WHERE group_type = 'affinity'
            ORDER BY group_id
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        for row in &group_rows {
            let group_id: String = row.get("group_id");

            // Get members of this group
            let member_rows = sqlx::query(
                "SELECT tenant_id FROM project_groups WHERE group_id = ? ORDER BY tenant_id",
            )
            .bind(&group_id)
            .fetch_all(&self.pool)
            .await?;

            let members: Vec<String> = member_rows
                .iter()
                .map(|r| r.get::<String, _>("tenant_id"))
                .collect();

            // Compute group mean embedding
            if let Some(group_embedding) = group_mean_embedding(&members, &embeddings) {
                let matches = tagger.classify(&group_embedding);
                if let Some(best) = matches.first() {
                    store_affinity_label(
                        &self.pool,
                        &group_id,
                        &best.term,
                        &best.category,
                        best.score,
                    )
                    .await?;

                    debug!(
                        group_id = group_id.as_str(),
                        label = best.term.as_str(),
                        category = best.category.as_str(),
                        score = best.score,
                        "Labeled affinity group"
                    );
                }
            }
        }

        Ok(groups_created)
    }

    /// Get the affinity group(s) for a specific project.
    ///
    /// Returns group_ids and their labels (if available).
    pub async fn get_project_affinity_groups(
        &self,
        tenant_id: &str,
    ) -> Result<Vec<AffinityGroupInfo>, sqlx::Error> {
        let rows = sqlx::query(
            r#"
            SELECT pg.group_id, pg.confidence, al.label, al.category, al.score
            FROM project_groups pg
            LEFT JOIN affinity_labels al ON pg.group_id = al.group_id
            WHERE pg.tenant_id = ? AND pg.group_type = 'affinity'
            ORDER BY pg.confidence DESC
            "#,
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;

        let groups: Vec<AffinityGroupInfo> = rows
            .iter()
            .map(|r| AffinityGroupInfo {
                group_id: r.get("group_id"),
                confidence: r.get("confidence"),
                label: r.get("label"),
                category: r.get("category"),
                label_score: r.get("score"),
            })
            .collect();

        Ok(groups)
    }
}

/// Info about an affinity group a project belongs to.
#[derive(Debug, Clone)]
pub struct AffinityGroupInfo {
    pub group_id: String,
    pub confidence: f64,
    pub label: Option<String>,
    pub category: Option<String>,
    pub label_score: Option<f64>,
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Compute mean similarity among all pairs within a group.
fn compute_group_mean_similarity(
    members: &[String],
    affinities: &[ProjectAffinity],
) -> f64 {
    let mut total = 0.0;
    let mut count = 0;

    for a in affinities {
        let a_in = members.contains(&a.tenant_a);
        let b_in = members.contains(&a.tenant_b);
        if a_in && b_in {
            total += a.similarity;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Store a label for an affinity group.
async fn store_affinity_label(
    pool: &SqlitePool,
    group_id: &str,
    label: &str,
    category: &str,
    score: f64,
) -> Result<(), sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();

    sqlx::query(
        r#"
        INSERT OR REPLACE INTO affinity_labels
            (group_id, label, category, score, updated_at)
        VALUES (?, ?, ?, ?, ?)
        "#,
    )
    .bind(group_id)
    .bind(label)
    .bind(category)
    .bind(score)
    .bind(&now)
    .execute(pool)
    .await?;

    Ok(())
}

/// Load the label for an affinity group.
pub async fn load_affinity_label(
    pool: &SqlitePool,
    group_id: &str,
) -> Result<Option<(String, String, f64)>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT label, category, score FROM affinity_labels WHERE group_id = ?",
    )
    .bind(group_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| {
        (
            r.get::<String, _>("label"),
            r.get::<String, _>("category"),
            r.get::<f64, _>("score"),
        )
    }))
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        sqlx::query(crate::project_groups_schema::CREATE_PROJECT_GROUPS_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for idx in crate::project_groups_schema::CREATE_PROJECT_GROUPS_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }

        pool
    }

    // ─── Blob serialization ──────────────────────────────────────────

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

    // ─── Store / load ────────────────────────────────────────────────

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

    // ─── Pairwise affinity ───────────────────────────────────────────

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
            ("proj-b".to_string(), vec![0.95, 0.31]),  // cos sim with a ≈ 0.95
            ("proj-c".to_string(), vec![0.0, 1.0]),     // orthogonal to a
        ];

        let affinities = compute_pairwise_affinities(&embeddings, 0.7);
        assert_eq!(affinities.len(), 1);
        assert_eq!(affinities[0].tenant_a, "proj-a");
        assert_eq!(affinities[0].tenant_b, "proj-b");
    }

    // ─── Connected components ────────────────────────────────────────

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

    // ─── Group ID ────────────────────────────────────────────────────

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

    // ─── Group mean similarity ───────────────────────────────────────

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

    // ─── AffinityGrouper integration ─────────────────────────────────

    #[tokio::test]
    async fn test_grouper_similar_projects() {
        let pool = setup_pool().await;

        // Two very similar embeddings (cosine sim ≈ 1.0)
        store_project_embedding(&pool, "proj-a", &[1.0, 0.0, 0.0], 5, None)
            .await
            .unwrap();
        store_project_embedding(&pool, "proj-b", &[0.99, 0.14, 0.0], 5, None)
            .await
            .unwrap();

        let grouper = AffinityGrouper::new(pool.clone(), AffinityConfig::default());
        let groups = grouper.compute_affinity_groups().await.unwrap();
        assert_eq!(groups, 1);

        let members = project_groups_schema::get_group_members(&pool, "proj-a")
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

        let members = project_groups_schema::get_group_members(&pool, "proj-a")
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

    // ─── Affinity label storage ──────────────────────────────────────

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

    // ─── Group mean embedding ────────────────────────────────────────

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
        // Only "a" contributes → [1.0, 0.0]
        assert!((mean[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_group_mean_embedding_no_members() {
        let embeddings: Vec<(String, Vec<f32>)> = Vec::new();
        let members: Vec<String> = Vec::new();
        assert!(group_mean_embedding(&members, &embeddings).is_none());
    }
}
