/// Welford's online mean algorithm for incremental aggregate embeddings.
///
/// All computation in f64 for numerical stability. Storage remains f32.
/// Deletion path: decrement count and subtract removed chunk's contribution.
use sqlx::{Row, SqlitePool};
use tracing::{debug, warn};
use wqm_common::timestamps::now_utc;

use super::storage::{blob_to_embedding, embedding_to_blob};

/// State for an incremental aggregate embedding.
#[derive(Debug, Clone)]
pub struct AggregateState {
    pub mean: Vec<f64>,
    pub count: u64,
    pub deletion_count: u64,
}

impl AggregateState {
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            count: 0,
            deletion_count: 0,
        }
    }

    pub fn from_f32(embedding: &[f32], count: u64) -> Self {
        Self {
            mean: embedding.iter().map(|&v| v as f64).collect(),
            count,
            deletion_count: 0,
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.mean.iter().map(|&v| v as f32).collect()
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Add a new embedding to the running mean (Welford's update).
    pub fn add(&mut self, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.mean.len());
        self.count += 1;
        let n = self.count as f64;
        for (m, &x) in self.mean.iter_mut().zip(embedding.iter()) {
            let delta = x as f64 - *m;
            *m += delta / n;
        }
    }

    /// Remove an embedding's contribution from the running mean.
    ///
    /// When count reaches 0, resets to zero vector.
    pub fn remove(&mut self, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.mean.len());
        if self.count <= 1 {
            self.mean.iter_mut().for_each(|m| *m = 0.0);
            self.count = 0;
            self.deletion_count = 0;
            return;
        }
        self.count -= 1;
        self.deletion_count += 1;
        let n = self.count as f64;
        for (m, &x) in self.mean.iter_mut().zip(embedding.iter()) {
            let delta = *m - x as f64;
            *m += delta / n;
        }
    }

    /// Whether a full recomputation is recommended (>20% deletions).
    pub fn needs_recomputation(&self) -> bool {
        if self.count == 0 {
            return false;
        }
        let total = self.count + self.deletion_count;
        (self.deletion_count as f64 / total as f64) > 0.20
    }
}

/// Load aggregate state from project_embeddings table.
pub async fn load_aggregate_state(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<Option<AggregateState>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT embedding, dim, chunk_count FROM project_embeddings WHERE tenant_id = ?",
    )
    .bind(tenant_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.and_then(|r| {
        let blob: Vec<u8> = r.get("embedding");
        let dim: i64 = r.get("dim");
        let count: i64 = r.get("chunk_count");
        blob_to_embedding(&blob, dim as usize)
            .map(|emb| AggregateState::from_f32(&emb, count as u64))
    }))
}

/// Persist aggregate state back to project_embeddings.
pub async fn save_aggregate_state(
    pool: &SqlitePool,
    tenant_id: &str,
    state: &AggregateState,
) -> Result<(), sqlx::Error> {
    let now = now_utc();
    let f32_emb = state.to_f32();
    let blob = embedding_to_blob(&f32_emb);
    let dim = state.dim() as i64;
    let count = state.count as i64;

    sqlx::query(
        "INSERT OR REPLACE INTO project_embeddings \
         (tenant_id, embedding, dim, chunk_count, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )
    .bind(tenant_id)
    .bind(&blob)
    .bind(dim)
    .bind(count)
    .bind(&now)
    .execute(pool)
    .await?;

    debug!(
        tenant_id,
        dim = state.dim(),
        count = state.count,
        "Saved incremental aggregate embedding"
    );
    Ok(())
}

/// Incrementally update a project's aggregate embedding with a new chunk.
pub async fn update_aggregate_add(
    pool: &SqlitePool,
    tenant_id: &str,
    new_embedding: &[f32],
) -> Result<AggregateState, sqlx::Error> {
    let mut state = load_aggregate_state(pool, tenant_id)
        .await?
        .unwrap_or_else(|| AggregateState::new(new_embedding.len()));

    if state.dim() != new_embedding.len() {
        warn!(
            tenant_id,
            old_dim = state.dim(),
            new_dim = new_embedding.len(),
            "Dimension mismatch, resetting aggregate"
        );
        state = AggregateState::new(new_embedding.len());
    }

    state.add(new_embedding);
    save_aggregate_state(pool, tenant_id, &state).await?;
    Ok(state)
}

/// Decrementally update a project's aggregate embedding when a chunk is removed.
pub async fn update_aggregate_remove(
    pool: &SqlitePool,
    tenant_id: &str,
    removed_embedding: &[f32],
) -> Result<Option<AggregateState>, sqlx::Error> {
    let Some(mut state) = load_aggregate_state(pool, tenant_id).await? else {
        return Ok(None);
    };

    state.remove(removed_embedding);
    save_aggregate_state(pool, tenant_id, &state).await?;

    if state.needs_recomputation() {
        debug!(
            tenant_id,
            deletion_ratio =
                state.deletion_count as f64 / (state.count + state.deletion_count) as f64,
            "Aggregate needs full recomputation (>20% deletions)"
        );
    }

    Ok(Some(state))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(val: f32, dim: usize) -> Vec<f32> {
        vec![val; dim]
    }

    #[test]
    fn test_single_add() {
        let mut state = AggregateState::new(4);
        state.add(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(state.count, 1);
        assert_eq!(state.mean, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_two_adds_average() {
        let mut state = AggregateState::new(3);
        state.add(&[0.0, 0.0, 0.0]);
        state.add(&[2.0, 4.0, 6.0]);
        assert_eq!(state.count, 2);
        let expected = vec![1.0, 2.0, 3.0];
        for (a, b) in state.mean.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_numerical_stability_10k() {
        let dim = 384;
        let mut state = AggregateState::new(dim);
        for i in 0..10_000 {
            let val = (i as f32) / 10_000.0;
            state.add(&make_embedding(val, dim));
        }
        assert_eq!(state.count, 10_000);
        let expected_mean = 4999.5 / 10_000.0;
        for &m in &state.mean {
            assert!(
                (m - expected_mean as f64).abs() < 1e-6,
                "Mean {m} should be ~{expected_mean}"
            );
        }
    }

    #[test]
    fn test_remove_to_zero() {
        let mut state = AggregateState::new(3);
        state.add(&[1.0, 2.0, 3.0]);
        state.remove(&[1.0, 2.0, 3.0]);
        assert_eq!(state.count, 0);
        assert_eq!(state.mean, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add_remove_consistency() {
        let mut state = AggregateState::new(3);
        state.add(&[1.0, 0.0, 0.0]);
        state.add(&[3.0, 0.0, 0.0]);
        state.add(&[5.0, 0.0, 0.0]);
        // Mean should be 3.0
        assert!((state.mean[0] - 3.0).abs() < 1e-10);

        state.remove(&[5.0, 0.0, 0.0]);
        // Mean of [1.0, 3.0] = 2.0
        assert_eq!(state.count, 2);
        assert!((state.mean[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_f32_roundtrip() {
        let mut state = AggregateState::new(4);
        state.add(&[1.5, -0.5, 3.14, 0.0]);
        let f32_vec = state.to_f32();
        let restored = AggregateState::from_f32(&f32_vec, state.count);
        for (a, b) in state.mean.iter().zip(restored.mean.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_needs_recomputation() {
        let mut state = AggregateState::new(3);
        for _ in 0..10 {
            state.add(&[1.0, 1.0, 1.0]);
        }
        assert!(!state.needs_recomputation());

        // Remove 3 out of 10 (30% > 20% threshold)
        for _ in 0..3 {
            state.remove(&[1.0, 1.0, 1.0]);
        }
        assert!(state.needs_recomputation());
    }

    #[test]
    fn test_no_recomputation_at_zero() {
        let state = AggregateState::new(3);
        assert!(!state.needs_recomputation());
    }

    #[tokio::test]
    async fn test_db_roundtrip() {
        use crate::schema_version::SchemaManager;
        use sqlx::sqlite::SqlitePoolOptions;

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();

        let emb = vec![1.0f32, 2.0, 3.0, 4.0];
        update_aggregate_add(&pool, "test_tenant", &emb)
            .await
            .unwrap();

        let state = load_aggregate_state(&pool, "test_tenant")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(state.count, 1);
        assert_eq!(state.dim(), 4);
        let f32_mean = state.to_f32();
        for (a, b) in f32_mean.iter().zip(emb.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_incremental_add_remove_db() {
        use crate::schema_version::SchemaManager;
        use sqlx::sqlite::SqlitePoolOptions;

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();

        update_aggregate_add(&pool, "t1", &[2.0, 4.0])
            .await
            .unwrap();
        update_aggregate_add(&pool, "t1", &[4.0, 6.0])
            .await
            .unwrap();

        let state = load_aggregate_state(&pool, "t1").await.unwrap().unwrap();
        assert_eq!(state.count, 2);
        assert!((state.mean[0] - 3.0).abs() < 1e-6);
        assert!((state.mean[1] - 5.0).abs() < 1e-6);

        update_aggregate_remove(&pool, "t1", &[4.0, 6.0])
            .await
            .unwrap();
        let state = load_aggregate_state(&pool, "t1").await.unwrap().unwrap();
        assert_eq!(state.count, 1);
        assert!((state.mean[0] - 2.0).abs() < 1e-6);
    }
}
