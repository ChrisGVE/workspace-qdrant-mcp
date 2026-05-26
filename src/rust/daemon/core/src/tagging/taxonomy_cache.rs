//! SQLite-backed taxonomy embedding cache for Tier 2 tagging.
//!
//! Persists pre-computed taxonomy term embeddings so they survive daemon
//! restarts. Cache invalidation uses SHA-256 of the taxonomy YAML content:
//! when the YAML changes, the hash changes and stale rows are purged.

use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use super::taxonomy::TaxonomyEntry;

/// Compute the SHA-256 hash of taxonomy YAML content for cache keying.
pub fn compute_taxonomy_hash(yaml_content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(yaml_content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Serialise a `Vec<f32>` embedding to little-endian bytes for SQLite BLOB storage.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialise little-endian bytes from a SQLite BLOB back to `Vec<f32>`.
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().expect("chunk is exactly 4 bytes");
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Result of a cache lookup: either a full cache hit with entries + embeddings,
/// or a miss indicating fresh embedding generation is needed.
pub enum CacheLookup {
    /// Cache hit: entries and their pre-computed embeddings.
    Hit {
        entries: Vec<TaxonomyEntry>,
        embeddings: Vec<Vec<f32>>,
    },
    /// Cache miss: the entries that need to be embedded.
    Miss { entries: Vec<TaxonomyEntry> },
}

/// Attempt to load cached taxonomy embeddings from SQLite.
///
/// Returns `CacheLookup::Hit` if all entries for the given taxonomy hash
/// exist in the cache, or `CacheLookup::Miss` if the cache is stale or
/// empty.
pub async fn load_cached_embeddings(
    pool: &SqlitePool,
    taxonomy_hash: &str,
    model_name: &str,
    entries: Vec<TaxonomyEntry>,
) -> CacheLookup {
    // Check if taxonomy_cache table exists (might not yet if migration hasn't run)
    let table_exists: bool = match sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='taxonomy_cache')",
    )
    .fetch_one(pool)
    .await
    {
        Ok(v) => v,
        Err(e) => {
            warn!("taxonomy_cache table check failed: {e}");
            return CacheLookup::Miss { entries };
        }
    };

    if !table_exists {
        debug!("taxonomy_cache table does not exist yet; cache miss");
        return CacheLookup::Miss { entries };
    }

    // Count cached rows for this taxonomy hash
    let cached_count: i64 = match sqlx::query_scalar(
        "SELECT COUNT(*) FROM taxonomy_cache WHERE taxonomy_hash = ?1 AND model_name = ?2",
    )
    .bind(taxonomy_hash)
    .bind(model_name)
    .fetch_one(pool)
    .await
    {
        Ok(c) => c,
        Err(e) => {
            warn!("taxonomy_cache count query failed: {e}");
            return CacheLookup::Miss { entries };
        }
    };

    if cached_count as usize != entries.len() {
        debug!(
            "taxonomy_cache count mismatch: cached={}, expected={}; cache miss",
            cached_count,
            entries.len()
        );
        return CacheLookup::Miss { entries };
    }

    // Load all cached embeddings for this hash, ordered by term_label for
    // deterministic mapping. We match by term_label against entries.
    let rows: Vec<(String, String, Vec<u8>)> = match sqlx::query_as(
        "SELECT term_label, category, embedding FROM taxonomy_cache \
         WHERE taxonomy_hash = ?1 AND model_name = ?2 ORDER BY term_label",
    )
    .bind(taxonomy_hash)
    .bind(model_name)
    .fetch_all(pool)
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("taxonomy_cache fetch failed: {e}");
            return CacheLookup::Miss { entries };
        }
    };

    // Build a lookup map from the cached rows
    let cached_map: std::collections::HashMap<String, (String, Vec<u8>)> = rows
        .into_iter()
        .map(|(label, cat, emb)| (label, (cat, emb)))
        .collect();

    // Verify every entry has a cached embedding
    let mut cached_entries = Vec::with_capacity(entries.len());
    let mut cached_embeddings = Vec::with_capacity(entries.len());

    for entry in &entries {
        match cached_map.get(&entry.term) {
            Some((_, emb_bytes)) => {
                cached_embeddings.push(bytes_to_embedding(emb_bytes));
                cached_entries.push(entry.clone());
            }
            None => {
                debug!(
                    "taxonomy_cache miss: term '{}' not found in cache",
                    entry.term
                );
                return CacheLookup::Miss { entries };
            }
        }
    }

    info!(
        "taxonomy_cache hit: loaded {} cached embeddings (hash={}..)",
        cached_entries.len(),
        &taxonomy_hash[..8.min(taxonomy_hash.len())]
    );

    CacheLookup::Hit {
        entries: cached_entries,
        embeddings: cached_embeddings,
    }
}

/// Persist freshly computed taxonomy embeddings to SQLite.
///
/// Deletes any stale entries (different taxonomy hash) first, then inserts
/// all entries for the current hash. Runs inside a transaction for atomicity.
pub async fn save_cached_embeddings(
    pool: &SqlitePool,
    taxonomy_hash: &str,
    model_name: &str,
    entries: &[TaxonomyEntry],
    embeddings: &[Vec<f32>],
) {
    assert_eq!(entries.len(), embeddings.len());

    // Check if taxonomy_cache table exists
    let table_exists: bool = match sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='taxonomy_cache')",
    )
    .fetch_one(pool)
    .await
    {
        Ok(v) => v,
        Err(e) => {
            warn!("taxonomy_cache table check failed on save: {e}");
            return;
        }
    };

    if !table_exists {
        warn!("taxonomy_cache table does not exist; skipping cache save");
        return;
    }

    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            warn!("taxonomy_cache transaction begin failed: {e}");
            return;
        }
    };

    // Delete stale entries (different taxonomy hash or model)
    if let Err(e) =
        sqlx::query("DELETE FROM taxonomy_cache WHERE taxonomy_hash != ?1 OR model_name != ?2")
            .bind(taxonomy_hash)
            .bind(model_name)
            .execute(&mut *tx)
            .await
    {
        warn!("taxonomy_cache stale deletion failed: {e}");
        let _ = tx.rollback().await;
        return;
    }

    // Insert current entries (INSERT OR REPLACE for idempotency)
    for (entry, embedding) in entries.iter().zip(embeddings.iter()) {
        let emb_bytes = embedding_to_bytes(embedding);
        if let Err(e) = sqlx::query(
            "INSERT OR REPLACE INTO taxonomy_cache \
             (taxonomy_hash, term_label, category, embedding, model_name) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .bind(taxonomy_hash)
        .bind(&entry.term)
        .bind(&entry.category)
        .bind(&emb_bytes)
        .bind(model_name)
        .execute(&mut *tx)
        .await
        {
            warn!("taxonomy_cache insert failed for '{}': {e}", entry.term);
            let _ = tx.rollback().await;
            return;
        }
    }

    if let Err(e) = tx.commit().await {
        warn!("taxonomy_cache commit failed: {e}");
        return;
    }

    info!(
        "taxonomy_cache: persisted {} embeddings (hash={}..)",
        entries.len(),
        &taxonomy_hash[..8.min(taxonomy_hash.len())]
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        pool
    }

    fn sample_entries() -> Vec<TaxonomyEntry> {
        vec![
            TaxonomyEntry {
                term: "rust programming".into(),
                category: "programming-languages".into(),
            },
            TaxonomyEntry {
                term: "web development".into(),
                category: "web-development".into(),
            },
        ]
    }

    fn sample_embeddings() -> Vec<Vec<f32>> {
        vec![vec![1.0, 0.0, 0.5, -0.3], vec![0.0, 1.0, -0.5, 0.7]]
    }

    #[test]
    fn test_compute_taxonomy_hash_deterministic() {
        let h1 = compute_taxonomy_hash("categories:\n  test:\n    - foo\n");
        let h2 = compute_taxonomy_hash("categories:\n  test:\n    - foo\n");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn test_compute_taxonomy_hash_different_content() {
        let h1 = compute_taxonomy_hash("content_a");
        let h2 = compute_taxonomy_hash("content_b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_embedding_serialization_roundtrip() {
        let original = vec![1.0f32, -0.5, 0.0, 3.14, f32::MIN, f32::MAX];
        let bytes = embedding_to_bytes(&original);
        let recovered = bytes_to_embedding(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_embedding_serialization_empty() {
        let original: Vec<f32> = Vec::new();
        let bytes = embedding_to_bytes(&original);
        assert!(bytes.is_empty());
        let recovered = bytes_to_embedding(&bytes);
        assert!(recovered.is_empty());
    }

    const TEST_MODEL: &str = "all-MiniLM-L6-v2";

    #[tokio::test]
    async fn test_cache_miss_on_empty_table() {
        let pool = setup_pool().await;
        let entries = sample_entries();
        let hash = "test_hash_001";

        let result = load_cached_embeddings(&pool, hash, TEST_MODEL, entries.clone()).await;
        assert!(matches!(result, CacheLookup::Miss { .. }));
    }

    #[tokio::test]
    async fn test_save_and_load_roundtrip() {
        let pool = setup_pool().await;
        let entries = sample_entries();
        let embeddings = sample_embeddings();
        let hash = "test_hash_002";

        save_cached_embeddings(&pool, hash, TEST_MODEL, &entries, &embeddings).await;

        let result = load_cached_embeddings(&pool, hash, TEST_MODEL, entries.clone()).await;
        match result {
            CacheLookup::Hit {
                entries: loaded_entries,
                embeddings: loaded_embeddings,
            } => {
                assert_eq!(loaded_entries.len(), 2);
                assert_eq!(loaded_embeddings.len(), 2);
                for (orig, loaded) in embeddings.iter().zip(loaded_embeddings.iter()) {
                    assert_eq!(orig, loaded);
                }
            }
            CacheLookup::Miss { .. } => panic!("Expected cache hit after save"),
        }
    }

    #[tokio::test]
    async fn test_cache_invalidation_on_hash_change() {
        let pool = setup_pool().await;
        let entries = sample_entries();
        let embeddings = sample_embeddings();

        save_cached_embeddings(&pool, "hash_A", TEST_MODEL, &entries, &embeddings).await;

        let result = load_cached_embeddings(&pool, "hash_B", TEST_MODEL, entries.clone()).await;
        assert!(matches!(result, CacheLookup::Miss { .. }));

        save_cached_embeddings(&pool, "hash_B", TEST_MODEL, &entries, &embeddings).await;

        let count_a: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM taxonomy_cache WHERE taxonomy_hash = 'hash_A'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count_a, 0, "Stale hash_A entries should be purged");

        let result = load_cached_embeddings(&pool, "hash_B", TEST_MODEL, entries.clone()).await;
        assert!(matches!(result, CacheLookup::Hit { .. }));
    }

    #[tokio::test]
    async fn test_cache_miss_on_partial_entries() {
        let pool = setup_pool().await;
        let entries = sample_entries();
        let embeddings = sample_embeddings();
        let hash = "test_hash_partial";

        save_cached_embeddings(&pool, hash, TEST_MODEL, &entries, &embeddings).await;

        let mut extended_entries = entries.clone();
        extended_entries.push(TaxonomyEntry {
            term: "extra term".into(),
            category: "extra".into(),
        });
        let result = load_cached_embeddings(&pool, hash, TEST_MODEL, extended_entries).await;
        assert!(matches!(result, CacheLookup::Miss { .. }));
    }

    #[tokio::test]
    async fn test_cache_miss_on_model_change() {
        let pool = setup_pool().await;
        let entries = sample_entries();
        let embeddings = sample_embeddings();
        let hash = "test_hash_model";

        save_cached_embeddings(&pool, hash, "model-a", &entries, &embeddings).await;

        let result = load_cached_embeddings(&pool, hash, "model-b", entries.clone()).await;
        assert!(
            matches!(result, CacheLookup::Miss { .. }),
            "Different model_name should cause cache miss"
        );

        let result = load_cached_embeddings(&pool, hash, "model-a", entries.clone()).await;
        assert!(
            matches!(result, CacheLookup::Hit { .. }),
            "Same model_name should cache hit"
        );
    }
}
