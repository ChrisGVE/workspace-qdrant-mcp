//! Symbol co-occurrence graph for concept extraction.
//!
//! Tracks which symbols appear together in source files, building a graph
//! where edge weight = co-occurrence count. Degree centrality of a symbol
//! indicates its importance across the codebase — central symbols make
//! better concept tags.
//!
//! Integration point: between LSP extraction (step 3) and tag selection
//! (step 7) in the keyword extraction pipeline.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use sqlx::SqlitePool;
use tracing::debug;

use super::lsp_candidates::{self, LspCandidateConfig};
use crate::cooccurrence_schema;

/// Extract normalized symbol phrases from source code.
///
/// Wraps `lsp_candidates::extract_import_candidates` and returns
/// unique normalized phrases suitable for co-occurrence tracking.
pub fn extract_symbols(
    source: &str,
    language: &str,
    config: &LspCandidateConfig,
) -> Vec<String> {
    let candidates = lsp_candidates::extract_import_candidates(source, language, config);

    let mut seen = std::collections::HashSet::new();
    let mut symbols = Vec::new();

    for candidate in candidates {
        let phrase = candidate.phrase.trim().to_lowercase();
        if phrase.len() >= 2 && seen.insert(phrase.clone()) {
            symbols.push(phrase);
        }
    }

    symbols
}

/// Generate all N*(N-1)/2 pairs from a symbol list with canonical ordering.
///
/// Each pair `(a, b)` satisfies `a < b` to avoid storing duplicates.
pub fn generate_pairs(symbols: &[String]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for i in 0..symbols.len() {
        for j in (i + 1)..symbols.len() {
            let (a, b) = if symbols[i] < symbols[j] {
                (symbols[i].clone(), symbols[j].clone())
            } else {
                (symbols[j].clone(), symbols[i].clone())
            };
            pairs.push((a, b));
        }
    }
    pairs
}

/// Update the co-occurrence graph with symbols from a single file.
///
/// Generates all pairs and upserts into SQLite, incrementing counts.
pub async fn update_graph(
    pool: &SqlitePool,
    tenant_id: &str,
    collection: &str,
    symbols: &[String],
) -> Result<(), sqlx::Error> {
    let pairs = generate_pairs(symbols);
    if pairs.is_empty() {
        return Ok(());
    }

    debug!(
        "Updating co-occurrence graph: {} symbols → {} pairs for {}/{}",
        symbols.len(),
        pairs.len(),
        tenant_id,
        collection,
    );

    cooccurrence_schema::upsert_cooccurrences(pool, tenant_id, collection, &pairs).await
}

/// Compute a concept score combining centrality and semantic similarity.
///
/// `weight` controls the balance: 0.3 = 30% centrality + 70% semantic.
pub fn compute_concept_score(centrality: f64, semantic_score: f64, weight: f64) -> f64 {
    weight * centrality + (1.0 - weight) * semantic_score
}

/// TTL-based cache for degree centrality scores.
///
/// Avoids recomputing centrality from SQLite for every file in a batch.
/// Default TTL: 5 minutes.
pub struct CentralityCache {
    cache: HashMap<(String, String), (HashMap<String, f64>, Instant)>,
    ttl: Duration,
}

impl CentralityCache {
    /// Create a new cache with the specified TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            ttl,
        }
    }

    /// Get centrality scores, computing from SQLite if cache is stale or empty.
    pub async fn get_or_compute(
        &mut self,
        pool: &SqlitePool,
        tenant_id: &str,
        collection: &str,
    ) -> Result<HashMap<String, f64>, sqlx::Error> {
        let key = (tenant_id.to_string(), collection.to_string());

        if let Some((scores, fetched_at)) = self.cache.get(&key) {
            if fetched_at.elapsed() < self.ttl {
                return Ok(scores.clone());
            }
        }

        let scores =
            cooccurrence_schema::get_degree_centrality(pool, tenant_id, collection).await?;
        self.cache
            .insert(key, (scores.clone(), Instant::now()));

        Ok(scores)
    }

    /// Invalidate a specific tenant/collection entry.
    pub fn invalidate(&mut self, tenant_id: &str, collection: &str) {
        self.cache
            .remove(&(tenant_id.to_string(), collection.to_string()));
    }
}

impl Default for CentralityCache {
    fn default() -> Self {
        Self::new(Duration::from_secs(300)) // 5 minutes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_pairs_canonical_order() {
        let symbols = vec!["beta".to_string(), "alpha".to_string(), "gamma".to_string()];
        let pairs = generate_pairs(&symbols);

        // All pairs should have a < b
        for (a, b) in &pairs {
            assert!(a < b, "Expected {} < {}", a, b);
        }
    }

    #[test]
    fn test_generate_pairs_count() {
        let symbols: Vec<String> = (0..5).map(|i| format!("sym_{}", i)).collect();
        let pairs = generate_pairs(&symbols);
        // N*(N-1)/2 = 5*4/2 = 10
        assert_eq!(pairs.len(), 10);
    }

    #[test]
    fn test_generate_pairs_single() {
        let symbols = vec!["only_one".to_string()];
        let pairs = generate_pairs(&symbols);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_generate_pairs_empty() {
        let pairs = generate_pairs(&[]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_concept_score_formula() {
        // weight=0.3: 0.3 * centrality + 0.7 * semantic
        let score = compute_concept_score(1.0, 0.5, 0.3);
        assert!((score - 0.65).abs() < 1e-6, "Expected 0.65, got {}", score);

        // weight=0.0: pure semantic
        let score = compute_concept_score(1.0, 0.5, 0.0);
        assert!((score - 0.5).abs() < 1e-6);

        // weight=1.0: pure centrality
        let score = compute_concept_score(0.8, 0.5, 1.0);
        assert!((score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_centrality_cache_default_ttl() {
        let cache = CentralityCache::default();
        assert_eq!(cache.ttl, Duration::from_secs(300));
    }

    #[tokio::test]
    async fn test_centrality_cache_hit() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(crate::cooccurrence_schema::CREATE_SYMBOL_COOCCURRENCE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let mut cache = CentralityCache::new(Duration::from_secs(300));

        // First call should compute
        let scores = cache.get_or_compute(&pool, "t1", "projects").await.unwrap();
        assert!(scores.is_empty()); // no data

        // Insert data directly
        let now = wqm_common::timestamps::now_utc();
        sqlx::query(
            "INSERT INTO symbol_cooccurrence VALUES ('a', 'b', 't1', 'projects', 5, ?1)",
        )
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();

        // Second call should return cached (empty) result since TTL hasn't expired
        let scores = cache.get_or_compute(&pool, "t1", "projects").await.unwrap();
        assert!(scores.is_empty(), "Should use cached result");

        // Invalidate and re-fetch
        cache.invalidate("t1", "projects");
        let scores = cache.get_or_compute(&pool, "t1", "projects").await.unwrap();
        assert!(scores.contains_key("a"), "Should have recomputed after invalidation");
    }

    #[test]
    fn test_extract_symbols_rust() {
        let source = r#"
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::Deserialize;
"#;
        let config = LspCandidateConfig::default();
        let symbols = extract_symbols(source, "rust", &config);

        // Should contain normalized phrases from imports
        assert!(symbols.iter().any(|s| s.contains("hash")), "Should contain HashMap phrase: {:?}", symbols);
        assert!(symbols.iter().any(|s| s == "tokio"), "Should contain tokio: {:?}", symbols);
    }

    #[test]
    fn test_extract_symbols_deduplicates() {
        let source = r#"
use tokio::sync::RwLock;
use tokio::time::sleep;
"#;
        let config = LspCandidateConfig::default();
        let symbols = extract_symbols(source, "rust", &config);

        // "tokio" should appear only once
        let tokio_count = symbols.iter().filter(|s| s == &"tokio").count();
        assert_eq!(tokio_count, 1, "tokio should be deduplicated");
    }
}
