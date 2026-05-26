/// Taxonomy loading and Tier 2 tagger initialization at daemon startup.
use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::embedding::EmbeddingGenerator;

use super::taxonomy::load_taxonomy;
use super::tier2::{Tier2Config, Tier2Tagger};

const BUNDLED_TAXONOMY: &str = include_str!("../../../../../../assets/taxonomy.yaml");

/// Initialize Tier 2 tagger using the bundled taxonomy YAML.
///
/// Tries cached embeddings first (by taxonomy hash + model name).
/// On cache miss, embeds all taxonomy terms (~500ms) and persists.
pub async fn initialize_tier2_tagger(
    pool: &SqlitePool,
    embedding_generator: &EmbeddingGenerator,
) -> Option<Arc<Tier2Tagger>> {
    let entries = match load_taxonomy(BUNDLED_TAXONOMY) {
        Ok(e) => e,
        Err(e) => {
            warn!("Failed to load bundled taxonomy: {e}");
            return None;
        }
    };

    info!(
        "Initializing Tier 2 tagger with {} taxonomy entries",
        entries.len()
    );

    let config = Tier2Config::default();

    match Tier2Tagger::from_cache_or_embed(
        entries,
        BUNDLED_TAXONOMY,
        embedding_generator,
        pool,
        config,
    )
    .await
    {
        Ok(tagger) => {
            info!("Tier 2 tagger initialized successfully");
            Some(Arc::new(tagger))
        }
        Err(e) => {
            warn!("Tier 2 tagger initialization failed: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_taxonomy_parses() {
        let entries = load_taxonomy(BUNDLED_TAXONOMY).unwrap();
        assert!(
            entries.len() >= 100,
            "Bundled taxonomy should have >=100 entries, got {}",
            entries.len()
        );
    }

    #[test]
    fn test_bundled_taxonomy_categories() {
        let entries = load_taxonomy(BUNDLED_TAXONOMY).unwrap();
        let categories: std::collections::HashSet<&str> =
            entries.iter().map(|e| e.category.as_str()).collect();
        assert!(
            categories.contains("programming-languages"),
            "Should contain programming-languages category"
        );
    }
}
