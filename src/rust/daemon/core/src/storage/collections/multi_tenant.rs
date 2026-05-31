//! Multi-tenant collection operations
//!
//! Creation of multi-tenant collections with optimized HNSW configuration,
//! payload indexing, and idempotent initialization of all canonical collections.

use std::collections::HashMap;

use qdrant_client::qdrant::{
    vectors_config, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance,
    FieldType, HnswConfigDiffBuilder, SparseVectorConfig, SparseVectorParams, VectorParams,
    VectorParamsBuilder, VectorParamsMap, VectorsConfig,
};
use tracing::{debug, error, info, warn};

use wqm_common::constants::{
    COLLECTION_IMAGES, COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES,
    COLLECTION_SCRATCHPAD,
};

use super::super::client::StorageClient;
use super::super::config::MultiTenantConfig;
use super::super::types::{MultiTenantInitResult, StorageError};

// ── Payload-index field manifests ───────────────────────────────────────────
//
// Single source of truth for which keyword-typed payload fields each canonical
// collection has indexed at startup. The init_*_collection helpers iterate
// over these and call create_payload_index for each entry.
//
// **Why constants instead of inline lists**: regression tests in `mod tests`
// assert that every field used in a production filter (e.g. the file_path +
// tenant_id pair in `storage/points/delete.rs::delete_points_by_filter`) is
// present here. Without these constants the assertion would need to inspect
// private function bodies, which is fragile. Changing a field name in either
// place forces a synchronous update in the test — the bug we just fixed
// (project_id indexed but tenant_id/file_path missing, causing 2.3s/delete
// because Qdrant fell back to full-scan) wouldn't have shipped.

/// Payload-index fields for the `projects` collection.
///
/// MUST include every keyword field used in a filter expression against
/// this collection. Adding a new filter field without updating this list
/// is a performance bug; see `multi_tenant_indexes` test below.
pub(crate) const PROJECTS_PAYLOAD_INDEX_FIELDS: &[&str] =
    &["tenant_id", "file_path", "branch", "project_id"];

/// Payload-index fields for the `libraries` collection.
pub(crate) const LIBRARIES_PAYLOAD_INDEX_FIELDS: &[&str] =
    &["tenant_id", "library_name", "file_path", "branch"];

/// Payload-index fields for the `rules` collection.
pub(crate) const RULES_PAYLOAD_INDEX_FIELDS: &[&str] = &["tenant_id"];

/// Payload-index fields for the `scratchpad` collection.
pub(crate) const SCRATCHPAD_PAYLOAD_INDEX_FIELDS: &[&str] = &["tenant_id"];

/// Payload-index fields for the `images` collection.
pub(crate) const IMAGES_PAYLOAD_INDEX_FIELDS: &[&str] = &["tenant_id", "source_document_id"];

/// Map a canonical collection name to its payload-index manifest.
///
/// Single source for the name→fields lookup used by the reembed recreate path
/// (`StorageClient::ensure_canonical_payload_indexes`). Returns an empty slice
/// for non-canonical names. Kept as a free function so the mapping is
/// unit-testable without a live Qdrant (see `mod tests`).
pub(crate) fn canonical_payload_index_fields(collection_name: &str) -> &'static [&'static str] {
    match collection_name {
        COLLECTION_PROJECTS => PROJECTS_PAYLOAD_INDEX_FIELDS,
        COLLECTION_LIBRARIES => LIBRARIES_PAYLOAD_INDEX_FIELDS,
        COLLECTION_RULES => RULES_PAYLOAD_INDEX_FIELDS,
        COLLECTION_SCRATCHPAD => SCRATCHPAD_PAYLOAD_INDEX_FIELDS,
        COLLECTION_IMAGES => IMAGES_PAYLOAD_INDEX_FIELDS,
        _ => &[],
    }
}

impl StorageClient {
    /// Create a multi-tenant collection with optimized HNSW configuration
    ///
    /// Collections are created with named vectors:
    /// - "dense": Dense semantic vectors (384 dimensions for all-MiniLM-L6-v2)
    /// - "sparse": Sparse BM25-style keyword vectors for hybrid search
    pub async fn create_multi_tenant_collection(
        &self,
        collection_name: &str,
        config: &MultiTenantConfig,
    ) -> Result<(), StorageError> {
        info!(
            "Creating multi-tenant collection: {} (vector_size={}, m={}, ef_construct={})",
            collection_name, config.vector_size, config.hnsw_m, config.hnsw_ef_construct
        );

        // Check if collection already exists (idempotency)
        if self.collection_exists(collection_name).await? {
            info!(
                "Collection {} already exists, skipping creation",
                collection_name
            );
            return Ok(());
        }

        let hnsw_config = HnswConfigDiffBuilder::default()
            .m(config.hnsw_m)
            .ef_construct(config.hnsw_ef_construct);

        let dense_vector_params: VectorParams =
            VectorParamsBuilder::new(config.vector_size, Distance::Cosine)
                .hnsw_config(hnsw_config)
                .on_disk(false)
                .build();

        let mut dense_vectors_map = HashMap::new();
        dense_vectors_map.insert("dense".to_string(), dense_vector_params);

        let named_vectors_config = VectorsConfig {
            config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
                map: dense_vectors_map,
            })),
        };

        let mut sparse_vectors_map = HashMap::new();
        sparse_vectors_map.insert(
            "sparse".to_string(),
            SparseVectorParams {
                index: None,
                modifier: None,
            },
        );

        let sparse_config = SparseVectorConfig {
            map: sparse_vectors_map,
        };

        let create_request = CreateCollectionBuilder::new(collection_name)
            .vectors_config(named_vectors_config)
            .sparse_vectors_config(sparse_config)
            .on_disk_payload(config.on_disk_payload)
            .shard_number(1)
            .replication_factor(1)
            .write_consistency_factor(1);

        self.retry_operation(|| async {
            self.client
                .create_collection(create_request.clone())
                .await
                .map_err(|e| StorageError::Collection(e.to_string()))
        })
        .await?;

        info!(
            "Successfully created multi-tenant collection with dense+sparse vectors: {}",
            collection_name
        );
        Ok(())
    }

    /// Create the images collection with CLIP 512-dim dense vectors only.
    pub async fn create_image_collection(
        &self,
        config: &MultiTenantConfig,
    ) -> Result<(), StorageError> {
        info!(
            "Creating images collection (CLIP 512-dim, dense-only, m={}, ef_construct={})",
            config.hnsw_m, config.hnsw_ef_construct
        );

        if self.collection_exists(COLLECTION_IMAGES).await? {
            info!(
                "Collection {} already exists, skipping creation",
                COLLECTION_IMAGES
            );
            return Ok(());
        }

        let hnsw_config = HnswConfigDiffBuilder::default()
            .m(config.hnsw_m)
            .ef_construct(config.hnsw_ef_construct);

        let dense_vector_params: VectorParams = VectorParamsBuilder::new(512, Distance::Cosine)
            .hnsw_config(hnsw_config)
            .on_disk(false)
            .build();

        let mut dense_vectors_map = HashMap::new();
        dense_vectors_map.insert("dense".to_string(), dense_vector_params);

        let named_vectors_config = VectorsConfig {
            config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
                map: dense_vectors_map,
            })),
        };

        let create_request = CreateCollectionBuilder::new(COLLECTION_IMAGES)
            .vectors_config(named_vectors_config)
            .on_disk_payload(config.on_disk_payload)
            .shard_number(1)
            .replication_factor(1)
            .write_consistency_factor(1);

        self.retry_operation(|| async {
            self.client
                .create_collection(create_request.clone())
                .await
                .map_err(|e| StorageError::Collection(e.to_string()))
        })
        .await?;

        info!("Successfully created images collection (512-dim CLIP, dense-only)");
        Ok(())
    }

    /// Create a payload index for efficient filtering
    pub async fn create_payload_index(
        &self,
        collection_name: &str,
        field_name: &str,
    ) -> Result<(), StorageError> {
        info!(
            "Creating payload index on {}.{}",
            collection_name, field_name
        );

        let index_request =
            CreateFieldIndexCollectionBuilder::new(collection_name, field_name, FieldType::Keyword);

        self.retry_operation(|| async {
            self.client
                .create_field_index(index_request.clone())
                .await
                .map_err(|e| {
                    StorageError::Collection(format!(
                        "Failed to create payload index on {}.{}: {}",
                        collection_name, field_name, e
                    ))
                })
        })
        .await?;

        info!(
            "Successfully created payload index on {}.{}",
            collection_name, field_name
        );
        Ok(())
    }

    /// Create a payload index, logging a warning on failure rather than
    /// propagating the error.
    ///
    /// Used by the init_* helpers below where each canonical collection needs
    /// several payload indexes. Qdrant returns success when the index already
    /// exists, so this helper is safe to call on every daemon startup —
    /// existing indexes are no-ops, missing ones get backfilled.
    async fn try_create_payload_index(&self, collection: &str, field: &str) {
        if let Err(e) = self.create_payload_index(collection, field).await {
            warn!(
                "Could not create payload index on {}.{} (may already exist): {}",
                collection, field, e
            );
        }
    }

    /// Backfill the canonical payload indexes for `collection_name`, matching
    /// the manifest the `init_*_collection` helpers create at startup.
    ///
    /// Idempotent — Qdrant treats an already-existing index as success, so this
    /// is safe to call repeatedly. The reembed recreate path needs it: that path
    /// *deletes* the collection before recreating, dropping the payload indexes
    /// built at startup by `initialize_multi_tenant_collections`. Without this
    /// backfill the reembedded collection keeps the correct vector schema but
    /// has no payload indexes, silently degrading every tenant-scoped filter /
    /// delete to a full-collection scan until the next daemon restart.
    ///
    /// Index-creation failures are logged and swallowed (same as startup) — a
    /// missing index is a performance issue, not a correctness one, and must not
    /// fail the reembed. Non-canonical names are a no-op.
    pub async fn ensure_canonical_payload_indexes(
        &self,
        collection_name: &str,
    ) -> Result<(), StorageError> {
        let fields = canonical_payload_index_fields(collection_name);
        if fields.is_empty() {
            debug!(
                "No canonical payload-index manifest for '{}'; skipping index backfill",
                collection_name
            );
            return Ok(());
        }
        for field in fields {
            self.try_create_payload_index(collection_name, field).await;
        }
        Ok(())
    }

    /// Initialize all multi-tenant collections with proper configuration
    ///
    /// Creates the unified collections: projects, libraries, rules, scratchpad,
    /// and images. This method is idempotent - existing collections are skipped.
    pub async fn initialize_multi_tenant_collections(
        &self,
        config: Option<MultiTenantConfig>,
    ) -> Result<MultiTenantInitResult, StorageError> {
        let config = config.unwrap_or_default();
        info!(
            "Initializing multi-tenant collections with config: {:?}",
            config
        );

        let mut result = MultiTenantInitResult::default();

        self.init_projects_collection(&config, &mut result).await?;
        self.init_libraries_collection(&config, &mut result).await?;
        self.init_rules_collection(&config, &mut result).await?;
        self.init_scratchpad_collection(&config, &mut result)
            .await?;
        self.init_images_collection(&config, &mut result).await?;

        info!("Multi-tenant collections initialized: {:?}", result);
        Ok(result)
    }

    async fn init_projects_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_PROJECTS, config)
            .await
            .map_err(|e| {
                error!("Failed to create {} collection: {}", COLLECTION_PROJECTS, e);
                e
            })?;
        result.projects_created = true;

        // Payload indexes for the fields actually used in filter queries.
        // Without these, every delete_by_filter / scoped search degrades to
        // a full collection scan (~2s per call on a 40k-point collection),
        // which starves the queue processor and blocks throughput.
        //
        // Field list lives in PROJECTS_PAYLOAD_INDEX_FIELDS so regression
        // tests can assert the production filter fields are present.
        for field in PROJECTS_PAYLOAD_INDEX_FIELDS {
            self.try_create_payload_index(COLLECTION_PROJECTS, field)
                .await;
        }
        result.projects_indexed = true;
        Ok(())
    }

    async fn init_libraries_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_LIBRARIES, config)
            .await
            .map_err(|e| {
                error!(
                    "Failed to create {} collection: {}",
                    COLLECTION_LIBRARIES, e
                );
                e
            })?;
        result.libraries_created = true;
        // Same rationale as init_projects_collection — see comments there.
        for field in LIBRARIES_PAYLOAD_INDEX_FIELDS {
            self.try_create_payload_index(COLLECTION_LIBRARIES, field)
                .await;
        }
        result.libraries_indexed = true;
        Ok(())
    }

    async fn init_rules_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_RULES, config)
            .await
            .map_err(|e| {
                error!("Failed to create {} collection: {}", COLLECTION_RULES, e);
                e
            })?;
        result.rules_created = true;
        // Rules are tenant-scoped; without the index, listing rules for a
        // tenant degrades to a full scan.
        for field in RULES_PAYLOAD_INDEX_FIELDS {
            self.try_create_payload_index(COLLECTION_RULES, field).await;
        }
        Ok(())
    }

    async fn init_scratchpad_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_SCRATCHPAD, config)
            .await
            .map_err(|e| {
                error!(
                    "Failed to create {} collection: {}",
                    COLLECTION_SCRATCHPAD, e
                );
                e
            })?;
        result.scratchpad_created = true;
        for field in SCRATCHPAD_PAYLOAD_INDEX_FIELDS {
            self.try_create_payload_index(COLLECTION_SCRATCHPAD, field)
                .await;
        }
        Ok(())
    }

    async fn init_images_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_image_collection(config).await.map_err(|e| {
            error!("Failed to create {} collection: {}", COLLECTION_IMAGES, e);
            e
        })?;
        result.images_created = true;
        for field in IMAGES_PAYLOAD_INDEX_FIELDS {
            self.try_create_payload_index(COLLECTION_IMAGES, field)
                .await;
        }
        Ok(())
    }
}

// ── Regression tests ────────────────────────────────────────────────────────
//
// These are pure unit tests on the payload-index manifest constants — no
// Qdrant, no async runtime needed. The integration test that hits a live
// Qdrant lives in tests/qdrant_payload_index_test.rs and is gated on
// connection availability.

#[cfg(test)]
mod tests {
    use super::*;

    /// Every keyword field used in a production filter MUST appear in the
    /// matching collection's payload-index manifest, or Qdrant falls back to
    /// a full-collection scan (the bug fixed in this commit: project_id was
    /// indexed but tenant_id+file_path — used by delete_points_by_filter —
    /// were not, costing ~2.3s per delete on a 40k-point collection).
    ///
    /// To keep this test honest: if you add a new payload filter in
    /// `storage/points/*.rs`, update both the manifest above AND this list.
    #[test]
    fn projects_indexes_cover_production_filter_fields() {
        // From storage/points/delete.rs::delete_points_by_filter
        let required = ["tenant_id", "file_path"];
        for field in required {
            assert!(
                PROJECTS_PAYLOAD_INDEX_FIELDS.contains(&field),
                "projects.{field} is used by a production filter but is not in \
                 PROJECTS_PAYLOAD_INDEX_FIELDS — delete_by_filter / scoped search \
                 will degrade to full-scan. Add it to the manifest.",
            );
        }
    }

    #[test]
    fn libraries_indexes_cover_production_filter_fields() {
        // Libraries collection uses the same delete_points_by_filter path.
        let required = ["tenant_id", "file_path"];
        for field in required {
            assert!(
                LIBRARIES_PAYLOAD_INDEX_FIELDS.contains(&field),
                "libraries.{field} missing from LIBRARIES_PAYLOAD_INDEX_FIELDS",
            );
        }
    }

    #[test]
    fn scratchpad_indexes_cover_tenant_isolation() {
        assert!(
            SCRATCHPAD_PAYLOAD_INDEX_FIELDS.contains(&"tenant_id"),
            "scratchpad needs tenant_id indexed for per-tenant scoping",
        );
    }

    #[test]
    fn rules_indexes_cover_tenant_isolation() {
        assert!(
            RULES_PAYLOAD_INDEX_FIELDS.contains(&"tenant_id"),
            "rules needs tenant_id indexed for per-tenant scoping",
        );
    }

    #[test]
    fn images_indexes_cover_tenant_isolation() {
        assert!(
            IMAGES_PAYLOAD_INDEX_FIELDS.contains(&"tenant_id"),
            "images needs tenant_id indexed for per-tenant scoping",
        );
    }

    /// Sanity: no manifest is empty. An empty manifest would silently disable
    /// the index-creation loop — much harder to spot in review than a missing
    /// field.
    #[test]
    fn no_payload_index_manifest_is_empty() {
        assert!(!PROJECTS_PAYLOAD_INDEX_FIELDS.is_empty());
        assert!(!LIBRARIES_PAYLOAD_INDEX_FIELDS.is_empty());
        assert!(!RULES_PAYLOAD_INDEX_FIELDS.is_empty());
        assert!(!SCRATCHPAD_PAYLOAD_INDEX_FIELDS.is_empty());
        assert!(!IMAGES_PAYLOAD_INDEX_FIELDS.is_empty());
    }

    /// Manifests must not contain duplicates — call_create_payload_index on a
    /// duplicate is harmless (idempotent) but wastes a roundtrip and signals
    /// a copy-paste mistake.
    #[test]
    fn no_payload_index_manifest_has_duplicates() {
        for (name, fields) in [
            ("projects", PROJECTS_PAYLOAD_INDEX_FIELDS),
            ("libraries", LIBRARIES_PAYLOAD_INDEX_FIELDS),
            ("rules", RULES_PAYLOAD_INDEX_FIELDS),
            ("scratchpad", SCRATCHPAD_PAYLOAD_INDEX_FIELDS),
            ("images", IMAGES_PAYLOAD_INDEX_FIELDS),
        ] {
            let mut sorted: Vec<&&str> = fields.iter().collect();
            sorted.sort();
            let original_len = sorted.len();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                original_len,
                "{name} payload-index manifest has duplicate fields: {fields:?}",
            );
        }
    }

    /// The reembed recreate path backfills payload indexes by collection name
    /// (`StorageClient::ensure_canonical_payload_indexes` →
    /// `canonical_payload_index_fields`). Lock the name→manifest mapping so a
    /// rename can't silently leave a reembedded collection index-less.
    #[test]
    fn canonical_payload_index_fields_maps_each_canonical_collection() {
        use wqm_common::constants::{
            COLLECTION_IMAGES, COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES,
            COLLECTION_SCRATCHPAD,
        };
        assert_eq!(
            canonical_payload_index_fields(COLLECTION_PROJECTS),
            PROJECTS_PAYLOAD_INDEX_FIELDS
        );
        assert_eq!(
            canonical_payload_index_fields(COLLECTION_LIBRARIES),
            LIBRARIES_PAYLOAD_INDEX_FIELDS
        );
        assert_eq!(
            canonical_payload_index_fields(COLLECTION_RULES),
            RULES_PAYLOAD_INDEX_FIELDS
        );
        assert_eq!(
            canonical_payload_index_fields(COLLECTION_SCRATCHPAD),
            SCRATCHPAD_PAYLOAD_INDEX_FIELDS
        );
        assert_eq!(
            canonical_payload_index_fields(COLLECTION_IMAGES),
            IMAGES_PAYLOAD_INDEX_FIELDS
        );
    }

    /// Unknown / non-canonical names map to an empty manifest (no-op backfill).
    #[test]
    fn canonical_payload_index_fields_unknown_is_empty() {
        assert!(canonical_payload_index_fields("not_a_canonical_collection").is_empty());
    }

    /// Every collection the reembed orchestrator recreates MUST have a non-empty
    /// payload-index manifest, or the recreate path leaves it index-less and
    /// tenant-scoped filters degrade to full scans. Mirrors `REEMBED_COLLECTIONS`
    /// in the grpc reembed orchestrator.
    #[test]
    fn reembed_canonical_collections_have_nonempty_manifests() {
        use wqm_common::constants::{
            COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
        };
        for name in [
            COLLECTION_PROJECTS,
            COLLECTION_LIBRARIES,
            COLLECTION_RULES,
            COLLECTION_SCRATCHPAD,
        ] {
            assert!(
                !canonical_payload_index_fields(name).is_empty(),
                "{name} is recreated on reembed but has no payload-index manifest",
            );
        }
    }
}
