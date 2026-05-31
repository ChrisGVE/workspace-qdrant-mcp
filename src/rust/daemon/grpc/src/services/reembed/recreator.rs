use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tonic::Status;
use tracing::warn;

use workspace_qdrant_core::storage::{MultiTenantConfig, StorageClient};

/// Abstraction over Qdrant collection recreation so the reembed flow can
/// be unit-tested without a running Qdrant instance.
#[async_trait]
pub trait CollectionRecreator: Send + Sync {
    async fn recreate(&self, name: &str, dim: u64) -> Result<(), Status>;
}

/// Production recreator backed by `StorageClient`: delete the collection
/// (best-effort) then create at `dim`.
pub struct StorageClientRecreator {
    pub storage: Arc<StorageClient>,
}

#[async_trait]
impl CollectionRecreator for StorageClientRecreator {
    async fn recreate(&self, name: &str, dim: u64) -> Result<(), Status> {
        if let Err(e) = self.storage.delete_collection(name).await {
            // Treat missing collection as success; surface other errors.
            let msg = e.to_string();
            if !msg.to_lowercase().contains("not found")
                && !msg.to_lowercase().contains("doesn't exist")
            {
                warn!(
                    collection = name,
                    error = %msg,
                    "delete_collection during reembed returned non-fatal error"
                );
            }
        }
        // Recreate with the SAME named-vector schema as the daemon's
        // create-on-index path (`shared::ensure_collection` →
        // `create_multi_tenant_collection`): a named `dense` vector + named
        // `sparse` sparse-vector. The plain `create_collection` builds a
        // single UNNAMED dense vector with no sparse config, which makes
        // Qdrant decline every subsequent hybrid upsert with
        // "Not existing vector name error: dense" (and `sparse`) — silently,
        // because batch upserts run with `wait=false`. That left every
        // reembedded collection at 0 points with no operator-visible error.
        let config = MultiTenantConfig {
            vector_size: dim,
            ..MultiTenantConfig::default()
        };
        self.storage
            .create_multi_tenant_collection(name, &config)
            .await
            .map_err(|e| Status::internal(format!("create_multi_tenant_collection({name}): {e}")))
    }
}

/// Compute the idempotency key for a `(item_type='collection', op='reembed',
/// tenant_id='_system', collection=…, payload='{}')` queue item using the
/// canonical 32-hex truncation of SHA256.
pub(super) fn collection_reembed_idempotency_key(collection: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("collection|reembed|_system|{}|{}", collection, "{}").as_bytes());
    let digest = hasher.finalize();
    digest
        .iter()
        .take(16)
        .map(|b| format!("{:02x}", b))
        .collect()
}
