//! Payloads for queue operations: delete and collection-level operations

use serde::{Deserialize, Serialize};

/// Payload for tenant delete operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteTenantPayload {
    /// Tenant ID to delete
    pub tenant_id_to_delete: String,
    /// Reason for deletion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Payload for doc delete operations (delete by document ID)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteDocumentPayload {
    /// Document identifier (UUID or path)
    pub document_id: String,
    /// Specific point IDs to delete (optional)
    #[serde(default)]
    pub point_ids: Vec<String>,
}

/// Payload for collection-level operations (uplift, reset)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionPayload {
    /// Name of the Qdrant collection
    pub collection_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_payload_serde() {
        let payload = CollectionPayload {
            collection_name: "projects".to_string(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("projects"));

        let back: CollectionPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.collection_name, "projects");
    }
}
