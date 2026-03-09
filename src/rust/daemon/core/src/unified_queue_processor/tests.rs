//! Tests for the unified queue processor module.

#[cfg(test)]
mod tests {
    use crate::unified_queue_processor::config::{
        UnifiedProcessingMetrics, UnifiedProcessorConfig, WarmupState,
    };
    use crate::unified_queue_processor::error::UnifiedProcessorError;
    use crate::unified_queue_processor::UnifiedQueueProcessor;
    use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus, UnifiedQueueItem};
    use std::sync::Arc;

    #[test]
    fn test_unified_processor_config_default() {
        let config = UnifiedProcessorConfig::default();
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.poll_interval_ms, 500);
        assert_eq!(config.lease_duration_secs, 300);
        assert_eq!(config.max_retries, 3);
        assert!(config.worker_id.starts_with("unified-worker-"));
        // Fairness scheduler settings (asymmetric anti-starvation)
        assert!(config.fairness_enabled);
        assert_eq!(config.high_priority_batch, 10);
        assert_eq!(config.low_priority_batch, 3);
        // Resource limits (Task 504)
        assert_eq!(config.inter_item_delay_ms, 50);
        assert_eq!(config.max_concurrent_embeddings, 2);
        assert_eq!(config.max_memory_percent, 70);
    }

    #[test]
    fn test_unified_processing_metrics_default() {
        let metrics = UnifiedProcessingMetrics::default();
        assert_eq!(metrics.items_failed, 0);
        assert_eq!(metrics.queue_depth, 0);
        assert!(metrics.items_processed_by_type.is_empty());
    }

    #[test]
    fn test_unified_processor_error_display() {
        let err = UnifiedProcessorError::InvalidPayload("missing field".to_string());
        assert_eq!(err.to_string(), "Invalid payload: missing field");

        let err = UnifiedProcessorError::FileNotFound("/path/to/file".to_string());
        assert_eq!(err.to_string(), "File not found: /path/to/file");

        let err = UnifiedProcessorError::Storage("connection refused".to_string());
        assert_eq!(err.to_string(), "Storage error: connection refused");
    }

    /// Test WarmupState tracking (Task 577)
    #[test]
    fn test_warmup_state_tracking() {
        let warmup_state = WarmupState::new(5); // 5 second warmup window

        // Should be in warmup immediately
        assert!(warmup_state.is_in_warmup());
        assert_eq!(warmup_state.elapsed_secs(), 0);

        // Sleep 1 second, still in warmup
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert!(warmup_state.is_in_warmup());

        // Sleep past warmup window
        std::thread::sleep(std::time::Duration::from_secs(5));
        assert!(!warmup_state.is_in_warmup());
        assert!(warmup_state.elapsed_secs() >= 5);
    }

    /// Test embedding semaphore starts with warmup permits (Task 578)
    #[tokio::test]
    async fn test_embedding_semaphore_starts_with_warmup_permits() {
        let config = UnifiedProcessorConfig {
            warmup_max_concurrent_embeddings: 1,
            max_concurrent_embeddings: 2,
            ..Default::default()
        };

        // Create semaphore as done in UnifiedQueueProcessor::new
        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            config.warmup_max_concurrent_embeddings,
        ));

        // Should have exactly 1 permit available (warmup limit)
        assert_eq!(semaphore.available_permits(), 1);

        // Acquire the one warmup permit
        let _permit = semaphore.acquire().await.unwrap();
        assert_eq!(semaphore.available_permits(), 0);

        // Try to acquire another - should fail immediately (would block if we awaited)
        assert!(semaphore.try_acquire().is_err());
    }

    /// Test semaphore transition from warmup to normal limits (Task 578)
    #[tokio::test]
    async fn test_embedding_semaphore_transition_to_normal_limits() {
        let config = UnifiedProcessorConfig {
            warmup_max_concurrent_embeddings: 1,
            max_concurrent_embeddings: 3,
            ..Default::default()
        };

        // Start with warmup permits
        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            config.warmup_max_concurrent_embeddings,
        ));
        assert_eq!(semaphore.available_permits(), 1);

        // Simulate warmup ending: add permits to reach normal limit
        let permits_to_add =
            config.max_concurrent_embeddings - config.warmup_max_concurrent_embeddings;
        semaphore.add_permits(permits_to_add);

        // Should now have 3 total permits (normal limit)
        assert_eq!(semaphore.available_permits(), 3);

        // Can acquire 3 permits
        let _p1 = semaphore.acquire().await.unwrap();
        let _p2 = semaphore.acquire().await.unwrap();
        let _p3 = semaphore.acquire().await.unwrap();
        assert_eq!(semaphore.available_permits(), 0);

        // Fourth acquire would block
        assert!(semaphore.try_acquire().is_err());
    }

    /// Test warmup config defaults (Task 577)
    #[test]
    fn test_warmup_config_defaults() {
        let config = UnifiedProcessorConfig::default();
        assert_eq!(config.warmup_window_secs, 30);
        assert_eq!(config.warmup_max_concurrent_embeddings, 1);
        assert_eq!(config.warmup_inter_item_delay_ms, 200);
        assert_eq!(config.max_concurrent_embeddings, 2);
        assert_eq!(config.inter_item_delay_ms, 50);
    }

    /// Test that warmup limits are more restrictive than normal limits (Task 578)
    #[test]
    fn test_warmup_limits_are_more_restrictive() {
        let config = UnifiedProcessorConfig::default();
        assert!(
            config.warmup_max_concurrent_embeddings <= config.max_concurrent_embeddings,
            "Warmup max_concurrent_embeddings should be <= normal limit"
        );
        assert!(
            config.warmup_inter_item_delay_ms >= config.inter_item_delay_ms,
            "Warmup inter_item_delay should be >= normal delay (slower processing)"
        );
    }

    // =========================================================================
    // DeleteDocument payload contract tests
    // =========================================================================

    /// Helper to create a minimal UnifiedQueueItem for testing
    fn make_delete_document_item(payload_json: &str) -> UnifiedQueueItem {
        UnifiedQueueItem {
            queue_id: "test-queue-id".to_string(),
            idempotency_key: "test-idempotency".to_string(),
            item_type: ItemType::Doc,
            op: QueueOperation::Delete,
            tenant_id: "test-tenant".to_string(),
            collection: "projects".to_string(),
            status: QueueStatus::InProgress,
            branch: "main".to_string(),
            payload_json: payload_json.to_string(),
            metadata: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: None,
            search_status: None,
            decision_json: None,
        }
    }

    #[test]
    fn test_delete_document_payload_uses_document_id_key() {
        // Validates that the payload field is "document_id" (not "doc_id")
        // matching the validation contract in queue_operations.rs
        let item = make_delete_document_item(r#"{"document_id":"doc-abc-123"}"#);
        let payload = item
            .parse_delete_document_payload()
            .expect("Should parse with document_id key");
        assert_eq!(payload.document_id, "doc-abc-123");
    }

    #[test]
    fn test_delete_document_payload_rejects_wrong_key() {
        // The old code used "doc_id" which would never match the validated payload
        let item = make_delete_document_item(r#"{"doc_id":"doc-abc-123"}"#);
        let payload = item.parse_delete_document_payload();
        // serde will deserialize but document_id will be missing/empty
        // since DeleteDocumentPayload requires "document_id"
        assert!(
            payload.is_err() || payload.unwrap().document_id.is_empty(),
            "Payload with 'doc_id' should fail or produce empty document_id"
        );
    }

    #[test]
    fn test_delete_document_payload_invalid_json() {
        let item = make_delete_document_item("not valid json");
        let result = item.parse_delete_document_payload();
        assert!(result.is_err(), "Invalid JSON should fail deserialization");
    }

    #[test]
    fn test_delete_document_payload_with_point_ids() {
        let item =
            make_delete_document_item(r#"{"document_id":"doc-xyz","point_ids":["p1","p2"]}"#);
        let payload = item
            .parse_delete_document_payload()
            .expect("Should parse with point_ids");
        assert_eq!(payload.document_id, "doc-xyz");
        assert_eq!(payload.point_ids.len(), 2);
    }

    #[test]
    fn test_lsp_enrichment_status_lowercase_in_payload() {
        use crate::lsp::project_manager::{EnrichmentStatus, LspEnrichment};
        use crate::strategies::processing::file::lsp_payload::add_lsp_enrichment_to_payload;

        let mut payload = std::collections::HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        add_lsp_enrichment_to_payload(&mut payload, &enrichment);
        let status = payload
            .get("lsp_enrichment_status")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(status, "success", "lsp_enrichment_status must be lowercase");

        let mut payload2 = std::collections::HashMap::new();
        let enrichment2 = LspEnrichment {
            enrichment_status: EnrichmentStatus::Failed,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: Some("test error".to_string()),
        };

        add_lsp_enrichment_to_payload(&mut payload2, &enrichment2);
        let status2 = payload2
            .get("lsp_enrichment_status")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(status2, "failed", "lsp_enrichment_status must be lowercase");
    }

    #[test]
    fn test_file_chunk_tags_construction() {
        use crate::file_classification::is_test_file;
        use crate::DocumentType;

        // Simulate tag construction logic from process_file_item
        let build_tags =
            |file_type: Option<&str>, doc_type: &DocumentType, file_path: &str| -> Vec<String> {
                let mut tags = Vec::new();
                if let Some(ft) = file_type {
                    tags.push(ft.to_lowercase());
                }
                if let Some(lang) = doc_type.language() {
                    tags.push(lang.to_string());
                }
                if let Some(ext) = std::path::Path::new(file_path)
                    .extension()
                    .and_then(|e| e.to_str())
                {
                    tags.push(ext.to_lowercase());
                }
                if is_test_file(std::path::Path::new(file_path)) {
                    tags.push("test".to_string());
                }
                tags
            };

        // Rust test file
        let tags = build_tags(
            Some("code"),
            &DocumentType::Code("rust".to_string()),
            "/project/src/test_utils.rs",
        );
        assert_eq!(tags, vec!["code", "rust", "rs", "test"]);

        // Python non-test file
        let tags = build_tags(
            Some("code"),
            &DocumentType::Code("python".to_string()),
            "/project/src/main.py",
        );
        assert_eq!(tags, vec!["code", "python", "py"]);

        // Markdown file (no language)
        let tags = build_tags(Some("docs"), &DocumentType::Markdown, "/project/README.md");
        assert_eq!(tags, vec!["docs", "md"]);

        // File with uppercase extension
        let tags = build_tags(
            Some("code"),
            &DocumentType::Code("cpp".to_string()),
            "/project/main.CPP",
        );
        assert_eq!(tags, vec!["code", "cpp", "cpp"]);
    }

    #[test]
    fn test_classify_error_permanent_categories() {
        // FileNotFound -> permanent_gone
        let err = UnifiedProcessorError::FileNotFound("/missing.rs".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "permanent_gone"
        );

        // InvalidPayload -> permanent_data
        let err = UnifiedProcessorError::InvalidPayload("bad json".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "permanent_data"
        );

        // QueueOperation with missing watch_folder -> permanent_gone (tenant no longer exists)
        let err = UnifiedProcessorError::QueueOperation("no watch_folder found".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "permanent_gone"
        );

        // QueueOperation with validation -> permanent_data
        let err = UnifiedProcessorError::QueueOperation("validation failed".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "permanent_data"
        );

        // ProcessingFailed with permission denied -> permanent_gone
        let err = UnifiedProcessorError::ProcessingFailed("Permission denied".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "permanent_gone"
        );

        // ProcessingFailed with unsupported -> permanent_data
        let err = UnifiedProcessorError::ProcessingFailed("Unsupported file format".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "permanent_data"
        );
    }

    #[test]
    fn test_classify_error_transient_categories() {
        // Storage -> transient_infrastructure
        let err = UnifiedProcessorError::Storage("connection refused".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "transient_infrastructure"
        );

        // Embedding -> transient_resource
        let err = UnifiedProcessorError::Embedding("out of memory".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "transient_resource"
        );

        // Generic ProcessingFailed -> transient_infrastructure
        let err = UnifiedProcessorError::ProcessingFailed("timeout".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "transient_infrastructure"
        );

        // Generic QueueOperation -> transient_infrastructure
        let err = UnifiedProcessorError::QueueOperation("database locked".into());
        assert_eq!(
            UnifiedQueueProcessor::classify_error(&err),
            "transient_infrastructure"
        );
    }

    #[test]
    fn test_is_permanent_category() {
        assert!(UnifiedQueueProcessor::is_permanent_category(
            "permanent_gone"
        ));
        assert!(UnifiedQueueProcessor::is_permanent_category(
            "permanent_data"
        ));
        assert!(!UnifiedQueueProcessor::is_permanent_category(
            "transient_infrastructure"
        ));
        assert!(!UnifiedQueueProcessor::is_permanent_category(
            "transient_resource"
        ));
        assert!(!UnifiedQueueProcessor::is_permanent_category("partial"));
    }
}
