//! Injectable trait abstractions for the `rules` MCP tool.
//!
//! Three traits are defined here:
//!
//! - [`RulesDaemon`]  — daemon I/O (ingest, enqueue, mirror writes, embed)
//! - [`RulesReader`]  — SQLite mirror read for list fallback
//! - [`RulesQdrant`]  — direct Qdrant operations for list + dup-check
//!
//! All three are injected by tests via mocks to avoid live gRPC/Qdrant/SQLite
//! dependencies.

use std::collections::HashMap;

use crate::qdrant::client::{QdrantReadClient, QdrantRetrievedPoint};
use crate::sqlite::rules_mirror::RulesMirrorEntry;

// ─────────────────────────────────────────────────────────────────────────────
// RulesDaemon
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over daemon I/O needed by the rules tool.
///
/// Injected by tests via a mock to avoid live gRPC dependencies.
pub trait RulesDaemon {
    /// Ingest text via DocumentService.
    ///
    /// Returns `Ok(true)` when the daemon accepted the text and `Ok(false)` on
    /// a soft failure (daemon returned success=false). Connectivity errors
    /// (`Err(is_connectivity_error=true, ...)`) trigger queue fallback.
    fn ingest_text(
        &mut self,
        content: String,
        collection_basename: String,
        tenant_id: String,
        document_id: String,
        metadata: HashMap<String, String>,
    ) -> impl std::future::Future<Output = Result<bool, (bool, String)>> + Send;

    /// Enqueue an item into the unified queue.
    ///
    /// Returns the `queue_id` assigned by the daemon.
    fn enqueue_item(
        &mut self,
        item_type: &str,
        op: &str,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata_json: Option<&str>,
    ) -> impl std::future::Future<Output = Result<String, String>> + Send;

    /// Upsert rule into mirror — fire-and-forget.
    ///
    /// Mirrors `upsertMirror` in rules-mutation-helpers.ts:98-113.
    fn upsert_rule_mirror(
        &mut self,
        rule_id: String,
        rule_text: String,
        scope: Option<String>,
        tenant_id: Option<String>,
        created_at: String,
        updated_at: String,
    ) -> impl std::future::Future<Output = ()> + Send;

    /// Delete rule from mirror — fire-and-forget.
    ///
    /// Mirrors `stateManager.deleteRulesMirror(label)` in rules-mutations.ts:122.
    fn delete_rule_mirror(
        &mut self,
        rule_id: String,
    ) -> impl std::future::Future<Output = ()> + Send;

    /// Generate a dense embedding for the given text.
    ///
    /// Mirrors `this.daemonClient.embedText({ text: content })` in rules.ts:125.
    /// Returns the embedding vector, or an empty `Vec` when the embedding
    /// fails (caller must treat empty as "skip dup-check").
    fn embed_text(&mut self, text: String) -> impl std::future::Future<Output = Vec<f32>> + Send;
}

// ─────────────────────────────────────────────────────────────────────────────
// RulesReader — SQLite mirror fallback for list
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over the rules list read path.
///
/// Separated from `RulesDaemon` so the list operation can use a SQLite
/// connection for the mirror read without a live daemon.
pub trait RulesReader {
    /// Read rules from the SQLite mirror, matching TS `readRulesFromMirror`.
    fn list_from_mirror(
        &self,
        scope: Option<&str>,
        tenant_id: Option<&str>,
        limit: usize,
    ) -> Vec<RulesMirrorEntry>;
}

// ─────────────────────────────────────────────────────────────────────────────
// RulesQdrant — direct Qdrant access for list (scroll) + dup-check (search)
// ─────────────────────────────────────────────────────────────────────────────

/// Qdrant read operations needed by the rules tool.
///
/// - `scroll_rules` — used by `list_rules` as the PRIMARY path
/// - `search_rules` — used by `find_similar_rules` for dup-check in `add`
pub trait RulesQdrant {
    /// Scroll the `rules` collection with an optional Qdrant filter JSON.
    ///
    /// Returns a list of `QdrantRetrievedPoint`s (no score).
    /// Any error is propagated as `Err(String)` — caller falls back to mirror.
    fn scroll_rules(
        &self,
        filter: Option<qdrant_client::qdrant::Filter>,
        limit: u32,
    ) -> impl std::future::Future<Output = Result<Vec<QdrantRetrievedPoint>, String>> + Send;

    /// Search the `rules` collection by dense vector for dup-check.
    ///
    /// Returns `(id, score, payload)` triples — caller rounds `score`.
    /// Any error is swallowed by the caller (returns empty `Vec`).
    fn search_rules(
        &self,
        vector: Vec<f32>,
        limit: u64,
        score_threshold: f32,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> impl std::future::Future<Output = Result<Vec<crate::qdrant::client::QdrantPoint>, String>> + Send;
}

// ─────────────────────────────────────────────────────────────────────────────
// Blanket impl: DaemonClient → RulesDaemon
// ─────────────────────────────────────────────────────────────────────────────

impl RulesDaemon for crate::grpc::DaemonClient {
    async fn ingest_text(
        &mut self,
        content: String,
        collection_basename: String,
        tenant_id: String,
        document_id: String,
        metadata: HashMap<String, String>,
    ) -> Result<bool, (bool, String)> {
        crate::grpc::DaemonClient::ingest_text(
            self,
            content,
            collection_basename,
            tenant_id,
            document_id,
            metadata,
        )
        .await
        .map(|resp| resp.success)
        .map_err(|e| {
            let msg = e.message().to_string();
            let is_conn = super::helpers::is_connectivity_error(&msg);
            (is_conn, msg)
        })
    }

    async fn enqueue_item(
        &mut self,
        item_type: &str,
        op: &str,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata_json: Option<&str>,
    ) -> Result<String, String> {
        crate::grpc::DaemonClient::enqueue_item(
            self,
            item_type.to_string(),
            op.to_string(),
            tenant_id.to_string(),
            collection.to_string(),
            payload_json.to_string(),
            branch.to_string(),
            metadata_json.map(str::to_string),
        )
        .await
        .map(|r| r.queue_id)
        .map_err(|e| e.to_string())
    }

    async fn upsert_rule_mirror(
        &mut self,
        rule_id: String,
        rule_text: String,
        scope: Option<String>,
        tenant_id: Option<String>,
        created_at: String,
        updated_at: String,
    ) {
        let _ = crate::grpc::DaemonClient::upsert_rule_mirror(
            self, rule_id, rule_text, scope, tenant_id, created_at, updated_at,
        )
        .await;
    }

    async fn delete_rule_mirror(&mut self, rule_id: String) {
        let _ = crate::grpc::DaemonClient::delete_rule_mirror(self, rule_id).await;
    }

    async fn embed_text(&mut self, text: String) -> Vec<f32> {
        match crate::grpc::DaemonClient::embed_text(self, &text).await {
            Ok(resp) if !resp.embedding.is_empty() => resp.embedding,
            _ => Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blanket impl: QdrantReadClient → RulesQdrant
// ─────────────────────────────────────────────────────────────────────────────

impl RulesQdrant for QdrantReadClient {
    async fn scroll_rules(
        &self,
        filter: Option<qdrant_client::qdrant::Filter>,
        limit: u32,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        self.scroll(wqm_common::constants::COLLECTION_RULES, filter, limit, None)
            .await
            .map(|(points, _next)| points)
            .map_err(|e| e.to_string())
    }

    async fn search_rules(
        &self,
        vector: Vec<f32>,
        limit: u64,
        score_threshold: f32,
        filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<Vec<crate::qdrant::client::QdrantPoint>, String> {
        self.search(
            wqm_common::constants::COLLECTION_RULES,
            crate::qdrant::fusion::DENSE_VECTOR_NAME,
            vector,
            limit,
            Some(score_threshold),
            filter,
        )
        .await
        .map_err(|e| e.to_string())
    }
}
