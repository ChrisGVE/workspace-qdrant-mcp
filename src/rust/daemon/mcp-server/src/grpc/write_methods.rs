//! QueueWriteService and TrackingWriteService RPC wrappers for [`DaemonClient`].
//!
//! Covers two service groups:
//!
//! ## QueueWriteService
//!
//! | Rust method    | Proto RPC                       | TS equivalent       |
//! |----------------|---------------------------------|---------------------|
//! | `enqueue_item` | `QueueWriteService::EnqueueItem`| `enqueueItem()`     |
//!
//! ## TrackingWriteService (mirror methods — fire-and-forget)
//!
//! | Rust method              | Proto RPC                                   | TS equivalent              |
//! |--------------------------|---------------------------------------------|----------------------------|
//! | `upsert_rule_mirror`     | `TrackingWriteService::UpsertRuleMirror`    | `upsertRuleMirror()`       |
//! | `delete_rule_mirror`     | `TrackingWriteService::DeleteRuleMirror`    | `deleteRuleMirror()`       |
//! | `upsert_scratchpad_mirror`| `TrackingWriteService::UpsertScratchpadMirror`| `upsertScratchpadMirror()` |
//! | `delete_scratchpad_mirror`| `TrackingWriteService::DeleteScratchpadMirror`| `deleteScratchpadMirror()` |
//!
//! # enqueue_item — payload_json contract
//!
//! The `payload_json` parameter must already be the **canonicalized** string
//! produced by `stable_stringify` (task 6 / `crate::canonicalize`).  This
//! wrapper forwards it verbatim.  It does **not** canonicalize.
//!
//! The daemon computes the idempotency key from the five-tuple
//! `(item_type, op, tenant_id, collection, payload_json)`.  The key is never
//! computed client-side.
//!
//! The `branch` field is required by the proto (`EnqueueItemRequest.branch`).
//! Callers that have no branch context should pass `"main"` (matching the TS
//! default in `queue-operations.ts:101`).
//!
//! # Fire-and-forget mirror methods
//!
//! `upsert_rule_mirror`, `delete_rule_mirror`, `upsert_scratchpad_mirror`, and
//! `delete_scratchpad_mirror` all have **fire-and-forget** semantics matching
//! the TS implementations in `rules-mirror-queries.ts` and
//! `scratchpad-mirror-queries.ts`:
//!
//! - TS uses `.catch(err => console.warn(...))` — errors are logged and
//!   **swallowed**; the returned `Promise<void>` is never awaited by the caller.
//! - The Rust wrappers return `Result<(), Status>` that is **always `Ok(())`**
//!   when the RPC fails; the error is logged via `tracing::warn!`.
//! - This matches the "rules_mirror is advisory" comment in
//!   `rules-mirror-queries.ts:49-50`.

use tonic::Status;
use tracing::warn;

use crate::proto::{
    DeleteRuleMirrorRequest, DeleteScratchpadMirrorRequest, EnqueueItemRequest,
    EnqueueItemResponse, UpsertRuleMirrorRequest, UpsertScratchpadMirrorRequest,
};

use super::client::DaemonClient;

// =============================================================================
// QueueWriteService
// =============================================================================

impl DaemonClient {
    /// Enqueue an item — mirrors TS `enqueueItem()` in `queue-operations.ts`.
    ///
    /// # Field mapping (queue-operations.ts:87-104)
    ///
    /// | Rust param     | Proto field    | TS field       |
    /// |----------------|----------------|----------------|
    /// | `item_type`    | `item_type`    | `item_type`    |
    /// | `op`           | `op`           | `op`           |
    /// | `tenant_id`    | `tenant_id`    | `tenant_id`    |
    /// | `collection`   | `collection`   | `collection`   |
    /// | `payload_json` | `payload_json` | `payload_json` |
    /// | `branch`       | `branch`       | `branch`       |
    /// | `metadata_json`| `metadata_json`| `metadata_json`|
    ///
    /// `payload_json` **must** be the already-canonicalized string from
    /// `stable_stringify` (task 6).  This method does not canonicalize.
    ///
    /// The daemon computes the idempotency key from the five-tuple
    /// `(item_type, op, tenant_id, collection, payload_json)`.
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn enqueue_item(
        &mut self,
        item_type: String,
        op: String,
        tenant_id: String,
        collection: String,
        payload_json: String,
        branch: String,
        metadata_json: Option<String>,
    ) -> Result<EnqueueItemResponse, Status> {
        let client = self.queue_write.clone();
        self.call("enqueueItem", None, move || {
            let mut c = client.clone();
            let req = EnqueueItemRequest {
                item_type: item_type.clone(),
                op: op.clone(),
                tenant_id: tenant_id.clone(),
                collection: collection.clone(),
                payload_json: payload_json.clone(),
                branch: branch.clone(),
                metadata_json: metadata_json.clone(),
            };
            async move {
                c.enqueue_item(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }
}

// =============================================================================
// TrackingWriteService — mirror methods (fire-and-forget)
// =============================================================================

impl DaemonClient {
    /// Upsert a rule into `rules_mirror` — fire-and-forget.
    ///
    /// Mirrors TS `upsertRuleMirror()` in `rules-mirror-queries.ts:26-52`.
    /// Field mapping (rules-mirror-queries.ts:32-47):
    ///
    /// | Rust param   | Proto field  | TS field     |
    /// |--------------|--------------|--------------|
    /// | `rule_id`    | `rule_id`    | `rule_id`    |
    /// | `rule_text`  | `rule_text`  | `rule_text`  |
    /// | `scope`      | `scope`      | `scope`      |
    /// | `tenant_id`  | `tenant_id`  | `tenant_id`  |
    /// | `created_at` | `created_at` | `created_at` |
    /// | `updated_at` | `updated_at` | `updated_at` |
    ///
    /// # Fire-and-forget
    /// On RPC failure the error is logged via `warn!` and `Ok(())` is returned.
    /// The mirror is advisory — errors must not break rule operations
    /// (rules-mirror-queries.ts:49-50).
    pub async fn upsert_rule_mirror(
        &mut self,
        rule_id: String,
        rule_text: String,
        scope: Option<String>,
        tenant_id: Option<String>,
        created_at: String,
        updated_at: String,
    ) -> Result<(), Status> {
        let client = self.tracking_write.clone();
        let result = self
            .call("upsertRuleMirror", None, move || {
                let mut c = client.clone();
                let req = UpsertRuleMirrorRequest {
                    rule_id: rule_id.clone(),
                    rule_text: rule_text.clone(),
                    scope: scope.clone(),
                    tenant_id: tenant_id.clone(),
                    created_at: created_at.clone(),
                    updated_at: updated_at.clone(),
                };
                async move {
                    c.upsert_rule_mirror(tonic::Request::new(req))
                        .await
                        .map(|r| r.into_inner())
                }
            })
            .await;
        if let Err(ref e) = result {
            warn!("upsertRuleMirror failed (advisory mirror): {}", e);
        }
        Ok(())
    }

    /// Delete a rule from `rules_mirror` — fire-and-forget.
    ///
    /// Mirrors TS `deleteRuleMirror()` in `rules-mirror-queries.ts:59-66`.
    /// The single field is `rule_id` (rules-mirror-queries.ts:62).
    ///
    /// # Fire-and-forget
    /// On RPC failure the error is logged via `warn!` and `Ok(())` is returned.
    pub async fn delete_rule_mirror(&mut self, rule_id: String) -> Result<(), Status> {
        let client = self.tracking_write.clone();
        let result = self
            .call("deleteRuleMirror", None, move || {
                let mut c = client.clone();
                let req = DeleteRuleMirrorRequest {
                    rule_id: rule_id.clone(),
                };
                async move {
                    c.delete_rule_mirror(tonic::Request::new(req))
                        .await
                        .map(|r| r.into_inner())
                }
            })
            .await;
        if let Err(ref e) = result {
            warn!("deleteRuleMirror failed (advisory mirror): {}", e);
        }
        Ok(())
    }

    /// Upsert a scratchpad entry into `scratchpad_mirror` — fire-and-forget.
    ///
    /// Mirrors TS `upsertScratchpadMirror()` in
    /// `scratchpad-mirror-queries.ts:25-52`.
    ///
    /// Field mapping (scratchpad-mirror-queries.ts:31-45):
    ///
    /// | Rust param      | Proto field     | TS field        |
    /// |-----------------|-----------------|-----------------|
    /// | `scratchpad_id` | `scratchpad_id` | `scratchpad_id` |
    /// | `content`       | `content`       | `content`       |
    /// | `title`         | `title`         | `title`         |
    /// | `tags`          | `tags`          | `tags`          |
    /// | `tenant_id`     | `tenant_id`     | `tenant_id`     |
    /// | `created_at`    | `created_at`    | `created_at`    |
    /// | `updated_at`    | `updated_at`    | `updated_at`    |
    ///
    /// `title` and `tags` are optional — only set when non-null in TS
    /// (scratchpad-mirror-queries.ts:46-47).
    ///
    /// # Fire-and-forget
    /// On RPC failure the error is logged via `warn!` and `Ok(())` is returned.
    pub async fn upsert_scratchpad_mirror(
        &mut self,
        scratchpad_id: String,
        content: String,
        title: Option<String>,
        tags: Option<String>,
        tenant_id: String,
        created_at: String,
        updated_at: String,
    ) -> Result<(), Status> {
        let client = self.tracking_write.clone();
        let result = self
            .call("upsertScratchpadMirror", None, move || {
                let mut c = client.clone();
                let req = UpsertScratchpadMirrorRequest {
                    scratchpad_id: scratchpad_id.clone(),
                    content: content.clone(),
                    title: title.clone(),
                    tags: tags.clone(),
                    tenant_id: tenant_id.clone(),
                    created_at: created_at.clone(),
                    updated_at: updated_at.clone(),
                };
                async move {
                    c.upsert_scratchpad_mirror(tonic::Request::new(req))
                        .await
                        .map(|r| r.into_inner())
                }
            })
            .await;
        if let Err(ref e) = result {
            warn!("upsertScratchpadMirror failed (advisory mirror): {}", e);
        }
        Ok(())
    }

    /// Delete a scratchpad entry from `scratchpad_mirror` — fire-and-forget.
    ///
    /// Mirrors TS `deleteScratchpadMirror()` in
    /// `scratchpad-mirror-queries.ts:58-66`.
    /// The single field is `scratchpad_id`
    /// (scratchpad-mirror-queries.ts:64).
    ///
    /// # Fire-and-forget
    /// On RPC failure the error is logged via `warn!` and `Ok(())` is returned.
    pub async fn delete_scratchpad_mirror(&mut self, scratchpad_id: String) -> Result<(), Status> {
        let client = self.tracking_write.clone();
        let result = self
            .call("deleteScratchpadMirror", None, move || {
                let mut c = client.clone();
                let req = DeleteScratchpadMirrorRequest {
                    scratchpad_id: scratchpad_id.clone(),
                };
                async move {
                    c.delete_scratchpad_mirror(tonic::Request::new(req))
                        .await
                        .map(|r| r.into_inner())
                }
            })
            .await;
        if let Err(ref e) = result {
            warn!("deleteScratchpadMirror failed (advisory mirror): {}", e);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "write_methods_tests.rs"]
mod tests;
