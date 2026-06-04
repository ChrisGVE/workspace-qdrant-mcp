//! Payload JSON builders for MCP queue operations.
//!
//! These functions produce the canonical `payload_json` string that is fed
//! into [`wqm_common::hashing::generate_idempotency_key`].  The JSON is
//! built via [`stable_stringify`] to guarantee byte-for-byte parity with the
//! TypeScript MCP server's live write path.
//!
//! # Live TypeScript references
//! - `buildRulePayload`  → `rules-mutation-helpers.ts` lines 41-63
//! - `buildStorePayload` → `store.ts` lines 228-234

use serde_json::{Map, Value};

use super::stable_stringify::stable_stringify;

// -----------------------------------------------------------------------
// Rule payload
// -----------------------------------------------------------------------

/// Input parameters for [`build_rule_payload`].
///
/// Field naming mirrors the TypeScript `operation` parameter of
/// `buildRulePayload` in `rules-mutation-helpers.ts`.
pub struct RulePayloadInput<'a> {
    /// `"add"` | `"update"` | `"remove"`
    pub action: &'a str,
    /// Rule label (identifier)
    pub label: &'a str,
    /// Rule content text — included only when `Some` and non-empty (truthy)
    pub content: Option<&'a str>,
    /// `"global"` | `"project"` — included only when `Some` and non-empty
    pub scope: Option<&'a str>,
    /// Project ID — included only when `Some` and non-empty
    pub project_id: Option<&'a str>,
    /// Rule title — included only when `Some` and non-empty
    /// (empty string `""` is dropped because `""` is falsy in JS)
    pub title: Option<&'a str>,
    /// Tags array.
    /// - `Some(vec![])` → `"tags":[]`  (empty array is **truthy** in JS)
    /// - `None`          → field omitted
    pub tags: Option<Vec<&'a str>>,
    /// Priority — included whenever `Some` (even `Some(0)`)
    pub priority: Option<i64>,
}

/// Build the canonical `payload_json` string for a rule queue operation.
///
/// Mirrors `buildRulePayload` in `rules-mutation-helpers.ts`:
///
/// ```ts
/// const payload: Record<string, unknown> = {
///   label: operation.label,
///   action: operation.action,
///   [FIELD_SOURCE_TYPE]: 'rule',
/// };
/// if (operation.content)    payload[FIELD_CONTENT]    = operation.content;
/// if (operation.scope)      payload['scope']          = operation.scope;
/// if (operation.projectId)  payload[FIELD_PROJECT_ID] = operation.projectId;
/// if (operation.title)      payload[FIELD_TITLE]      = operation.title;
/// if (operation.tags)       payload['tags']           = operation.tags;
/// if (operation.priority !== undefined) payload['priority'] = operation.priority;
/// ```
///
/// The resulting `Record` is then passed to `stableStringify` inside
/// `buildEnqueueRequest` (queue-operations.ts line 100).
pub fn build_rule_payload(input: RulePayloadInput<'_>) -> String {
    let mut map = Map::new();

    // Always-present fields
    map.insert(
        "action".to_string(),
        Value::String(input.action.to_string()),
    );
    map.insert("label".to_string(), Value::String(input.label.to_string()));
    map.insert("source_type".to_string(), Value::String("rule".to_string()));

    // Conditionally included (truthy — non-empty string)
    if let Some(c) = input.content {
        if !c.is_empty() {
            map.insert("content".to_string(), Value::String(c.to_string()));
        }
    }
    if let Some(s) = input.scope {
        if !s.is_empty() {
            map.insert("scope".to_string(), Value::String(s.to_string()));
        }
    }
    if let Some(pid) = input.project_id {
        if !pid.is_empty() {
            map.insert("project_id".to_string(), Value::String(pid.to_string()));
        }
    }
    if let Some(t) = input.title {
        if !t.is_empty() {
            map.insert("title".to_string(), Value::String(t.to_string()));
        }
    }

    // tags: Some(vec) → always included (even empty slice) because [] is truthy in JS
    // tags: None → omitted
    if let Some(tags) = input.tags {
        let arr: Vec<Value> = tags
            .into_iter()
            .map(|s| Value::String(s.to_string()))
            .collect();
        map.insert("tags".to_string(), Value::Array(arr));
    }

    // priority: included whenever Some (even Some(0))
    if let Some(p) = input.priority {
        map.insert("priority".to_string(), Value::Number(p.into()));
    }

    stable_stringify(&Value::Object(map))
}

// -----------------------------------------------------------------------
// Store payload
// -----------------------------------------------------------------------

/// Build the canonical `payload_json` string for a store (library) queue operation.
///
/// Mirrors `queueStoreOperation` in `store.ts` lines 228-234:
///
/// ```ts
/// const payload: Record<string, unknown> = {
///   [FIELD_CONTENT]:      params.content,
///   [FIELD_DOCUMENT_ID]:  params.documentId,
///   [FIELD_SOURCE_TYPE]:  params.sourceType,
///   metadata:             params.metadata,
///   [FIELD_LIBRARY_NAME]: params.libraryName,
/// };
/// ```
///
/// `metadata` is **always** present even when the map is empty (`{}`).
///
/// Field name constants (mirror `wqm_common::constants::field`):
/// - `FIELD_CONTENT`      = `"content"`
/// - `FIELD_DOCUMENT_ID`  = `"document_id"`
/// - `FIELD_SOURCE_TYPE`  = `"source_type"`
/// - `FIELD_LIBRARY_NAME` = `"library_name"`
pub fn build_store_payload(
    content: &str,
    document_id: &str,
    source_type: &str,
    metadata: &Map<String, Value>,
    library_name: &str,
) -> String {
    let mut map = Map::new();
    map.insert("content".to_string(), Value::String(content.to_string()));
    map.insert(
        "document_id".to_string(),
        Value::String(document_id.to_string()),
    );
    map.insert(
        "library_name".to_string(),
        Value::String(library_name.to_string()),
    );
    // metadata is ALWAYS present — even {} — per store.ts:228-234
    map.insert("metadata".to_string(), Value::Object(metadata.clone()));
    map.insert(
        "source_type".to_string(),
        Value::String(source_type.to_string()),
    );

    stable_stringify(&Value::Object(map))
}
