//! URL and scratchpad sub-handlers for the `store` tool.

use rmcp::model::CallToolResult;
use serde_json::{Map, Value};

use wqm_common::constants::{COLLECTION_SCRATCHPAD, TENANT_GLOBAL};
use wqm_common::timestamps::now_utc;

use crate::canonicalize::stable_stringify::stable_stringify;

use super::{StoreDaemon, StoreInput, StoreUrlScratchpadResult};
use crate::tools::envelope::ok_text;

/// Validate a URL string — mirrors `validateUrlInput` in store-handlers.ts:27-52.
///
/// Error messages match TS byte-for-byte:
/// - empty/non-string  → `"url is required when type is \"url\""` (line 29)
/// - parse failure     → `"url is malformed (failed to parse)"` (line 36)
/// - non-http(s) scheme → `"url must use http:// or https:// (got <scheme>:)"` (line 40)
///   Note: the suffix is `<scheme>:` (e.g. `ftp:`), NOT `ftp://`.
///   `parsed.protocol` in JS includes the colon but not `//`.
pub fn validate_url(raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("url is required when type is \"url\"".to_string());
    }
    // Attempt to parse as a URL (mirrors `new URL(trimmed)` in TS — line 33-36).
    // We extract scheme and host manually to avoid a url-crate dependency.
    let scheme_end = trimmed
        .find("://")
        .ok_or_else(|| "url is malformed (failed to parse)".to_string())?;
    let scheme_raw = &trimmed[..scheme_end];
    // Scheme must be non-empty and consist only of valid chars (alpha+digit+'-'+'+')
    if scheme_raw.is_empty()
        || !scheme_raw
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '-' || c == '+')
    {
        return Err("url is malformed (failed to parse)".to_string());
    }
    // Canonicalize scheme to lowercase: TS `new URL(...)` normalizes protocol to
    // lowercase (so "HTTP://x" has `protocol === 'http:'` and is accepted).
    // Mirror that by lowercasing before comparison.
    let scheme = scheme_raw.to_ascii_lowercase();
    // Non-http(s) scheme — error uses `<scheme>:` (parsed.protocol includes colon, not `://`)
    // store-handlers.ts:40: `url must use http:// or https:// (got ${parsed.protocol})`
    if scheme != "http" && scheme != "https" {
        return Err(format!("url must use http:// or https:// (got {scheme}:)"));
    }
    // Extract hostname
    let after_scheme = &trimmed[scheme_end + 3..];
    let host = after_scheme.split('/').next().unwrap_or("");
    // Remove port if present
    let host = host.split(':').next().unwrap_or(host);
    if host.is_empty() {
        return Err("url has empty hostname".to_string());
    }
    if host.chars().all(|c| c == '.' || c == ' ') {
        return Err("url has invalid hostname (dots/whitespace only)".to_string());
    }
    Ok(())
}

/// Build the URL queue payload — mirrors `buildUrlPayload` in store-handlers.ts:55-68.
///
/// Uses `stable_stringify` (sorted-key canonical JSON) so the byte sequence
/// matches the TypeScript write path (queue-operations.ts:36-47, :100) and the
/// daemon computes the same idempotency key.
fn build_url_payload(url: &str, library_name: Option<&str>, title: Option<&str>) -> String {
    let mut map = Map::new();
    map.insert("url".to_string(), Value::String(url.trim().to_string()));
    map.insert("crawl".to_string(), Value::Bool(false));
    map.insert("max_depth".to_string(), Value::Number(0.into()));
    map.insert("max_pages".to_string(), Value::Number(1.into()));
    if let Some(lib) = library_name {
        map.insert(
            "library_name".to_string(),
            Value::String(lib.trim().to_string()),
        );
    }
    if let Some(t) = title {
        map.insert("title".to_string(), Value::String(t.to_string()));
    }
    stable_stringify(&Value::Object(map))
}

/// Handle store type=url — mirrors `storeUrl` in store-handlers.ts:78-121.
pub(super) async fn store_url<D>(
    input: StoreInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: StoreDaemon,
{
    let url = input.url.as_deref().unwrap_or("").to_string();
    if let Err(e) = validate_url(&url) {
        let result = StoreUrlScratchpadResult {
            success: false,
            message: e,
            queue_id: None,
            collection: String::new(),
        };
        return ok_text(&result);
    }

    // TS: `libraryName?.trim() || sessionState.projectId || TENANT_GLOBAL`
    // (store-handlers.ts:92) — whitespace-only libraryName trims to '' which
    // is falsy, so it falls back to projectId/global.  Mirror by treating a
    // trimmed-empty libraryName as absent before tenant resolution.
    let library_name = input
        .library_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let collection = if library_name.is_some() {
        wqm_common::constants::COLLECTION_LIBRARIES.to_string()
    } else {
        COLLECTION_SCRATCHPAD.to_string()
    };
    let tenant_id = library_name
        .map(str::to_string)
        .or_else(|| session_project_id.map(str::to_string))
        .unwrap_or_else(|| TENANT_GLOBAL.to_string());

    let payload_json = build_url_payload(&url, library_name, input.title.as_deref());

    match daemon
        .enqueue_item(
            "url",
            "add",
            &tenant_id,
            &collection,
            &payload_json,
            "main",
            Some("{\"source\":\"mcp_store_url\"}"),
        )
        .await
    {
        Err(e) => {
            let result = StoreUrlScratchpadResult {
                success: false,
                message: format!("Failed to queue URL: {e}"),
                queue_id: None,
                collection,
            };
            ok_text(&result)
        }
        Ok(queue_id) => {
            let result = StoreUrlScratchpadResult {
                success: true,
                message: format!("URL queued for fetch and ingestion ({collection}/{tenant_id})"),
                queue_id: Some(queue_id),
                collection,
            };
            ok_text(&result)
        }
    }
}

/// Build the scratchpad queue payload — mirrors `buildScratchpadPayload`
/// in store-handlers.ts:150-158.
///
/// Uses `stable_stringify` (sorted-key canonical JSON) so the byte sequence
/// matches the TypeScript write path (queue-operations.ts:36-47, :100) and the
/// daemon computes the same idempotency key.
fn build_scratchpad_payload(content: &str, title: Option<&str>, tags: &[String]) -> String {
    let mut map = Map::new();
    map.insert(
        "content".to_string(),
        Value::String(content.trim().to_string()),
    );
    map.insert(
        "source_type".to_string(),
        Value::String("scratchpad".to_string()),
    );
    if let Some(t) = title {
        if !t.trim().is_empty() {
            map.insert("title".to_string(), Value::String(t.trim().to_string()));
        }
    }
    if !tags.is_empty() {
        let arr: Vec<Value> = tags.iter().map(|s| Value::String(s.clone())).collect();
        map.insert("tags".to_string(), Value::Array(arr));
    }
    stable_stringify(&Value::Object(map))
}

/// Handle store type=scratchpad — mirrors `storeScratchpad` in
/// store-handlers.ts:161-222.
pub(super) async fn store_scratchpad<D>(
    input: StoreInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: StoreDaemon,
{
    let content = match input.content.as_deref() {
        Some(c) if !c.trim().is_empty() => c.to_string(),
        _ => {
            return ok_text(&StoreUrlScratchpadResult {
                success: false,
                message: "content is required when type is \"scratchpad\"".to_string(),
                queue_id: None,
                collection: COLLECTION_SCRATCHPAD.to_string(),
            });
        }
    };

    let tenant_id = session_project_id
        .map(str::to_string)
        .unwrap_or_else(|| TENANT_GLOBAL.to_string());

    let payload_json = build_scratchpad_payload(&content, input.title.as_deref(), &input.tags);

    let result = daemon
        .enqueue_item(
            "text",
            "add",
            &tenant_id,
            COLLECTION_SCRATCHPAD,
            &payload_json,
            "main",
            Some("{\"source\":\"mcp_store_scratchpad\"}"),
        )
        .await;

    match result {
        Err(e) => ok_text(&StoreUrlScratchpadResult {
            success: false,
            message: format!("Failed to queue scratchpad entry: {e}"),
            queue_id: None,
            collection: COLLECTION_SCRATCHPAD.to_string(),
        }),
        Ok(queue_id) => {
            // Fire-and-forget mirror write — store-handlers.ts:208
            let now = now_utc();
            let tags_json = if input.tags.is_empty() {
                "[]".to_string()
            } else {
                serde_json::to_string(&input.tags).unwrap_or_else(|_| "[]".to_string())
            };
            daemon
                .upsert_scratchpad_mirror(
                    uuid::Uuid::new_v4().to_string(),
                    content.trim().to_string(),
                    input.title.as_ref().and_then(|t| {
                        let trimmed = t.trim();
                        if trimmed.is_empty() {
                            None
                        } else {
                            Some(trimmed.to_string())
                        }
                    }),
                    Some(tags_json),
                    tenant_id.clone(),
                    now.clone(),
                    now,
                )
                .await;

            ok_text(&StoreUrlScratchpadResult {
                success: true,
                message: format!("Scratchpad entry queued for processing ({tenant_id})"),
                queue_id: Some(queue_id),
                collection: COLLECTION_SCRATCHPAD.to_string(),
            })
        }
    }
}
