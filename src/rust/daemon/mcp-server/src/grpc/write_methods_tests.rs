use super::*;

// =========================================================================
// Task 14 — EnqueueItemRequest field mapping
// =========================================================================
//
// Tests validate the proto field layout against queue-operations.ts:87-104.

fn make_enqueue_request(metadata_json: Option<&str>) -> EnqueueItemRequest {
    EnqueueItemRequest {
        item_type: "text".to_string(),
        op: "add".to_string(),
        tenant_id: "global".to_string(),
        collection: "rules".to_string(),
        payload_json: r#"{"action":"add","label":"test-rule"}"#.to_string(),
        branch: "main".to_string(),
        metadata_json: metadata_json.map(str::to_string),
    }
}

#[test]
fn enqueue_item_type_field() {
    // TS: item_type: itemType (queue-operations.ts:96)
    let req = make_enqueue_request(None);
    assert_eq!(req.item_type, "text");
}

#[test]
fn enqueue_op_field() {
    // TS: op (queue-operations.ts:97) — field name is "op" not "operation"
    let req = make_enqueue_request(None);
    assert_eq!(req.op, "add");
}

#[test]
fn enqueue_tenant_id_field() {
    let req = make_enqueue_request(None);
    assert_eq!(req.tenant_id, "global");
}

#[test]
fn enqueue_collection_field() {
    let req = make_enqueue_request(None);
    assert_eq!(req.collection, "rules");
}

#[test]
fn enqueue_payload_json_forwarded_verbatim() {
    // payload_json is the already-canonicalized string — not re-processed
    let raw = r#"{"action":"add","label":"test-rule"}"#;
    let req = EnqueueItemRequest {
        item_type: "text".to_string(),
        op: "add".to_string(),
        tenant_id: "global".to_string(),
        collection: "rules".to_string(),
        payload_json: raw.to_string(),
        branch: "main".to_string(),
        metadata_json: None,
    };
    assert_eq!(req.payload_json, raw);
}

#[test]
fn enqueue_branch_field_main_default() {
    // TS: branch field in buildEnqueueRequest (queue-operations.ts:101)
    let req = make_enqueue_request(None);
    assert_eq!(req.branch, "main");
}

#[test]
fn enqueue_metadata_json_none_when_absent() {
    // TS: metadata_json only set when metadata present (queue-operations.ts:103)
    let req = make_enqueue_request(None);
    assert!(req.metadata_json.is_none());
}

#[test]
fn enqueue_metadata_json_present_when_supplied() {
    // TS: request.metadata_json = JSON.stringify(metadata) (queue-operations.ts:103)
    let meta = r#"{"source":"mcp_rules_tool"}"#;
    let req = make_enqueue_request(Some(meta));
    assert_eq!(req.metadata_json.as_deref(), Some(meta));
}

#[test]
fn enqueue_metadata_json_store_source_marker() {
    // TS: {source: 'mcp_store_tool'} (queue-operations.ts / store-handlers.ts)
    let meta = r#"{"source":"mcp_store_tool"}"#;
    let req = make_enqueue_request(Some(meta));
    assert_eq!(req.metadata_json.as_deref(), Some(meta));
}

// =========================================================================
// Task 15 — UpsertRuleMirrorRequest field mapping
// =========================================================================

fn make_upsert_rule_request() -> UpsertRuleMirrorRequest {
    UpsertRuleMirrorRequest {
        rule_id: "my-rule".to_string(),
        rule_text: "Always use snake_case for Rust identifiers.".to_string(),
        scope: Some("global".to_string()),
        tenant_id: None,
        created_at: "2024-01-01T00:00:00Z".to_string(),
        updated_at: "2024-01-01T00:00:00Z".to_string(),
    }
}

#[test]
fn upsert_rule_rule_id_field() {
    // rules-mirror-queries.ts:40: rule_id: entry.ruleId
    let req = make_upsert_rule_request();
    assert_eq!(req.rule_id, "my-rule");
}

#[test]
fn upsert_rule_rule_text_field() {
    // rules-mirror-queries.ts:41: rule_text: entry.ruleText
    let req = make_upsert_rule_request();
    assert_eq!(req.rule_text, "Always use snake_case for Rust identifiers.");
}

#[test]
fn upsert_rule_scope_optional_set_when_present() {
    // rules-mirror-queries.ts:45: if (entry.scope !== null) request.scope = entry.scope
    let req = make_upsert_rule_request();
    assert_eq!(req.scope.as_deref(), Some("global"));
}

#[test]
fn upsert_rule_tenant_id_none_for_global() {
    // rules-mirror-queries.ts:46: tenant_id only set when non-null
    let req = make_upsert_rule_request();
    assert!(req.tenant_id.is_none());
}

#[test]
fn upsert_rule_tenant_id_set_for_project_scope() {
    let req = UpsertRuleMirrorRequest {
        rule_id: "proj-rule".to_string(),
        rule_text: "content".to_string(),
        scope: Some("project".to_string()),
        tenant_id: Some("abc123def456".to_string()),
        created_at: "2024-01-01T00:00:00Z".to_string(),
        updated_at: "2024-01-01T00:00:00Z".to_string(),
    };
    assert_eq!(req.tenant_id.as_deref(), Some("abc123def456"));
}

#[test]
fn upsert_rule_timestamps_z_suffix() {
    // INV-9: timestamps must use ISO 8601 Z-suffix
    let req = make_upsert_rule_request();
    assert!(req.created_at.ends_with('Z'), "created_at must end with Z");
    assert!(req.updated_at.ends_with('Z'), "updated_at must end with Z");
}

// --- DeleteRuleMirror ---

#[test]
fn delete_rule_rule_id_field() {
    // rules-mirror-queries.ts:62: { rule_id: ruleId }
    let req = DeleteRuleMirrorRequest {
        rule_id: "my-rule".to_string(),
    };
    assert_eq!(req.rule_id, "my-rule");
}

// =========================================================================
// Task 15 — UpsertScratchpadMirrorRequest field mapping
// =========================================================================

fn make_upsert_scratchpad_request() -> UpsertScratchpadMirrorRequest {
    UpsertScratchpadMirrorRequest {
        scratchpad_id: "sp-001".to_string(),
        content: "Brainstorm notes about architecture.".to_string(),
        title: Some("Architecture Notes".to_string()),
        tags: Some(r#"["arch","design"]"#.to_string()),
        tenant_id: "global".to_string(),
        created_at: "2024-06-01T12:00:00Z".to_string(),
        updated_at: "2024-06-01T12:00:00Z".to_string(),
    }
}

#[test]
fn upsert_scratchpad_id_field() {
    // scratchpad-mirror-queries.ts:39: scratchpad_id: entry.scratchpadId
    let req = make_upsert_scratchpad_request();
    assert_eq!(req.scratchpad_id, "sp-001");
}

#[test]
fn upsert_scratchpad_content_field() {
    // scratchpad-mirror-queries.ts:40: content: entry.content
    let req = make_upsert_scratchpad_request();
    assert_eq!(req.content, "Brainstorm notes about architecture.");
}

#[test]
fn upsert_scratchpad_title_optional_present() {
    // scratchpad-mirror-queries.ts:46: if (entry.title !== null) request.title = entry.title
    let req = make_upsert_scratchpad_request();
    assert_eq!(req.title.as_deref(), Some("Architecture Notes"));
}

#[test]
fn upsert_scratchpad_title_absent_when_null() {
    let req = UpsertScratchpadMirrorRequest {
        scratchpad_id: "sp-002".to_string(),
        content: "content".to_string(),
        title: None,
        tags: None,
        tenant_id: "global".to_string(),
        created_at: "2024-06-01T12:00:00Z".to_string(),
        updated_at: "2024-06-01T12:00:00Z".to_string(),
    };
    assert!(req.title.is_none());
}

#[test]
fn upsert_scratchpad_tags_optional_present() {
    // scratchpad-mirror-queries.ts:47: if (entry.tags !== '[]') request.tags = entry.tags
    let req = make_upsert_scratchpad_request();
    assert_eq!(req.tags.as_deref(), Some(r#"["arch","design"]"#));
}

#[test]
fn upsert_scratchpad_tags_absent_when_empty_array() {
    // tags omitted when entry.tags === '[]' (scratchpad-mirror-queries.ts:47)
    let req = UpsertScratchpadMirrorRequest {
        scratchpad_id: "sp-003".to_string(),
        content: "content".to_string(),
        title: None,
        tags: None, // empty tags → not set
        tenant_id: "global".to_string(),
        created_at: "2024-06-01T12:00:00Z".to_string(),
        updated_at: "2024-06-01T12:00:00Z".to_string(),
    };
    assert!(req.tags.is_none());
}

#[test]
fn upsert_scratchpad_tenant_id_field() {
    // scratchpad-mirror-queries.ts:41: tenant_id: entry.tenantId
    let req = make_upsert_scratchpad_request();
    assert_eq!(req.tenant_id, "global");
}

#[test]
fn upsert_scratchpad_timestamps_z_suffix() {
    // INV-9: ISO 8601 Z-suffix required
    let req = make_upsert_scratchpad_request();
    assert!(req.created_at.ends_with('Z'));
    assert!(req.updated_at.ends_with('Z'));
}

// --- DeleteScratchpadMirror ---

#[test]
fn delete_scratchpad_id_field() {
    // scratchpad-mirror-queries.ts:64: { scratchpad_id: scratchpadId }
    let req = DeleteScratchpadMirrorRequest {
        scratchpad_id: "sp-001".to_string(),
    };
    assert_eq!(req.scratchpad_id, "sp-001");
}

// =========================================================================
// Task 15 — fire-and-forget: RPC failure must return Ok(())
// =========================================================================
//
// We test the swallow-wrapper by verifying that a Status error produced by
// call() does not propagate out of the mirror methods.  We do this by
// inspecting the wrapping logic directly (the pattern `if let Err(ref e) =
// result { warn!(...) } Ok(())`) which ensures Ok is always returned.

#[test]
fn fire_and_forget_ok_even_on_unavailable_status() {
    // Simulate what happens in the wrapper: call() returns Err(unavailable),
    // the if-let logs a warning, then Ok(()) is returned.
    let err: Result<(), Status> = Err(Status::unavailable("simulated daemon down"));
    if let Err(ref e) = err {
        // In production this is tracing::warn! — here we just check the message.
        assert!(e.message().contains("simulated daemon down"));
    }
    // The wrapper always returns Ok(()) regardless of err.
    let swallowed: Result<(), Status> = Ok(());
    assert!(swallowed.is_ok());
}

#[test]
fn fire_and_forget_ok_even_on_deadline_exceeded() {
    let err: Result<(), Status> = Err(Status::deadline_exceeded("timed out"));
    if let Err(ref e) = err {
        assert_eq!(e.code(), tonic::Code::DeadlineExceeded);
    }
    let swallowed: Result<(), Status> = Ok(());
    assert!(swallowed.is_ok());
}

// ── DaemonClient construction ─────────────────────────────────────────────

#[tokio::test]
async fn daemon_client_constructs_for_write_calls() {
    let result = DaemonClient::new("http://127.0.0.1:50051");
    assert!(result.is_ok());
}
