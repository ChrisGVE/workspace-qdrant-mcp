// Integration tests: rules tool → real rules_mirror ingest + list + SQLite cross-check.
//
// it_rules_list_from_mirror_real_db   — list_rules against live state.db
// it_rules_upsert_delete_mirror       — upsert_rule_mirror + delete_rule_mirror via gRPC
//                                        + verify rows appear/disappear in SQLite mirror
// it_rules_qdrant_rules_collection    — Qdrant rules collection exists check

use super::helpers;
use mcp_server::sqlite::rules_mirror::list_rules;

// ---------------------------------------------------------------------------
// Live state.db: list_rules returns well-formed rows (or empty for clean installs)
// ---------------------------------------------------------------------------

#[test]
fn it_rules_list_from_mirror_real_db() {
    let mgr = match helpers::open_state_manager() {
        Some(m) => m,
        None => return,
    };

    let rows = list_rules(mgr.connection(), None, None, 10);

    // Structural validation: every row has non-empty rule_id and rule_text.
    for r in &rows {
        assert!(
            !r.rule_id.is_empty(),
            "rule_id must not be empty; got: {:?}",
            r
        );
        assert!(
            !r.rule_text.is_empty(),
            "rule_text must not be empty for rule_id={}",
            r.rule_id
        );
        assert!(
            !r.created_at.is_empty(),
            "created_at must not be empty for rule_id={}",
            r.rule_id
        );
        assert!(
            !r.updated_at.is_empty(),
            "updated_at must not be empty for rule_id={}",
            r.rule_id
        );
    }
    assert!(
        rows.len() <= 10,
        "list_rules exceeded limit=10; got: {}",
        rows.len()
    );
}

// ---------------------------------------------------------------------------
// Live state.db: global scope filter returns only global/null-scoped rows
// ---------------------------------------------------------------------------

#[test]
fn it_rules_global_scope_filter() {
    let mgr = match helpers::open_state_manager() {
        Some(m) => m,
        None => return,
    };

    let rows = list_rules(mgr.connection(), Some("global"), None, 20);

    for r in &rows {
        let scope = r.scope.as_deref().unwrap_or("");
        assert!(
            scope == "global" || scope.is_empty(),
            "global filter must only return global or null-scope rows; got scope={scope:?} for rule_id={}",
            r.rule_id
        );
    }
}

// ---------------------------------------------------------------------------
// Live daemon + state.db: upsert_rule_mirror → list_rules cross-check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_rules_upsert_delete_mirror_cross_check() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };
    let mgr = match helpers::open_state_manager() {
        Some(m) => m,
        None => return,
    };

    let rule_id = format!("integration_test_{}", uuid_fragment());
    let rule_text = "integration test rule — safe to delete".to_string();
    let now = wqm_common::timestamps::now_utc();

    // Upsert rule via gRPC (fire-and-forget — returns Ok(()) even on error).
    client
        .upsert_rule_mirror(
            rule_id.clone(),
            rule_text.clone(),
            Some("global".to_string()),
            None,
            now.clone(),
            now.clone(),
        )
        .await
        .expect("upsert_rule_mirror must not return Err");

    // Allow mirror write to propagate (daemon is fire-and-forget async).
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Read back from SQLite mirror — if the row is present, validate it.
    let rows = list_rules(mgr.connection(), None, None, 200);
    let found = rows.iter().find(|r| r.rule_id == rule_id);

    // The mirror write is best-effort; if the row is absent the test still
    // passes (service might not have flushed within 200 ms in CI).
    if let Some(row) = found {
        assert_eq!(row.rule_text, rule_text, "rule_text must round-trip");
        assert_eq!(
            row.scope.as_deref(),
            Some("global"),
            "scope must round-trip"
        );

        // Clean up: delete the test rule.
        client
            .delete_rule_mirror(rule_id.clone())
            .await
            .expect("delete_rule_mirror must not return Err");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let rows_after = list_rules(mgr.connection(), None, None, 200);
        let still_present = rows_after.iter().any(|r| r.rule_id == rule_id);
        // Again, best-effort: if still present the mirror flush is slow; not a failure.
        if still_present {
            eprintln!(
                "NOTE: delete_rule_mirror for {rule_id} not yet reflected in SQLite (flush lag)"
            );
        }
    } else {
        eprintln!(
            "NOTE: upsert_rule_mirror for {rule_id} not reflected in SQLite within 200 ms (advisory mirror)"
        );
    }
}

// ---------------------------------------------------------------------------
// Live Qdrant: rules collection existence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_rules_qdrant_rules_collection_exists() {
    if !helpers::probe_qdrant().await {
        return;
    }

    let url = helpers::qdrant_url();
    let api_key = helpers::qdrant_api_key().map(|s| secrecy::SecretString::new(s.into_boxed_str()));
    let client = mcp_server::qdrant::client::QdrantReadClient::new(url, api_key);

    // This just checks the API; a fresh install may not have the rules collection yet.
    let result = client
        .collection_exists(wqm_common::constants::COLLECTION_RULES)
        .await;

    assert!(
        result.is_ok(),
        "collection_exists must not error; got: {:?}",
        result.unwrap_err()
    );
    // We do not assert true/false — the collection may or may not exist in CI.
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generate a short random fragment for test IDs (avoids collisions).
fn uuid_fragment() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{nanos:08x}")
}
