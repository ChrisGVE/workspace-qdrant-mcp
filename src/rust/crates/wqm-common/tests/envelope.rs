//! N12 envelope contract tests -- the sealed seven-key shape (MCP-SURFACE.md §3.1)
//! and the two closed vocabularies (§4.3 errors, §4.4 notices).
//!
//! These assert the WIRE, not the Rust: every check goes through `serde_json` and
//! looks at the object an agent actually receives, because that is what the sealed
//! document specifies.

use wqm_common::envelope::{Envelope, ErrorCode, Notice, NoticeCode, Severity, ToolError};

#[test]
fn a_success_envelope_has_all_seven_keys_with_error_null() {
    let env = Envelope::success("status", 3, serde_json::json!({"a": 1}));
    let v = serde_json::to_value(&env).unwrap();
    let obj = v.as_object().unwrap();

    for key in [
        "ok",
        "tool",
        "elapsed_ms",
        "defaults_applied",
        "notices",
        "data",
        "error",
    ] {
        assert!(obj.contains_key(key), "envelope is missing `{key}`");
    }
    assert_eq!(obj.len(), 7, "the envelope has exactly seven keys");
    assert_eq!(obj["ok"], serde_json::json!(true));
    assert_eq!(obj["error"], serde_json::Value::Null);
    // Empty, not absent: the positive statement "nothing was narrowed for you".
    assert_eq!(obj["defaults_applied"], serde_json::json!([]));
    assert_eq!(obj["notices"], serde_json::json!([]));
}

#[test]
fn a_failure_envelope_has_the_same_seven_keys_with_data_null() {
    let env = Envelope::failure(
        "query",
        1,
        ToolError {
            code: ErrorCode::Internal,
            message: "something went wrong".to_owned(),
            details: serde_json::json!({"correlation_id": "abc"}),
            retryable: false,
        },
    );
    let v = serde_json::to_value(&env).unwrap();
    let obj = v.as_object().unwrap();

    assert_eq!(obj.len(), 7, "failure carries the same seven keys");
    assert_eq!(obj["ok"], serde_json::json!(false));
    assert_eq!(obj["data"], serde_json::Value::Null);
    assert_eq!(obj["error"]["code"], serde_json::json!("internal"));
    assert_eq!(obj["error"]["retryable"], serde_json::json!(false));
}

#[test]
fn the_closed_vocabularies_serialize_as_the_sealed_strings() {
    assert_eq!(
        serde_json::to_value(NoticeCode::ProtocolDowngraded).unwrap(),
        serde_json::json!("protocol_downgraded")
    );
    assert_eq!(
        serde_json::to_value(NoticeCode::SchemaBudgetTruncated).unwrap(),
        serde_json::json!("schema_budget_truncated")
    );
    assert_eq!(
        serde_json::to_value(ErrorCode::BackendUnavailable).unwrap(),
        serde_json::json!("backend_unavailable")
    );
    assert_eq!(
        serde_json::to_value(Severity::Warn).unwrap(),
        serde_json::json!("warn")
    );
}

#[test]
fn a_notice_lands_in_the_channel_that_is_always_present() {
    let env = Envelope::success("status", 0, serde_json::json!({})).with_notice(Notice {
        code: NoticeCode::ProtocolDowngraded,
        severity: Severity::Warn,
        message: "served 2026-07-28 instead of 2099-01-01".to_owned(),
        details: serde_json::json!({"requested": "2099-01-01", "served": "2026-07-28"}),
    });
    let v = serde_json::to_value(&env).unwrap();
    assert_eq!(v["notices"].as_array().unwrap().len(), 1);
    assert_eq!(
        v["notices"][0]["code"],
        serde_json::json!("protocol_downgraded")
    );
}
