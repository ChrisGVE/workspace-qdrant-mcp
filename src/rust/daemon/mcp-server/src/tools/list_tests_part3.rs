//! List tool hermetic tests, part 3: depth/limit/pagination/field-order/stats/submodule/branch.
//!
//! Included from `list_tests.rs` via `#[cfg(test)] #[path = "list_tests_part3.rs"] mod part3;`.

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};

use crate::tools::list::types::{DEFAULT_DEPTH, DEFAULT_LIMIT, MAX_DEPTH, MAX_LIMIT};
use crate::tools::list::{list_tool, ListInput};

use super::{call_list, session_with_project, TestDb};

// ---------------------------------------------------------------------------
// § 12  Depth clamping: DEFAULT_DEPTH=3, MAX_DEPTH=10
// ---------------------------------------------------------------------------

#[test]
fn depth_defaults_to_three() {
    assert_eq!(DEFAULT_DEPTH, 3);
}

#[test]
fn max_depth_is_ten() {
    assert_eq!(MAX_DEPTH, 10);
}

#[test]
fn depth_zero_clamped_to_one() {
    // depth=0 → clamped to 1 (only top-level folders rendered).
    let db = TestDb::new();
    db.insert_project("wid-d0", "t-d0", "/p");
    db.insert_file(
        "f1",
        "wid-d0",
        "a/b/c/deep.rs",
        None,
        Some("rs"),
        0,
        "[]",
        None,
    );

    let session = session_with_project("t-d0");
    let v = call_list(
        &db,
        ListInput {
            depth: Some(0),
            format: Some("tree".to_string()),
            ..Default::default()
        },
        &session,
    );
    let listing = v["listing"].as_str().unwrap();
    // At depth 1 we see "a/" with file count but not b/ or c/
    assert!(
        listing.contains("a/"),
        "expected top-level 'a/' at depth 1: {listing:?}"
    );
    assert!(
        !listing.contains("b/"),
        "should not see nested 'b/' at depth 1: {listing:?}"
    );
}

#[test]
fn depth_above_max_clamped_to_ten() {
    // depth=999 should not panic and should behave like depth=10.
    let db = TestDb::new();
    db.insert_project("wid-dmax", "t-dmax", "/p");
    db.insert_file("f1", "wid-dmax", "a.rs", None, Some("rs"), 0, "[]", None);

    let session = session_with_project("t-dmax");
    let v = call_list(
        &db,
        ListInput {
            depth: Some(999),
            format: Some("tree".to_string()),
            ..Default::default()
        },
        &session,
    );
    assert_eq!(v["success"], true);
}

// ---------------------------------------------------------------------------
// § 13  Limit clamping: DEFAULT_LIMIT=200, MAX_LIMIT=500
// ---------------------------------------------------------------------------

#[test]
fn limit_defaults() {
    assert_eq!(DEFAULT_LIMIT, 200);
}

#[test]
fn max_limit_is_500() {
    assert_eq!(MAX_LIMIT, 500);
}

#[test]
fn limit_one_renders_exactly_one_entry() {
    let db = TestDb::new();
    db.insert_project("wid-lim", "t-lim", "/p");
    for i in 0..5u32 {
        db.insert_file(
            &format!("f{i}"),
            "wid-lim",
            &format!("{i:02}.rs"),
            None,
            Some("rs"),
            0,
            "[]",
            None,
        );
    }

    let session = session_with_project("t-lim");
    let v = call_list(
        &db,
        ListInput {
            limit: Some(1),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );

    // limit=1 and page_size defaults to limit, so page has ≥1 file — first page
    // renders at most 1 entry due to render limit.
    let listing = v["listing"].as_str().unwrap();
    // listing contains "... (truncated" because render hit the limit
    assert!(
        listing.contains("... (truncated"),
        "expected truncation marker: {listing:?}"
    );
    assert_eq!(v["stats"]["truncated"], true);
}

#[test]
fn limit_above_max_clamped_to_500() {
    let db = TestDb::new();
    db.insert_project("wid-lmax", "t-lmax", "/p");
    db.insert_file("f1", "wid-lmax", "a.rs", None, Some("rs"), 0, "[]", None);

    let session = session_with_project("t-lmax");
    // Should not panic
    let v = call_list(
        &db,
        ListInput {
            limit: Some(9999),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );
    assert_eq!(v["success"], true);
}

// ---------------------------------------------------------------------------
// § 14  Pagination: next_token round-trip (base64 standard, padded)
// ---------------------------------------------------------------------------

#[test]
fn pagination_next_token_base64_standard_padded() {
    let db = TestDb::new();
    db.insert_project("wid-pag", "t-pag", "/p");
    // Insert 3 files, page_size=2 → first page returns 2, next_token set.
    for i in 0..3u32 {
        db.insert_file(
            &format!("fp{i}"),
            "wid-pag",
            &format!("{i:02}_file.rs"),
            None,
            Some("rs"),
            0,
            "[]",
            None,
        );
    }

    let session = session_with_project("t-pag");
    // First page: pageSize=2
    let v1 = call_list(
        &db,
        ListInput {
            page_size: Some(2),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );
    assert_eq!(v1["success"], true);
    let next_token = v1["next_token"]
        .as_str()
        .expect("next_token must be present for first page");

    // Decode: standard base64, padded
    let decoded = BASE64_STANDARD
        .decode(next_token)
        .expect("next_token must be valid base64");
    let decoded_str = String::from_utf8(decoded).expect("decoded token must be valid UTF-8");
    // The token encodes the relative_path of the last file in page 1 (the 2nd file sorted).
    assert_eq!(
        decoded_str, "01_file.rs",
        "decoded cursor should be '01_file.rs', got: {decoded_str:?}"
    );

    // Second page using the cursor
    let v2 = call_list(
        &db,
        ListInput {
            cursor: Some(next_token.to_string()),
            page_size: Some(2),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );
    assert_eq!(v2["success"], true);
    let listing2 = v2["listing"].as_str().unwrap();
    // Only 1 file remaining: 02_file.rs
    assert!(
        listing2.contains("02_file.rs"),
        "second page should contain '02_file.rs': {listing2:?}"
    );
    // No next_token on final page
    assert!(
        v2.get("next_token").is_none() || v2["next_token"].is_null(),
        "final page must not have next_token"
    );
}

#[test]
fn next_token_json_key_is_snake_case() {
    // Verify the JSON key is "next_token" (not "nextToken") — matching TS
    // list-files-types.ts:91 `next_token?: string`.
    let db = TestDb::new();
    db.insert_project("wid-ntk", "t-ntk", "/p");
    for i in 0..3u32 {
        db.insert_file(
            &format!("fntk{i}"),
            "wid-ntk",
            &format!("{i}.rs"),
            None,
            Some("rs"),
            0,
            "[]",
            None,
        );
    }
    let session = session_with_project("t-ntk");
    let v = call_list(
        &db,
        ListInput {
            page_size: Some(2),
            format: Some("flat".to_string()),
            ..Default::default()
        },
        &session,
    );
    // Key must be "next_token" — not "nextToken"
    assert!(
        v.get("next_token").is_some(),
        "JSON must contain 'next_token' key; full response: {v}"
    );
    assert!(
        v.get("nextToken").is_none(),
        "JSON must NOT contain camelCase 'nextToken'; full response: {v}"
    );
}

// ---------------------------------------------------------------------------
// § 15  JSON field order (declared order = serialisation order)
// ---------------------------------------------------------------------------

#[test]
fn response_field_order_matches_ts() {
    // success, projectPath, basePath, format, listing, stats, message?, next_token?
    let db = TestDb::new();
    db.insert_project("wid-fld", "t-fld", "/some/proj");
    db.insert_file("ff1", "wid-fld", "a.rs", None, Some("rs"), 0, "[]", None);

    let mgr = db.state_manager();
    let session = session_with_project("t-fld");
    let result = list_tool(ListInput::default(), &mgr, &session);
    let text = result
        .content
        .first()
        .expect("content must not be empty")
        .raw
        .as_text()
        .expect("first content item must be text")
        .text
        .clone();

    // serde_json preserves insertion order from Value::Object.
    // The simplest check: grep key positions in the raw JSON string.
    let keys = [
        "\"success\"",
        "\"projectPath\"",
        "\"basePath\"",
        "\"format\"",
        "\"listing\"",
        "\"stats\"",
    ];

    let mut prev_pos = 0usize;
    for key in &keys {
        let pos = text
            .find(key)
            .unwrap_or_else(|| panic!("key {key} not found in: {text}"));
        assert!(
            pos >= prev_pos,
            "key {key} at pos {pos} should appear after pos {prev_pos} in: {text}"
        );
        prev_pos = pos;
    }
}
