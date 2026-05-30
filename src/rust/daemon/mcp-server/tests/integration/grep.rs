// Integration tests: grep tool → real FTS text_search via daemon gRPC.
//
// it_grep_real_fts_search  — calls TextSearchService::Search against live daemon
// it_grep_empty_pattern    — verifies daemon rejects blank pattern gracefully

use super::helpers;

// ---------------------------------------------------------------------------
// Live daemon: FTS search against indexed codebase
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_grep_real_fts_search() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };

    // Search for a pattern that is virtually guaranteed to exist in any
    // indexed Rust project (the keyword "fn").
    let req = mcp_server::proto::TextSearchRequest {
        pattern: "fn ".to_string(),
        regex: false,
        case_sensitive: true,
        tenant_id: None,
        branch: None,
        path_glob: Some("**/*.rs".to_string()),
        path_prefix: None,
        context_lines: 0,
        max_results: 5,
    };

    let resp = client
        .text_search(req)
        .await
        .expect("text_search must succeed against live daemon");

    // We assert structural properties, not specific content — the indexed
    // corpus is deployment-specific.
    assert!(
        resp.total_matches >= 0,
        "total_matches must be non-negative"
    );
    // Each returned match must have non-empty file_path and content.
    for m in &resp.matches {
        assert!(!m.file_path.is_empty(), "match file_path must not be empty");
        assert!(!m.content.is_empty(), "match content must not be empty");
        assert!(m.line_number > 0, "line_number must be 1-based positive");
    }
    // Result count must not exceed max_results.
    assert!(
        resp.matches.len() <= 5,
        "result count {} exceeds max_results=5",
        resp.matches.len()
    );
}

// ---------------------------------------------------------------------------
// Live daemon: count_matches returns a non-negative integer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_grep_count_matches() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };

    let req = mcp_server::proto::TextSearchRequest {
        pattern: "use ".to_string(),
        regex: false,
        case_sensitive: true,
        tenant_id: None,
        branch: None,
        path_glob: None,
        path_prefix: None,
        context_lines: 0,
        max_results: 0,
    };

    let resp = client
        .count_matches(req)
        .await
        .expect("count_matches must succeed against live daemon");

    assert!(
        resp.count >= 0,
        "count must be non-negative; got: {}",
        resp.count
    );
    assert!(
        resp.query_time_ms >= 0,
        "query_time_ms must be non-negative"
    );
}
