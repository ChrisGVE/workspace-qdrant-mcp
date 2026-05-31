//! GENERATED per-case error-parity wrappers for `grep_inband` (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One test per corpus row.

use super::error_asserts::assert_grep_inband;

#[tokio::test]
async fn egi_pattern_missing() {
    assert_grep_inband("pattern_missing").await;
}

#[tokio::test]
async fn egi_pattern_null() {
    assert_grep_inband("pattern_null").await;
}

#[tokio::test]
async fn egi_pattern_empty() {
    assert_grep_inband("pattern_empty").await;
}
