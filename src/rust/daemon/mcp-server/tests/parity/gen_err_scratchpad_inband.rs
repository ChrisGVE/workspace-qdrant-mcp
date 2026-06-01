//! GENERATED per-case error-parity wrappers for `scratchpad_inband` (do not edit by hand).
//! Regenerate via the tmp capture+codegen harness. One test per corpus row.

use super::error_asserts::assert_scratchpad_inband;

#[tokio::test]
async fn esc_content_missing() {
    assert_scratchpad_inband("content_missing").await;
}

#[tokio::test]
async fn esc_content_null() {
    assert_scratchpad_inband("content_null").await;
}

#[tokio::test]
async fn esc_content_empty() {
    assert_scratchpad_inband("content_empty").await;
}

#[tokio::test]
async fn esc_content_ws() {
    assert_scratchpad_inband("content_ws").await;
}

#[tokio::test]
async fn esc_content_tab() {
    assert_scratchpad_inband("content_tab").await;
}
